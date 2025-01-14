#################################################################
#### Universal Prediction Single-, Multi-Word, and Sentences ####
#################################################################

import os
os.chdir('/content/drive/MyDrive/Emotion_CLIP')
print(f"Current working directory: {os.getcwd()}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from CLIP import clip
from EmotionCLIP.src.models.base import EmotionCLIP
import pickle
import langdetect
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load translation model and tokenizer globally
print("Loading M2M100 model and tokenizer...")
model_name = "facebook/m2m100_1.2B"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
translation_model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)

########################################################
#  Extra helper to decide if an input is "a sentence"  #
########################################################

def is_sentence(text):
    """
    Decide if 'text' is considered a full sentence 
    (rather than a single or multiword expression).
    You can use any heuristic you wish here.
    I use: if it has more than 8 words, call it a sentence.
    """
    words = text.strip().split()
    if len(words) > 8:  # or some other threshold
        return True
    return False

########################################################
#   Helper: get token-level embeddings from CLIP text  #
########################################################

def get_clip_token_embeddings(text, clip_model):
    """
    Obtain token-level embeddings from CLIP’s text transformer.
    We will then re-fuse these tokens back into their original words.
    Returns:
      word_embeddings = list of torch.FloatTensor (one vector per original word)
    """

    # 1) Tokenize (returns shape [1, n_ctx], with special tokens)
    text_tokens = clip.tokenize([text], truncate=True).to(device)  # batch_size=1
    # text_tokens[0] is the list of token ids, including start/end tokens.

    with torch.no_grad():
        # (A) Embed tokens
        # shape = (1, n_ctx, d_model)
        token_embeds = clip_model.token_embedding(text_tokens)  

        # (B) Add positional embeddings
        token_embeds = token_embeds + clip_model.positional_embedding[:token_embeds.shape[1], :]

        # (C) Transformer blocks
        x = token_embeds.permute(1, 0, 2)  # (seq_len, batch=1, d_model)
        for block in clip_model.transformer.resblocks:
            x = block(x)

        # (D) Final layer norm
        x = x.permute(1, 0, 2)  # back to (batch=1, seq_len, d_model)
        x = clip_model.ln_final(x)  # shape: (1, seq_len, d_model)

        # (E) Project to text_projection if you want them in the same dimensional space
        # as normal clip embeddings. This is optional, but often done:
        token_level_embeds = x[0] @ clip_model.text_projection  
        # shape: (seq_len, text_projection_dim)

    # Now, token_level_embeds is a sequence of subword embeddings. 
    # We skip the first token (start_of_text) and last token (end_of_text).
    # Also need to re-fuse subwords belonging to the same “word”.

    # Convert token IDs back to actual string tokens to see where to fuse
    # clip.tokenize uses a simple BPE. We can do:
    tokens_str = clip.tokenizer.decode(text_tokens[0])
    # But we want them individually:
    # Instead, we can decode them one by one, or do something simpler.
    # We'll do a minimal approach: skip the first and last special tokens.
    subword_embeddings = token_level_embeds[1:-1]  # remove <|startoftext|> <|endoftext|>

    # We can do a naive approach: split on whitespace at the original string 
    # and try to guess how many subwords belong to each word. 
    # That can be tricky in BPE. A more robust way is to decode each subword. 
    # I found that this simplified version works well:

    # A. Grab the batch of subword IDs (minus specials)
    subword_ids = text_tokens[0, 1:-1].tolist()  
    subword_texts = [clip.tokenizer.decoder[sid] for sid in subword_ids]

    # B. Re-fuse: if a subword starts with a 'Ġ' or a whitespace, we assume it's a new word
    # or if it’s BPE-split. This part is heuristic and worked well. Adjust as needed.

    word_embeddings = []
    current_tokens = []
    for i, sw in enumerate(subword_texts):
        # sw might look like "ing", "Ġembedding", or punctuation. 
        # If it starts with whitespace or 'Ġ', we treat it as a new word boundary.
        # Or if it doesn't start with that, it's continuing the same word.
        # (OpenAI's CLIP BPE can differ from RoBERTa or other models.)

        # We check if there's a leading space in the actual decoded text:
        # (In some tokenization schemes, CLIP might store ' hello' vs 'hello'.)
        if (i == 0) or sw.startswith(' ') or sw.startswith('Ġ'):
            # finalize the previous tokens if any
            if len(current_tokens) > 0:
                # average them:
                word_embeddings.append(torch.mean(torch.stack(current_tokens), dim=0))
                current_tokens = []
            # start new group
            current_tokens = [subword_embeddings[i]]
        else:
            # continue the same group
            current_tokens.append(subword_embeddings[i])

    # finalize any leftover
    if len(current_tokens) > 0:
        word_embeddings.append(torch.mean(torch.stack(current_tokens), dim=0))

    return word_embeddings

########################################################
#                 The Regressor Classes                #
########################################################

class DeepRegressor(nn.Module):
    def __init__(self, input_dim):
        super(DeepRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze()

class ImprovedEnsemble:
    def __init__(self, classifier_threshold=0.6):
        self.device = device
        self.classifier_threshold = classifier_threshold
        self.scaler_og = StandardScaler()
        self.scaler_emotion = StandardScaler()
        self.output_scaler = StandardScaler()
        self.combined_regressor = None

    def constrain_predictions(self, predictions):
        return torch.clamp(predictions, min=1.0, max=5.0)

    def normalize_predictions(self, predictions, target_mean=3.0, target_std=1.0):
        pred_mean = torch.mean(predictions)
        pred_std = torch.std(predictions)
        normalized = (predictions - pred_mean) / pred_std
        return normalized * target_std + target_mean

    def fit(self, concrete_words, concrete_ratings, abstract_words, abstract_ratings):
        print("Getting embeddings...")
        all_words = concrete_words + abstract_words
        og_embeddings = get_original_clip_embeddings(all_words)
        emotion_embeddings = get_finetuned_clip_embeddings(all_words)

        combined_embeddings_np = np.concatenate([og_embeddings.cpu().numpy(), emotion_embeddings.cpu().numpy()], axis=1)
        combined_embeddings_scaled = self.scaler_og.fit_transform(combined_embeddings_np)
        combined_embeddings_scaled = torch.FloatTensor(combined_embeddings_scaled).to(self.device)

        all_ratings = torch.cat([concrete_ratings, abstract_ratings]).to(self.device)
        print("\nTraining combined regressor...")
        self.combined_regressor = self._train_deep_regressor(
            combined_embeddings_scaled,
            all_ratings,
            "Combined Regressor"
        )

    def evaluate_and_plot(self, test_words, true_ratings, max_r2=None):
        predictions = self.predict(test_words)
        predictions = predictions.cpu()
        true_ratings = true_ratings.cpu()

        r2 = r2_score(true_ratings.numpy(), predictions.numpy())
        correlation, _ = pearsonr(predictions.numpy(), true_ratings.numpy())

        r2_display = f"R²: {r2:.2f}"
        if max_r2 is not None:
            r2_display += f" ({(r2/max_r2*100):.1f}% of human reliability)"

        # Scatter Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(true_ratings, predictions, alpha=0.7, edgecolors='k', label=r2_display)
        plt.plot([1, 5], [1, 5], 'r--', label="Ideal Fit (y=x)")
        plt.xlabel("True Ratings")
        plt.ylabel("Predicted Ratings")
        plt.title("Model Predictions vs True Ratings")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Generate additional plots (residuals, histogram, density)
        self._generate_additional_plots(true_ratings, predictions)

    def _generate_additional_plots(self, true_ratings, predictions):
        # Residual Plot
        residuals = predictions - true_ratings
        plt.figure(figsize=(8, 6))
        plt.scatter(true_ratings, residuals, alpha=0.7, edgecolors='k')
        plt.axhline(0, color='r', linestyle='--', label="Zero Residual Line")
        plt.xlabel("True Ratings")
        plt.ylabel("Residuals (Predicted - True)")
        plt.title("Residual Plot")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Histogram of Residuals
        plt.figure(figsize=(8, 6))
        plt.hist(residuals.numpy(), bins=15, alpha=0.7, edgecolor='k')
        plt.axvline(0, color='r', linestyle='--', label="Mean Residual")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.title("Distribution of Residuals")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Density Plot
        plt.figure(figsize=(8, 6))
        sns.kdeplot(true_ratings.numpy(), label="True Ratings", fill=True, alpha=0.5)
        sns.kdeplot(predictions.numpy(), label="Predicted Ratings", fill=True, alpha=0.5)
        plt.xlabel("Rating Value")
        plt.ylabel("Density")
        plt.title("Density Plot of Predicted vs True Ratings")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _train_deep_regressor(self, X, y, name="", lr=0.0005):
        model = DeepRegressor(X.shape[1]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_loss = float('inf')
        best_model_state = None
        patience = 5
        patience_counter = 0

        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            outputs = model(X)
            mse_loss = criterion(outputs, y)
            total_loss = mse_loss
            total_loss.backward()
            optimizer.step()

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"{name} - Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % 20 == 0:
                print(f'{name} - Epoch [{epoch+1}/100], MSE Loss: {mse_loss.item():.4f}')

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        return model

    def predict(self, words):
        """
        For an input list of words (strings), run them through the original or emotion CLIP 
        and get a single embedding for each, then pass to the regressor. 
        This has NOT changed from the original pipeline.
        """
        og_embeddings = get_original_clip_embeddings(words)
        emotion_embeddings = get_finetuned_clip_embeddings(words)

        combined_embeddings_np = np.concatenate([og_embeddings.cpu().numpy(), emotion_embeddings.cpu().numpy()], axis=1)
        combined_embeddings_scaled = self.scaler_og.transform(combined_embeddings_np)
        combined_embeddings_scaled = torch.FloatTensor(combined_embeddings_scaled).to(self.device)

        with torch.no_grad():
            predictions = self.combined_regressor(combined_embeddings_scaled)
            predictions = self.constrain_predictions(predictions)
            predictions = self.normalize_predictions(predictions)

        return predictions

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.combined_regressor.state_dict(), os.path.join(path, "combined_regressor.pth"))
        with open(os.path.join(path, "scalers.pkl"), "wb") as f:
            pickle.dump({
                'scaler_og': self.scaler_og,
                'scaler_emotion': self.scaler_emotion,
                'output_scaler': self.output_scaler
            }, f)
        print(f"Model and scalers saved to {path}.")

    @staticmethod
    def load_model(path):
        instance = ImprovedEnsemble()
        regressor_state_dict = torch.load(os.path.join(path, "combined_regressor.pth"), map_location=device)
        instance.combined_regressor = DeepRegressor(
            input_dim=regressor_state_dict['model.0.weight'].size(1)
        ).to(device)
        instance.combined_regressor.load_state_dict(regressor_state_dict)

        with open(os.path.join(path, "scalers.pkl"), "rb") as f:
            scalers = pickle.load(f)
            instance.scaler_og = scalers['scaler_og']
            instance.scaler_emotion = scalers['scaler_emotion']
            instance.output_scaler = scalers['output_scaler']

        print(f"Model and scalers loaded from {path}.")
        return instance

########################################################
#                    Translation etc.                  #
########################################################

def clean_translation(translation):
    if '...' in translation:
        parts = [p.strip() for p in translation.split('...') if p.strip()]
        if parts:
            translation = parts[0]

    parts = [p.strip() for p in translation.split(',')]
    parts = [p.strip() for p in ' '.join(parts).split('–')]
    parts = [p.strip() for p in ' '.join(parts).split('-')]
    parts = [p.strip() for p in ' '.join(parts).split('and')]

    unique_parts = []
    for part in parts:
        part = part.strip().strip('.,;')
        if part and part not in unique_parts:
            unique_parts.append(part)

    if unique_parts:
        translation = unique_parts[0]

    translation = translation.lower()
    translation = translation.replace('the ', '').replace(' the', '')
    translation = translation.replace('a ', '').replace(' a', '')
    translation = translation.replace('an ', '').replace(' an', '')
    translation = translation.replace('to ', '').replace(' to', '')

    translation = translation.strip('.,;:-–—()[]{}')

    return translation.strip()

def translate_to_english(text):
    try:
        detected_lang = langdetect.detect(text)
        print(f"Detected language for '{text}': {detected_lang}")  # Debug print

        if detected_lang != 'en':
            text_to_translate = text.split('...')[0] if '...' in text else text

            tokenizer.src_lang = detected_lang

            encoded = tokenizer(text_to_translate, return_tensors="pt").to(device)
            generated_tokens = translation_model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.get_lang_id("en"),
                max_length=128,
                num_beams=5,
                num_return_sequences=1
            )

            translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            cleaned_translation = clean_translation(translation)

            if not cleaned_translation or cleaned_translation.isspace():
                print(f"Warning: Empty translation for '{text}'")
            elif cleaned_translation.lower() == text.lower():
                print(f"Warning: Word remained untranslated - '{text}'")
            elif len(cleaned_translation.split()) > 3:
                print(f"Warning: Long translation for '{text}' -> '{cleaned_translation}'")

            return cleaned_translation, detected_lang, True

        return text, 'en', False
    except Exception as e:
        print(f"Translation error for text '{text}': {str(e)}")
        return text, 'unknown', False

def is_multiword(text):
    """Check if input is a multiword expression by counting spaces"""
    return len(text.strip().split()) > 1

#####################################################
#       Original Embedding Routines from code       #
#####################################################

def get_original_clip_embeddings(text_inputs, batch_size=50):
    print("Getting original CLIP embeddings...")
    model_clip, preprocess = clip.load("ViT-B/32", device=device)
    model_clip.eval()

    embeddings_list = []
    for i in tqdm(range(0, len(text_inputs), batch_size)):
        batch = text_inputs[i:i + batch_size]
        text_tokens = clip.tokenize(batch).to(device)
        with torch.no_grad():
            text_features = model_clip.encode_text(text_tokens)
            embeddings_list.append(text_features)

    return torch.cat(embeddings_list, dim=0)

def get_finetuned_clip_embeddings(text_inputs, batch_size=50):
    print("Getting emotion-finetuned CLIP embeddings...")
    model_emotionclip = EmotionCLIP(
        video_len=8,
        backbone_config="./EmotionCLIP/src/models/model_configs/ViT-B-32.json",
        backbone_checkpoint=None
    ).to(device)

    checkpoint = torch.load("emotionclip_latest.pt", map_location=device)
    model_emotionclip.load_state_dict(checkpoint['model'], strict=True)
    model_emotionclip.eval()

    embeddings_list = []
    for i in tqdm(range(0, len(text_inputs), batch_size)):
        batch = text_inputs[i:i + batch_size]
        text_tokens = clip.tokenize(batch).to(device)
        with torch.no_grad():
            text_features = model_emotionclip.encode_text(text_tokens)
            embeddings_list.append(text_features)

    return torch.cat(embeddings_list, dim=0)

def normalize_distribution(predictions):
    """
    Systematically adjust the distribution of predictions by smoothing high values
    using a percentile-based, graduated approach
    """
    preds_np = predictions.numpy()
    high_threshold = np.percentile(preds_np, 90)
    mask_high = preds_np > high_threshold
    if mask_high.any():
        relative_distance = ((preds_np[mask_high] - high_threshold) / (5 - high_threshold))
        smoothing = 1.0 - 0.1 * relative_distance
        preds_np[mask_high] *= smoothing
    return torch.tensor(preds_np)

#############################################################
#  The main function with new "sentence" embedding logic   #
#############################################################

def combined_predict(csv_path, single_model_path="saved_model", multiword_model_path="multiword_model"):
    print(f"Loading dataset from {csv_path}...")

    try:
        data = pd.read_csv(csv_path, delimiter=",", encoding='utf-8')
    except UnicodeDecodeError:
        data = pd.read_csv(csv_path, delimiter=",", encoding='latin-1')

    word_column = 'Expression' if 'Expression' in data.columns else 'Word'
    rating_column = 'Mean_C' if 'Mean_C' in data.columns else 'Conc.M'

    data = data.dropna(subset=[word_column])
    expressions = data[word_column]

    # Load both models (unchanged)
    print("Loading models...")
    single_model = ImprovedEnsemble.load_model(path=single_model_path)
    multiword_model = ImprovedEnsemble.load_model(path=multiword_model_path)

    # Also load the base CLIP model for token-level embedding if needed
    base_clip_model, _ = clip.load("ViT-B/32", device=device)
    base_clip_model.eval()

    # Process translations
    translations = []
    languages = []

    print("Processing translations...")
    for expr in expressions:
        translated_expr, detected_lang, was_translated = translate_to_english(expr)
        translations.append(translated_expr)
        languages.append(detected_lang)

    # We'll store the final predictions in a tensor
    predictions = torch.zeros(len(translations))

    # Also store per-word predictions if it's a sentence
    # (We can store them as a list of strings: "3.1;4.2;5.0;...")
    sentence_word_preds = [""] * len(translations)

    # Indices for printing or analyzing
    single_indices, multi_indices, sentence_indices = [], [], []

    # Classify each expression
    for idx, expr in enumerate(translations):
        if is_sentence(expr):
            sentence_indices.append(idx)
        elif is_multiword(expr):
            multi_indices.append(idx)
        else:
            single_indices.append(idx)

    print(f"\nFound {len(single_indices)} single-word expressions, {len(multi_indices)} multiword expressions, and {len(sentence_indices)} sentences.")

    # Single words -> single_model
    if len(single_indices) > 0:
        single_exprs = [translations[i] for i in single_indices]
        single_preds = single_model.predict(single_exprs)
        for i, pred in zip(single_indices, single_preds):
            predictions[i] = pred

    # Multi words -> multiword_model
    if len(multi_indices) > 0:
        multi_exprs = [translations[i] for i in multi_indices]
        multi_preds = multiword_model.predict(multi_exprs)
        for i, pred in zip(multi_indices, multi_preds):
            predictions[i] = pred

    # Sentences -> new token-level logic
    if len(sentence_indices) > 0:
        for i in sentence_indices:
            expr = translations[i]
            # 1) Get token-level embeddings from base_clip_model
            word_embs = get_clip_token_embeddings(expr, base_clip_model)  # list of torch vecs

            if len(word_embs) == 0:
                # If for some reason it was empty, default to 3.0
                predictions[i] = 3.0
                sentence_word_preds[i] = "3.0"
                continue

            # Step 2a) Re-fuse using emotion CLIP
            #   We'll do a function get_clip_token_embeddings but for the emotion model:
            emotion_clip = EmotionCLIP(
                video_len=8,
                backbone_config="./EmotionCLIP/src/models/model_configs/ViT-B-32.json",
                backbone_checkpoint=None
            ).to(device)
            checkpoint = torch.load("emotionclip_latest.pt", map_location=device)
            emotion_clip.load_state_dict(checkpoint['model'], strict=True)
            emotion_clip.eval()

            text_tokens = clip.tokenize([expr]).to(device)
            with torch.no_grad():
                emotion_vec = emotion_clip.encode_text(text_tokens)  # shape (1, 512)
            # We'll replicate it for each word
            emotion_vecs = emotion_vec.repeat(len(word_embs), 1)

            # Now we have: 
            #   original token-level embeddings = word_embs[j].shape = [d_model]
            #   repeated emotion embeddings = emotion_vecs[j].shape = [d_model]
            # We can combine them:
            # shape: (n_words, d_model * 2)
            stacked_word_embs = []
            for j in range(len(word_embs)):
                # cat original + emotion
                comb = torch.cat([word_embs[j], emotion_vecs[j]], dim=0)
                stacked_word_embs.append(comb.unsqueeze(0))

            combined_embeddings = torch.cat(stacked_word_embs, dim=0).to(device)  # (n_words, d_model*2)

            # Next, we replicate the scaling + regression from single_model
            # single_model.scaler_og is used for the original dims, but that was fit on dimension=2*512. 
            # We'll do: 
            combined_np = combined_embeddings.cpu().numpy()
            combined_scaled = single_model.scaler_og.transform(combined_np)
            combined_scaled_torch = torch.FloatTensor(combined_scaled).to(device)

            with torch.no_grad():
                word_predictions = single_model.combined_regressor(combined_scaled_torch)
                word_predictions = single_model.constrain_predictions(word_predictions)
                word_predictions = single_model.normalize_predictions(word_predictions)
            # shape: (n_words,)

            # Store the word-level predictions
            # e.g. "3.1;4.5;2.9"
            word_preds_list = [f"{p.item():.2f}" for p in word_predictions]
            sentence_word_preds[i] = ";".join(word_preds_list)

            # We produce an overall sentence rating by average or sum
            predictions[i] = torch.mean(word_predictions).item()

    # Now we've assigned predictions for single_words, multi_words, or sentences
    # Let's do distribution normalization
    predictions = normalize_distribution(predictions)
    predictions = torch.clamp(predictions, min=1.0, max=5.0)

    print(f"\nPredictions summary:")
    print(f"Min: {predictions.min().item():.2f}")
    print(f"Max: {predictions.max().item():.2f}")
    print(f"Mean: {predictions.mean().item():.2f}")

    # If we have true ratings, evaluate
    if rating_column in data.columns:
        true_ratings = torch.tensor(data[rating_column].values, dtype=torch.float32)
        predictions_np = predictions.numpy()
        true_ratings_np = true_ratings.numpy()

        # Overall metrics
        mse = np.mean((predictions_np - true_ratings_np) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions_np - true_ratings_np))
        r2 = r2_score(true_ratings_np, predictions_np)
        correlation, _ = pearsonr(predictions_np, true_ratings_np)

        print("\nOverall Metrics:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"Correlation: {correlation:.4f}")

        # Some quick plots
        plt.figure(figsize=(8, 6))
        plt.scatter(true_ratings_np, predictions_np, alpha=0.7, edgecolors='k',
                    label=f"R²: {r2:.2f}, Correlation: {correlation:.2f}")
        plt.plot([1, 5], [1, 5], 'r--', label="Ideal Fit (y=x)")
        plt.xlabel("True Ratings")
        plt.ylabel("Predicted Ratings")
        plt.title("Combined Model Predictions vs True Ratings")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Residual Plot
        residuals = predictions_np - true_ratings_np
        plt.figure(figsize=(8, 6))
        plt.scatter(true_ratings_np, residuals, alpha=0.7, edgecolors='k')
        plt.axhline(0, color='r', linestyle='--', label="Zero Residual Line")
        plt.xlabel("True Ratings")
        plt.ylabel("Residuals (Predicted - True)")
        plt.title("Residual Plot")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Histogram
        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=15, alpha=0.7, edgecolor='k')
        plt.axvline(0, color='r', linestyle='--', label="Mean Residual")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.title("Distribution of Residuals")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Density
        plt.figure(figsize=(8, 6))
        sns.kdeplot(true_ratings_np, label="True Ratings", fill=True, alpha=0.5)
        sns.kdeplot(predictions_np, label="Predicted Ratings", fill=True, alpha=0.5)
        plt.xlabel("Rating Value")
        plt.ylabel("Density")
        plt.title("Density Plot of Predicted vs True Ratings")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Save everything
    output_data = data.copy()
    output_data['Predicted_Concreteness'] = predictions.numpy()
    output_data['English_Translation'] = translations
    output_data['Detected_Language'] = languages
    output_data['Is_Multiword'] = [is_multiword(expr) for expr in translations]
    output_data['Is_Sentence'] = [is_sentence(expr) for expr in translations]
    output_data['Model_Used'] = [
        'Sentence' if is_sentence(expr) else 
        ('Multiword' if is_multiword(expr) else 'Single-word')
        for expr in translations
    ]

    # Add the per-word predictions if sentence
    output_data['Per_Word_Predictions'] = sentence_word_preds

    output_csv = "predictions_combined_methods_estonian.csv"
    output_data.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\nPredictions saved to {output_csv}")

if __name__ == "__main__":
    csv_path = "Estonian_full.csv"  # Your input file
    combined_predict(csv_path, single_model_path="saved_model", multiword_model_path="multiword_model")
