# Universal Concreteness Predictor

A deep learning system for predicting concreteness ratings of words, multi-word expressions, and sentences using CLIP and EmotionCLIP embeddings. The system supports multiple languages through automatic translation to English.

## Overview

This project implements a concreteness prediction system that:
- Handles single words, multi-word expressions, and full sentences
- Supports multiple languages through M2M100 translation
- Uses both original CLIP and emotion-fine-tuned CLIP embeddings
- Provides detailed visualizations and evaluation metrics
- Outputs per-word predictions for sentences

## Requirements

### Python Dependencies
```
torch
torchvision
numpy 
pandas
scikit-learn
scipy
matplotlib
seaborn
tqdm
langdetect
transformers
```

### External Models and Files
1. **EmotionCLIP Model**
   - Download `emotionclip_latest.pt` from [EmotionCLIP releases](https://github.com/emotionclip/releases)
   - Place in root directory

2. **Facebook M2M100 Translator**
   - Will be downloaded automatically through Hugging Face transformers
   - Requires ~1.2GB disk space

### Directory Structure
```
.
├── README.md
├── Predictor.py
├── EmotionCLIP/
│   └── src/
│       └── models/
│           ├── base.py
│           └── model_configs/
│               └── ViT-B-32.json
├── saved_model/
│   ├── combined_regressor.pth
│   └── scalers.pkl
├── multiword_model/
│   ├── combined_regressor.pth
│   └── scalers.pkl
└── emotionclip_latest.pt  # Download separately
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/universal-concreteness-predictor.git
cd universal-concreteness-predictor
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download EmotionCLIP model:
```bash
# Download emotionclip_latest.pt and place in root directory
wget https://github.com/emotionclip/releases/download/v1.0/emotionclip_latest.pt
```

## Usage

1. Prepare your input CSV file with a column named either 'Expression' or 'Word' containing the text to analyze.

2. Run the predictor:
```python
from Predictor import combined_predict

# Run prediction
combined_predict(
    csv_path="your_input.csv",
    single_model_path="saved_model",
    multiword_model_path="multiword_model"
)
```

The script will:
- Load and process your input data
- Translate non-English text automatically
- Generate predictions using appropriate models
- Create visualizations if true ratings are provided
- Save results to a new CSV file with predictions and metadata

### Output Format

The output CSV will contain:
- Original input data
- Predicted concreteness ratings (1-5 scale)
- English translations (if applicable)
- Detected languages
- Classification (Single-word/Multiword/Sentence)
- Per-word predictions for sentences

## Model Details

The system uses three main components:

1. **Single-word Model**: For individual words
2. **Multi-word Model**: For phrases and expressions
3. **Sentence Processing**: Uses token-level CLIP embeddings

Each model combines embeddings from:
- Original CLIP (ViT-B/32)
- Emotion-fine-tuned CLIP

## Acknowledgments

This project uses:
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [EmotionCLIP](https://github.com/emotionclip)
- [Facebook's M2M100](https://huggingface.co/facebook/m2m100_1.2B)

## Citation

If you use this code in your research, please cite:

```TBD
```

## License

[MIT License](LICENSE)
