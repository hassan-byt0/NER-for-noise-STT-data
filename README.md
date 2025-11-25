# PII Named Entity Recognition for STT Transcripts

## Overview
This project implements a machine learning model to detect and classify Personally Identifiable Information (PII) entities in noisy Speech-to-Text (STT) transcripts. The model performs token-level Named Entity Recognition (NER) to identify sensitive information such as credit card numbers, phone numbers, emails, names, and dates.

## Entity Types
The model detects 7 entity types:
- **PII Entities** (require masking/redaction):
  - `CREDIT_CARD` - Credit card numbers
  - `PHONE` - Phone numbers
  - `EMAIL` - Email addresses
  - `PERSON_NAME` - Person names
  - `DATE` - Dates
- **Non-PII Entities** (informational):
  - `CITY` - City names
  - `LOCATION` - Location names

## Project Structure
```
.
├── data/
│   ├── train.jsonl          # Training data (851 samples)
│   ├── dev.jsonl            # Development/validation data (175 samples)
│   ├── test.jsonl           # Test data (no labels)
│   └── stress.jsonl         # Adversarial stress test set (100 samples)
├── src/
│   ├── train.py             # Training script
│   ├── predict.py           # Inference script
│   ├── eval_span_f1.py      # Evaluation metrics
│   ├── measure_latency.py   # Latency benchmarking
│   ├── dataset.py           # Dataset loading and preprocessing
│   ├── model.py             # Model creation
│   ├── labels.py            # Label definitions and PII mapping
│   └── generate_dev_data.py # Data augmentation script
├── out/                     # Model checkpoints and predictions
├── requirements.txt         # Python dependencies
└── assignment.md           # Assignment details
```

## Installation

### 1. Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
pip install protobuf sentencepiece  # Required for DeBERTa models
```

## Usage

### Training
Train the model with default settings (DeBERTa-v3-base):
```powershell
python src/train.py --model_name microsoft/deberta-v3-base --train data/train.jsonl --dev data/dev.jsonl --out_dir out --epochs 4 --batch_size 8 --lr 2e-5 --weight_decay 0.01
```

**Key Training Parameters:**
- `--model_name`: Base model (default: `microsoft/deberta-v3-base`)
- `--epochs`: Number of training epochs (default: 3)
- `--lr`: Learning rate (default: 2e-5)
- `--weight_decay`: Weight decay for regularization (default: 0.01)
- `--warmup_ratio`: Warmup ratio for LR scheduler (default: 0.1)
- `--max_grad_norm`: Gradient clipping threshold (default: 1.0)
- `--batch_size`: Training batch size (default: 8)
- `--max_length`: Maximum sequence length (default: 256)

### Inference/Prediction
Run predictions with constrained decoding (recommended):
```powershell
python src/predict.py --model_dir out --input data/dev.jsonl --output out/dev_pred.json --decoding constrained
```

**Decoding Options:**
- `--decoding greedy`: Standard argmax decoding (faster, less precise)
- `--decoding constrained`: Enforces BIO consistency (slower, more precise)

### Evaluation
Evaluate predictions against gold labels:
```powershell
python src/eval_span_f1.py --gold data/dev.jsonl --pred out/dev_pred.json
```

Outputs:
- Per-entity precision, recall, and F1 scores
- Macro-averaged F1 across all entities
- PII-only metrics (combined for all PII entities)
- Non-PII metrics (for CITY and LOCATION)

### Latency Measurement
```powershell
python src/measure_latency.py --model_dir out --input data/dev.jsonl --runs 50
```

## Model Architecture

### Base Model
**DeBERTa-v3-base**: Enhanced RoBERTa with disentangled attention
- Better handling of code-mixed text (Hinglish)
- Improved contextual understanding
- 86M parameters

### Training Configuration
- **Optimizer**: AdamW with weight decay (0.01)
- **Learning Rate**: 2e-5 with linear warmup (10% of steps)
- **Gradient Clipping**: Max norm 1.0
- **Label Encoding**: BIO tagging scheme

### Decoding Strategy
**Constrained Decoding**:
- Enforces BIO consistency (I-X only after B-X or I-X)
- Prevents spurious predictions
- Improves PII precision by ~10-20%

## Performance Metrics

### Dev Set (175 samples) - Constrained Decoding
- **PII Precision**: 0.631
- **PII Recall**: 0.767
- **PII F1**: 0.693
- **Macro F1**: 0.553

### Stress Test (100 samples)
- **PII Precision**: 0.479
- **PII F1**: 0.604

### Latency (CPU)
- **p50**: ~65ms
- **p95**: ~86ms
- Note: Exceeds 20ms target (trade-off for accuracy)

## Data Augmentation

Dev set augmented from 10 → 175 samples:
```powershell
python src/generate_dev_data.py
```

## Key Features

1. **Robust to Noisy STT**: Handles missing punctuation, spoken forms, code-mixing
2. **BIO Consistency**: Constrained decoding prevents invalid sequences
3. **Optimized Training**: LR scheduling, weight decay, gradient clipping
4. **Flexible Inference**: Greedy or constrained decoding

## Trade-offs

- **DeBERTa-v3-base**: Higher accuracy, slower (~86ms p95)
- **DistilBERT**: Lower accuracy, faster (~20ms p95)
- **Constrained decoding**: +10% precision, +2-3ms latency

## References

- Assignment: `assignment.md`
- DeBERTa: https://huggingface.co/microsoft/deberta-v3-base
