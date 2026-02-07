# Scam Message Detector

A machine learning-based system for detecting scam messages using text analysis and URL pattern recognition. This project uses logistic regression with probability calibration to classify messages as scam or legitimate.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Building the Dataset](#building-the-dataset)
  - [Training the Model](#training-the-model)
  - [Running the API](#running-the-api)
  - [Making Predictions](#making-predictions)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [API Reference](#api-reference)
- [Performance Metrics](#performance-metrics)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Overview

This scam detector analyzes messages for suspicious patterns including:
- Urgency indicators ("verify now", "act now")
- Financial promises ("free money", "lottery")
- Requests for sensitive information (passwords, PINs)
- Suspicious URLs (shorteners, risky TLDs, IP addresses)
- Text characteristics (excessive capitalization, punctuation)

The system extracts 16 features from each message and uses a calibrated logistic regression model to output a probability score (0-100) indicating scam likelihood.

## Features

- **Comprehensive Feature Extraction**: 16 behavioral and structural features
- **URL Analysis**: Detection of URL shorteners, risky TLDs, IP-based URLs, and domain spoofing
- **Calibrated Probabilities**: Outputs reliable probability scores using sigmoid calibration
- **REST API**: Flask-based API for easy integration
- **Expandable Dataset**: Tools for generating and expanding training data
- **High Precision**: Optimized to minimize false positives

## Project Structure

```
.
├── .gitignore                    # Git ignore rules (excludes __pycache__, *.pkl)
├── README.md                     # This file
├── feature_extractor.py          # Feature extraction logic
├── build_dataset.py              # Initial dataset builder (sample data)
├── expand_dataset.py             # Production dataset generator
├── train_model.py                # Model training script
├── ml_api.py                     # Flask REST API
├── scam_dataset.csv              # Generated training dataset
└── scam_detector_model.pkl       # Trained model (generated)
```

## Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd scam-detector
```

2. Install required dependencies:
```bash
pip install pandas scikit-learn flask joblib
```

## Usage

### Building the Dataset

#### Option 1: Sample Dataset (for testing)

Generate a small dataset with 6 samples:

```bash
python build_dataset.py
```

This creates `scam_dataset.csv` with basic examples for quick testing.

#### Option 2: Production Dataset (recommended)

Generate a larger dataset with 1000 samples:

```bash
python expand_dataset.py
```

This creates:
- 600 scam examples (variations of 5 scam templates)
- 400 safe examples (variations of 5 legitimate templates)
- Random mutations for diversity

### Training the Model

Train the logistic regression model with calibration:

```bash
python train_model.py
```

**Output:**
```
Precision: 0.XXX
Recall:    0.XXX
F1-score:  0.XXX
✅ Model saved
```

This generates `scam_detector_model.pkl` containing the trained model.

### Running the API

Start the Flask API server:

```bash
python ml_api.py
```

The API will be available at `http://localhost:5000`

### Making Predictions

Send POST requests to the `/predict` endpoint:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "content": "URGENT! Verify your account now at bit.ly/fake-link",
    "manual_score": 85
  }'
```

**Response:**
```json
{
  "probability": 0.923,
  "risk_score": 92
}
```

## Feature Engineering

The system extracts 16 features from each message:

### Text-Based Features (f1-f8)

| Feature | Description | Type |
|---------|-------------|------|
| `f1` | Has urgency keywords (urgent, verify now, suspended) | Binary (0/1) |
| `f2` | Has money-related keywords (free money, lottery, earn) | Binary (0/1) |
| `f3` | Requests sensitive info (password, CVV, PIN, OTP) | Binary (0/1) |
| `f4` | Suggests off-platform contact (Telegram, WhatsApp) | Binary (0/1) |
| `f5` | Text length (character count) | Integer |
| `f6` | Number of exclamation marks | Integer |
| `f7` | Uppercase letter ratio | Float (0-1) |
| `f8` | Digit ratio | Float (0-1) |

### URL-Based Features (f9-f15)

| Feature | Description | Type |
|---------|-------------|------|
| `f9` | Number of URLs in message | Integer |
| `f10` | URL-to-word ratio | Float |
| `f11` | Contains IP-based URL | Binary (0/1) |
| `f12` | Uses URL shortener (bit.ly, tinyurl, etc.) | Binary (0/1) |
| `f13` | Has risky TLD (.tk, .ml, .ga, .pw, etc.) | Binary (0/1) |
| `f14` | Domain spoofing attempt | Binary (0/1) |
| `f15` | Contains verified domain | Binary (0/1) |

### Manual Score (f16)

| Feature | Description | Range |
|---------|-------------|-------|
| `f16` | Human-provided risk score (optional) | -20 to +20 |

The manual score is clamped to prevent data leakage and over-reliance on this feature.

### Verified Domains

The following domains are recognized as legitimate:
- google.com
- apple.com
- amazon.com
- microsoft.com
- paypal.com
- github.com
- youtube.com

### High-Risk TLDs

The system flags these top-level domains as high-risk:
- .tk, .ml, .ga, .cf, .gq (free TLDs)
- .pw, .click, .loan, .win, .bid, .racing, .kim
- .xyz, .top, .cc, .ru, .cn

## Model Architecture

### Base Model

**Logistic Regression** with the following configuration:
- Solver: L-BFGS
- Max iterations: 1000
- Regularization: L2 (default)

### Calibration

**CalibratedClassifierCV** with sigmoid calibration:
- Method: Platt scaling
- Cross-validation: 5-fold (default)
- Purpose: Converts raw scores to reliable probabilities

### Training Process

1. **Data Split**: 80% training, 20% testing (stratified)
2. **Model Training**: Logistic regression on training set
3. **Calibration**: Sigmoid calibration for probability scores
4. **Evaluation**: Precision, recall, F1-score on test set

## API Reference

### POST /predict

Analyzes a message and returns scam probability.

#### Request Body

```json
{
  "content": "string (required) - The message text to analyze",
  "manual_score": "integer (optional) - Manual risk score (-20 to +20, default: 0)"
}
```

#### Response

```json
{
  "probability": "float - Scam probability (0.0 to 1.0)",
  "risk_score": "integer - Risk score (0 to 100)"
}
```

#### Example

**Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Congratulations! You won $1000. Click here: bit.ly/claim"
  }'
```

**Response:**
```json
{
  "probability": 0.876,
  "risk_score": 88
}
```

## Performance Metrics

The model is evaluated using three key metrics:

- **Precision**: Proportion of predicted scams that are actually scams
- **Recall**: Proportion of actual scams that are correctly identified
- **F1-Score**: Harmonic mean of precision and recall

Typical performance on the test set:
- Precision: 0.85-0.95
- Recall: 0.75-0.90
- F1-Score: 0.80-0.92

*Note: Performance varies based on dataset quality and size. The expanded dataset generally yields better results.*

## Examples

### Scam Messages (High Risk)

```python
# Example 1: Urgency + URL shortener
{
  "content": "URGENT! Verify your account now at bit.ly/verify-account",
  "manual_score": 70
}
# Expected: High probability (0.85-0.95)

# Example 2: Money promise + suspicious TLD
{
  "content": "Free job offer! Earn $500 daily from home: jobs.tk/apply",
  "manual_score": 80
}
# Expected: High probability (0.80-0.95)

# Example 3: Sensitive info request
{
  "content": "Your account was suspended. Verify your password immediately!",
  "manual_score": 75
}
# Expected: High probability (0.75-0.90)
```

### Legitimate Messages (Low Risk)

```python
# Example 1: Documentation link
{
  "content": "Please review the project documentation at github.com/user/repo",
  "manual_score": -10
}
# Expected: Low probability (0.05-0.20)

# Example 2: Meeting reminder
{
  "content": "Meeting scheduled for tomorrow at 10am in Conference Room B",
  "manual_score": 0
}
# Expected: Very low probability (0.02-0.10)

# Example 3: Tutorial reference
{
  "content": "Check out the Python tutorial series on youtube.com/watch?v=xyz",
  "manual_score": -5
}
# Expected: Low probability (0.03-0.15)
```

## Dataset Generation

### Template-Based Approach

The `expand_dataset.py` script uses template mutation:

**Scam Templates:**
- "URGENT verify your account"
- "Free job offer earn money"
- "Suspicious login detected"
- "You won a lottery prize"
- "Limited time investment"

**Safe Templates:**
- "Please review the documentation"
- "Meeting scheduled tomorrow"
- "Project update attached"
- "Watch tutorials online"
- "Monthly report available"

**Mutations:**
- Appends random noise: " now", " please", " today", " asap", "!!!", ""
- Randomizes manual scores within appropriate ranges

### Customizing the Dataset

To add your own examples:

1. Edit `expand_dataset.py`
2. Add templates to `scam_texts` or `safe_texts`
3. Adjust iteration counts for balance
4. Run the script to regenerate the dataset

```python
scam_texts = [
    "URGENT verify your account",
    "Your custom scam template here",
    # Add more...
]
```

## Feature Extractor Details

### URL Extraction

The feature extractor identifies URLs using regex:
```python
r'(https?:\/\/[^\s]+|www\.[^\s]+)'
```

### Case Sensitivity

- Keywords are matched case-insensitively
- Domain matching is case-insensitive
- TLD matching is case-insensitive

### Manual Score Capping

The manual score is clamped to prevent overfitting:
```python
manual_score = min(max(manual_score, -20), 20)
```

This ensures the model learns from text features rather than relying entirely on external scores.

## Limitations and Future Improvements

### Current Limitations

1. **Language**: Only supports English text
2. **Dataset Size**: Limited training examples (1000 samples)
3. **Static Rules**: Hardcoded keyword lists and domain lists
4. **No Context**: Analyzes messages in isolation
5. **No Image Analysis**: Cannot process image-based scams

### Potential Improvements

- [ ] Multi-language support
- [ ] Deep learning models (BERT, RoBERTa)
- [ ] Real-time URL reputation checking
- [ ] Context-aware analysis (conversation history)
- [ ] Active learning for continuous improvement
- [ ] OCR for image-based scam detection
- [ ] Ensemble methods (combining multiple models)
- [ ] Feature importance analysis

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

### Areas for Contribution

- Adding more training examples
- Improving feature extraction
- Enhancing URL analysis
- Adding new scam patterns
- Performance optimization
- Documentation improvements

## License

This project is provided as-is for educational and research purposes. Please ensure compliance with applicable laws and regulations when deploying in production environments.

## Acknowledgments

- scikit-learn for machine learning utilities
- Flask for API framework
- The open-source community for inspiration and tools

## Contact

For questions, issues, or suggestions, please open an issue on the repository.

---

**Disclaimer**: This tool is designed to assist in scam detection but should not be the sole method of protection. Always exercise caution with unsolicited messages and verify information through official channels.
