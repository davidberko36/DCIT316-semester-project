# Model Evaluation Guide for DCIT316 Semester Project

## Overview

This document provides instructions for evaluating the two machine learning models in the DCIT316 semester project:
1. **Fake News Detection Model** (Logistic Regression)
2. **News Recommendation Model** (Neural Network)

The evaluation script (`evaluate_models.py`) will assess each model's performance on their respective validation datasets and generate comprehensive reports.

## Prerequisites

- Python environment set up (virtual environment)
- All dependencies installed (requirements.txt)
- Trained models available in the models/ directory
- Data available in the Data/ directory

## Running the Evaluation

### Option 1: Using the Provided Scripts

#### Windows (Command Prompt)
```cmd
evaluate_models.bat
```

#### Windows (PowerShell)
```powershell
.\evaluate_models.ps1
```

#### Linux/macOS (Bash)
```bash
chmod +x evaluate_models.sh
./evaluate_models.sh
```

### Option 2: Running Python Script Directly

```bash
# Activate virtual environment
.\myenv\Scripts\Activate.ps1  # PowerShell
# OR
myenv\Scripts\activate.bat    # Command Prompt

# Run evaluation
python evaluate_models.py
```

## Customizing Evaluation

The evaluation script accepts several command-line arguments to customize the evaluation process:

```bash
python evaluate_models.py --help
```

Common options:
- `--output-dir OUTPUT_DIR`: Directory to save evaluation results (default: "evaluation_results")
- `--skip-fake-news`: Skip fake news detection model evaluation
- `--skip-recommendation`: Skip news recommendation model evaluation
- `--fake-news-test-data PATH`: Path to fake news test data (default: "Data/liar_dataset/valid.tsv")
- `--news-data PATH`: Path to news data (default: "Data/MINDsmall_dev/news.tsv")
- `--behaviors-data PATH`: Path to behaviors data (default: "Data/MINDsmall_dev/behaviors.tsv")

## Understanding the Results

The evaluation will generate three types of files in the `evaluation_results` directory:

1. **Fake News Evaluation** (`fake_news_evaluation_[timestamp].json`):
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix
   - Classification report by class
   - AUC (if applicable)

2. **Recommendation Evaluation** (`recommendation_evaluation_[timestamp].json`):
   - NDCG@k (Normalized Discounted Cumulative Gain)
   - MRR@k (Mean Reciprocal Rank)
   - Performance metrics for different k values

3. **Combined Report** (`evaluation_report_[timestamp].txt`):
   - Human-readable summary of both models' performance
   - Key metrics and parameters
   - Timestamp and configuration details

## Interpreting Metrics

### Fake News Detection
- **Accuracy**: Overall correctness (higher is better)
- **Precision**: Ratio of true positives to all positive predictions (higher is better)
- **Recall**: Ratio of true positives to all actual positives (higher is better)
- **F1-score**: Harmonic mean of precision and recall (higher is better)
- **AUC**: Area Under the ROC Curve (higher is better, ideal = 1.0)

### News Recommendation
- **NDCG@k**: Normalized Discounted Cumulative Gain at position k (higher is better, ideal = 1.0)
- **MRR@k**: Mean Reciprocal Rank at position k (higher is better, ideal = 1.0)

## Troubleshooting

If you encounter issues:

1. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Verify model files exist in the models/ directory:
   ```bash
   dir models\lr_pipeline.onnx models\recommender_scoring.onnx
   ```

3. Check data files exist in the Data/ directory

4. For detailed error messages, run with Python directly:
   ```bash
   python evaluate_models.py
   ```

