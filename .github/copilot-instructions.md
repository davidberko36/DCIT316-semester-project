# AI Agent Instructions for DCIT316-semester-project

## Project Overview
This project implements machine learning models for two distinct use cases:
1. **Fake News Detection**: Using the LIAR dataset with logistic regression
2. **News Recommendation**: Using the MIND dataset with neural networks

## Architecture

### Data Flow
- Raw data (TSV format) → Model Training (Python scripts) → Model Export (ONNX) → Inference API (TBD)

### Key Components
- **Data Processing**: TSV parsing in both `logistic_regression.py` and `neural_nets.py`
- **Model Training**: Scikit-learn pipeline for fake news detection, PyTorch for recommendations
- **Model Export**: Models are exported to ONNX format for cross-platform deployment
- **Backend/Frontend**: Currently empty directories, likely planned for future development

## Development Workflow

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv myenv
.\myenv\Scripts\Activate.ps1  # PowerShell

# Install dependencies
pip install -r requirements.txt
```

### Training Models
```bash
# Train logistic regression model for fake news detection
python scripts/logistic_regression.py --train Data/liar_dataset/train.tsv --valid Data/liar_dataset/valid.tsv --out-model models/lr_pipeline.onnx

# Train neural network model for news recommendation (typical usage)
python scripts/neural_nets.py --train-behaviors Data/MINDsmall_train/behaviors.tsv --train-news Data/MINDsmall_train/news.tsv
```

## Data Formats

### LIAR Dataset (Fake News)
- TSV format with columns: ID, label, statement, subject, etc.
- Used in `logistic_regression.py` with configurable label/text columns

### MIND Dataset (News Recommendation)
- Multiple TSV files:
  - `news.tsv`: News article metadata with title, abstract, entities
  - `behaviors.tsv`: User interaction data with history and impressions
- Used in `neural_nets.py` with the `MINDBehaviorDataset` class

## Code Patterns & Conventions

### Model Pipeline Pattern
- Scikit-learn pipelines are used for text preprocessing + classification
- Models are exported to ONNX format for standardized deployment
- Example in `logistic_regression.py`:
  ```python
  pipe = Pipeline([
      ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000)),
      ("clf", LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced'))
  ])
  ```

### PyTorch Dataset Implementation
- Custom Dataset classes extend `torch.utils.data.Dataset`
- Implement `__init__`, `__len__`, and `__getitem__` methods
- Handle data loading and preprocessing
- Example: `MINDBehaviorDataset` in `neural_nets.py`

## Key Integration Points
- ONNX model format serves as the bridge between training and deployment
- Models saved to `models/` directory
- Exported models: `lr_pipeline.onnx`, `recommender_scoring.onnx`, `recommender_scoring.pth`

## Future Work Considerations
- Implement backend API for model inference (empty `backend/` directory)
- Develop frontend for user interaction (empty `frontend/` directory)
- Integrate both models into a unified application