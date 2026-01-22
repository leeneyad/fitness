---
description: Repository Information Overview
alwaysApply: true
---

# GymML - Fitness & Machine Learning Web Application

## Summary
GymML is a Flask-based web application that combines machine learning with fitness recommendations. The application predicts body fat percentage using trained ML models and provides personalized exercise recommendations based on user input. It features email integration for sending workout plans and uses scikit-learn for predictive modeling.

## Structure
**Main Components**:
- **app.py**: Flask web application with 5 routes serving the main user interface
- **models.py**: ML model training module for body fat percentage prediction
- **templates/**: HTML templates (index.html, goals.html, results.html)
- **tests/**: Unit tests for core functionality
- **dat/, dat2/, food/**: Data directories containing CSV files (git-ignored)
- **gym.ipynb**: Jupyter notebook for exploratory data analysis
- **Visualizations**: PNG files showing BMI distribution, fat percentage analysis, and exercise statistics

## Language & Runtime
**Language**: Python  
**Package Manager**: pip  
**Web Framework**: Flask  
**ML Framework**: scikit-learn

## Dependencies
**Main Dependencies**:
- `Flask` - Web framework
- `flask-mail` - Email integration
- `scikit-learn` - Machine learning models (LinearRegression, DecisionTree, RandomForest)
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `joblib` - Model serialization

**Development Dependencies**:
- `pytest` - Testing framework
- `jupyter` - Interactive notebooks
- `matplotlib` - Data visualization
- `seaborn` - Statistical visualizations

## Build & Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Access at http://localhost:5000 (debug mode enabled)
```

## Main Files
**Entry Point**: `app.py:260` - Flask application runs with `app.run(debug=True)`

**Routes**:
- `/` (GET) - Home page
- `/submit_basic_data` (POST) - User data submission
- `/goals` (GET) - Goals selection page
- `/submit_goals_and_predict` (POST) - Prediction and recommendations
- `/send_email` (POST) - Email workout plans

**ML Module**: `models.py` contains:
- `train_fat_models()` - Trains and compares 3 regression models (Linear, DecisionTree, RandomForest)
- `evaluate_model()` - Model evaluation with RMSE and R² metrics
- Model persistence using joblib

**Configuration**: `app.py:14-21` - Mail server settings (SMTP via Gmail)

## Testing
**Framework**: pytest  
**Test Location**: `tests/test_core.py`  
**Test Files**: Files matching `test_*.py` pattern

**Run Command**:
```bash
pytest tests/
```

**Test Coverage**:
- Data normalization tests
- Recommendation building tests
- Model training smoke tests

**Note**: Some tests reference modules (`prep.py`, `recs.py`) that may be outdated or removed from the current codebase.

## Data Management
**Data Files** (git-ignored):
- CSV files in `dat/`, `dat2/`, `food/` directories
- Trained models (`*.joblib`, `*.pkl`)
- Jupyter checkpoints

**Visualizations**:
- BMI Distribution
- Weight vs Fat Percentage correlation
- Important Features for Fat prediction
- Number of Exercises per Body Part
