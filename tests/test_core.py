import pandas as pd
import os
from prep import normalize_columns, build_recommendation_df, clean_meals
from models import train_fat_models
from recs import build_tfidf_similarity


def test_normalize_columns():
    df = pd.DataFrame({'Col One (x)': [1, 2], 'another_col': [3, 4]})
    df2 = normalize_columns(df)
    assert all(' ' not in c for c in df2.columns)
    assert df2.columns.str.islower().all()


def test_build_recommendation_and_tfidf():
    df = pd.DataFrame({'type': ['push', 'pull'], 'body_part': ['chest', 'back'], 'equipment': ['bodyweight', 'dumbbell'], 'level': ['beginner', 'intermediate']})
    rec = build_recommendation_df(df)
    vec, mat, sim = build_tfidf_similarity(rec)
    assert mat.shape[0] == 2


def test_train_models_smoke():
    X = pd.DataFrame({'feat1': [1, 2, 3, 4, 5], 'feat2': [2, 3, 4, 5, 6]})
    y = pd.Series([10, 12, 14, 16, 18])
    name, rmse, r2, model = train_fat_models(X, y, save_path='test_model.joblib')
    assert model is not None
    if os.path.exists('test_model.joblib'):
        os.remove('test_model.joblib')
