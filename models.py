from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def train_fat_models(X, y, save_path='fat_percentage_predictor.joblib'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        'LinearRegression': LinearRegression(),
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # compute RMSE in a way compatible with different sklearn versions
        mse = mean_squared_error(y_test, y_pred)
        try:
            import numpy as _np
            rmse = _np.sqrt(mse)
        except Exception:
            rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        results[name] = {'model': model, 'rmse': rmse, 'r2': r2}
    best_name = max(results, key=lambda k: results[k]['r2'])
    best_model = results[best_name]['model']
    joblib.dump(best_model, save_path)
    return best_name, results[best_name]['rmse'], results[best_name]['r2'], best_model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    return {'rmse': rmse, 'r2': r2}
