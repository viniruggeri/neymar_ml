import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

X_PATH = "data/X_neymar_2023_2025.csv"
Y_PATH = "data/y_neymar_2023_2025.csv"
MODELS_DIR = Path("models")
TEST_SIZE = 0.2
RANDOM_STATE = 42

# sensible hyperparams
RF_PARAMS = {"n_estimators": 200, "max_depth": 7, "random_state": RANDOM_STATE, "n_jobs": -1}
XGB_BASE_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
    "verbosity": 0
}

def rmse(y_true, y_pred):
    """
    RMSE robusto que não depende de versões do sklearn.
    y_true, y_pred -> arrays 1D ou 2D
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def print_metrics(prefix, y_true, y_pred, target_names=None):
    """
    y_true, y_pred: numpy arrays shape (n_samples, n_targets)
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    n_targets = y_true.shape[1]
    if target_names is None:
        target_names = [f"t{i}" for i in range(n_targets)]

    print(f"\n=== {prefix} ===")
    maes, rmses, r2s = [], [], []
    for i in range(n_targets):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r = rmse(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        maes.append(mae); rmses.append(r); r2s.append(r2)
        print(f"[{target_names[i]}] MAE: {mae:.4f}  RMSE: {r:.4f}  R2: {r2:.4f}")
    # summary
    print(f"[avg ] MAE: {np.mean(maes):.4f}  RMSE: {np.mean(rmses):.4f}  R2: {np.mean(r2s):.4f}")

def main():
    # checagens iniciais
    if not Path(X_PATH).exists() or not Path(Y_PATH).exists():
        print("Erro: não achei os arquivos pré-processados. Verifique:")
        print(f" - {X_PATH}")
        print(f" - {Y_PATH}")
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # carrega
    X = pd.read_csv(X_PATH)
    y = pd.read_csv(Y_PATH)

    # quick sanity
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    if X.shape[0] != y.shape[0]:
        print("Erro: X e y têm números diferentes de linhas. Confira o preprocess.")
        sys.exit(1)

    target_names = list(y.columns)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Split feito: {X_train.shape[0]} treino / {X_test.shape[0]} teste")

    # RandomForest 
    print("\nTreinando RandomForest...")
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train, y_train)
    y_train_pred_rf = rf.predict(X_train)
    y_test_pred_rf = rf.predict(X_test)

    print_metrics("RF - treino", y_train, y_train_pred_rf, target_names)
    print_metrics("RF - teste", y_test, y_test_pred_rf, target_names)

    # salva RF
    rf_path = MODELS_DIR / "rf_neymar_2023_2025.pkl"
    joblib.dump(rf, rf_path)
    print(f"RandomForest salvo em: {rf_path}")


    print("\nTreinamento concluído.")
if __name__ == "__main__":
    main()
