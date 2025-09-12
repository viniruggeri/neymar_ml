import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# dados de teste
X_test = pd.read_csv("data/X_neymar_2023_2025.csv")
y_test = pd.read_csv("data/y_neymar_2023_2025.csv")

# carrega RF
rf = joblib.load(MODELS_DIR / "rf_neymar_2023_2025.pkl")

# faz previsões base
pred_base = rf.predict(X_test)

# residual std por target
residuals_std = {t: np.std(y_test[t] - pred_base[:, i]) for i, t in enumerate(y_test.columns)}

# simulação Monte Carlo
noise_perc = [0.05, 0.15, 0.3]
N_sim = 1000

for i, target in enumerate(y_test.columns):
    plt.figure(figsize=(8,6))
    for perc in noise_perc:
        sims = []
        for _ in range(N_sim):
            noise = np.random.normal(0, residuals_std[target]*perc, size=pred_base.shape[0])
            sims.append(pred_base[:, i] + noise)
        sims = np.array(sims)
        pred_mean = sims.mean(axis=0)
        plt.scatter(y_test[target], pred_mean, alpha=0.5, label=f"Noise ±{int(perc*100)}%")
        np.save("sim_data/" + f"rf_{target}_sim_{int(perc*100)}.npy", sims)

    # linha y=x
    lims = [y_test[target].min()-1, y_test[target].max()+1]
    plt.plot(lims, lims, 'k--', alpha=0.7)
    plt.title(f"Monte Carlo Real vs Pred - RF {target}")
    plt.xlabel("Real")
    plt.ylabel("Predito")
    plt.legend()
    plt.savefig(PLOTS_DIR / f"rf_{target}_montecarlo.png")
    plt.show()
