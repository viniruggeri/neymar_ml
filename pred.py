import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import joblib
import numpy as np
from train_rf import *
# Carregar X dos jogos simulados
X_games = pd.read_csv("data/X_games_brasileirao_pred.csv")


# Carregar modelos
rf = joblib.load("models/rf_neymar_2015_2018.pkl")


# Previsões RF
y_pred_rf = rf.predict(X_games) 


# Criar DataFrame com resultados
games_pred = X_games.copy()
games_pred['Goals_pred_RF'] = y_pred_rf[:, 0]
games_pred['Assists_pred_RF'] = y_pred_rf[:, 1]
games_pred['G_A_pred_RF'] = y_pred_rf[:, 2]



# Calcular totals e médias por 90 (supondo Minutes_num por jogo)
games_pred['Goals_per90_RF'] = games_pred['Goals_pred_RF'] / (games_pred['Minutes_num']/90)
games_pred['Assists_per90_RF'] = games_pred['Assists_pred_RF'] / (games_pred['Minutes_num']/90)
games_pred['G_A_per90_RF'] = games_pred['G_A_pred_RF'] / (games_pred['Minutes_num']/90)


# Salvar CSV final da simulação
games_pred.to_csv("data/X_games_pred.csv", index=False)
print("Simulação completa! Resultados salvos em data/X_games_injury_pred.csv")

# Exemplo de análise resumida
summary = pd.DataFrame({
    "Model": ["RF"],
    "Total Goals/90": [
        games_pred['Goals_per90_RF'].sum()
    ],
    "Total Assists/90": [
        games_pred['Assists_per90_RF'].sum()
    ],
    "Total G/A/90": [
        games_pred['G_A_per90_RF'].sum()
    ]
})

print("\nResumo da temporada simulada:")
print(summary)

print("\nJogos simulados:")
print(games_pred[['Home', 'Goals_per90_RF', 'Assists_per90_RF', 'G_A_per90_RF']])


games_pred['Game_Num'] = range(1, len(games_pred)+1)

plt.figure(figsize=(12,6))
plt.plot(games_pred['Game_Num'], games_pred['Goals_per90_RF'], marker='o', label='Goals/90')
plt.plot(games_pred['Game_Num'], games_pred['Assists_per90_RF'], marker='s', label='Assists/90')
plt.plot(games_pred['Game_Num'], games_pred['G_A_per90_RF'], marker='^', label='G/A/90')

plt.title("Evolução de Neymar jogo a jogo (simulação RF)")
plt.xlabel("Número do Jogo")
plt.ylabel("Gols / Assistências por 90min")
plt.xticks(games_pred['Game_Num'])
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig(f"plots/pred_neymar_brasileirao_rf.png")