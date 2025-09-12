import pandas as pd
import numpy as np

# Carregar CSV dos jogos simulados do Brasileirão
X_bra = pd.read_csv("data/X_games_brasileirao_2025.csv")

# Coluna Minute numérica
X_bra['Minutes_num'] = X_bra['Minute'].str.replace("'", "").astype(int)

# Coluna Home (1 se o Santos for o mandante)
X_bra['Home'] = (X_bra['Venue'] == "Home").astype(int)

# Diferença de gols do resultado (ex: 3-0 -> 3)
def goal_diff(res):
    try:
        goals = res.split('-')
        return int(goals[0]) - int(goals[1])
    except:
        return 0

X_bra['Result_goal_diff'] = X_bra['Result'].apply(goal_diff)

# One-hot da competição (Brasileirão)
X_bra['Tournament_Brasileirao'] = 1  # só tem esse torneio

# Selecionar features finais (como foi treinado)
features = ['Minutes_num', 'Home', 'Result_goal_diff', 'Tournament_Brasileirao']
X_ready = X_bra[features]

# Salvar CSV pronto para o modelo
X_ready.to_csv("data/X_games_brasileirao_pred.csv", index=False)
print("CSV pronto salvo em data/X_games_brasileirao_pred.csv")
