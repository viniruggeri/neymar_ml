import pandas as pd

X_bra = pd.read_csv("data/X_games_brasileirao_2025.csv")

# Minute numérico
X_bra['Minutes_num'] = X_bra['Minute'].str.replace("'", "").astype(int)

# Home
X_bra['Home'] = (X_bra['Venue'] == "Home").astype(int)

# Result_goal_diff
def goal_diff(res):
    try:
        goals = res.split('-')
        return int(goals[0]) - int(goals[1])
    except:
        return 0

X_bra['Result_goal_diff'] = X_bra['Result'].apply(goal_diff)

# Criar colunas faltantes de torneios do modelo treinado
tournaments = [
    "Champions League",
    "Copa del Rey",
    "Coupe de France",
    "Coupe de la Ligue",
    "LaLiga",
    "Ligue 1"
]

for t in tournaments:
    X_bra[f'Tournament_{t}'] = 0  # ele não joga nesses torneios

# Selecionar features na ordem do treino
features = ['Minutes_num', 'Home', 'Result_goal_diff'] + [f'Tournament_{t}' for t in tournaments]
X_ready = X_bra[features]

# Salvar CSV pronto
X_ready.to_csv("data/X_games_brasileirao_pred.csv", index=False)
print("CSV pronto para simulação no RF")
