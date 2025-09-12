import pandas as pd
from datetime import datetime

# Jogos do Brasileirão restantes do Santos/2025
games = [
    ("2025-09-14", "Atlético-MG", "Arena MRV", 0),
    ("2025-09-21", "São Paulo", "Vila Belmiro", 1),
    ("2025-09-28", "Bragantino", "Cícero de Souza Marques", 0),
    ("2025-10-01", "Grêmio", "Vila Belmiro", 1),
    ("2025-10-05", "Ceará", "Castelão (CE)", 0),
    ("2025-10-15", "Corinthians", "Vila Belmiro", 1),
    ("2025-10-19", "Vitória", "Vila Belmiro", 1),
    ("2025-10-26", "Botafogo", "Vila Belmiro", 0),
    ("2025-11-02", "Fortaleza", "Vila Belmiro", 1),
    ("2025-11-05", "Palmeiras", "Allianz Parque", 0),
    ("2025-11-09", "Flamengo", "Maracanã", 0),
    ("2025-11-19", "Mirassol", "Vila Belmiro", 1),
    ("2025-11-23", "Internacional", "Beira-Rio", 0),
    ("2025-11-30", "Sport", "Vila Belmiro", 1),
    ("2025-12-03", "Juventude", "Arena do Jacaré", 0),
    ("2025-12-07", "Cruzeiro", "Vila Belmiro", 1)
]

# Criar DataFrame
df = pd.DataFrame({
    "Date": [g[0] for g in games],
    "Opponent": [g[1] for g in games],
    "Venue": ["Home" if g[3]==1 else "Away" for g in games],
    "Competition": ["Brasileirão"]*len(games),
    "Result": [""]*len(games),
    "Position": ["LW"]*len(games),
    "Minute": ["90'"]*len(games),
    "Quando marcou": [""]*len(games),
    "Tipo de gol": [""]*len(games),
    "Assistência": ["Não reportado"]*len(games)
})

# Salvar CSV
df.to_csv("data/X_games_brasileirao_2025.csv", index=False)
print("CSV de jogos do Brasileirão 2025 gerado: data/X_games_brasileirao_2025.csv")