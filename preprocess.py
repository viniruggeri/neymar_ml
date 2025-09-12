import pandas as pd
import numpy as np

# Carrega CSV
df = pd.read_csv('data/neymar.csv')

# Converte coluna Date
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Filtra datas
start = pd.to_datetime('2023-08-01')
end = pd.to_datetime('2025-08-25')
df = df[(df['Date'] >= start) & (df['Date'] <= end)]

# Limpa colunas de minuto (ex: '90+1' → 91)
df['Minute'] = df['Minute'].str.replace("'", "").str.split('+').apply(lambda x: sum(int(i) for i in x if i.isdigit()))
df['Minute'] = df['Minute'].fillna(0).astype(int)

# Gols e Assistências por 90min
df['Goals_per90'] = df['Goal Type'].apply(lambda x: 0 if x=='Not Applicable' else 1) / (df['Minute']/90)
df['Assists_per90'] = df['Assist'].apply(lambda x: 0 if x=='Not Applicable' else 1) / (df['Minute']/90)
df['G_A_per90'] = df['Goals_per90'] + df['Assists_per90']

# Home / Away
df['Home'] = df['Venue'].apply(lambda x: 1 if x.lower()=='home' else 0)

# Diferença de gols
def result_diff(result):
    try:
        f, a = result.split(':')
        return int(f) - int(a)
    except:
        return 0
df['Result_goal_diff'] = df['Result'].apply(result_diff)

# One-hot encoding da competição
tourn_dummies = pd.get_dummies(df['Tournament'], prefix='Tournament')
df = pd.concat([df, tourn_dummies], axis=1)

# Seleciona features e targets
feature_cols = ['Minute', 'Home', 'Result_goal_diff'] + list(tourn_dummies.columns)
target_cols = ['Goals_per90', 'Assists_per90', 'G_A_per90']

X = df[feature_cols]
y = df[target_cols]

# Salva pré-processado
X.to_csv('data/X_neymar_2023_2025.csv', index=False)
y.to_csv('data/y_neymar_2023_2025.csv', index=False)

print("Pré-processamento finalizado!")
print("Features:\n", X.head())
print("Targets:\n", y.head())