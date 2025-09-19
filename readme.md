🟡⚫ Neymar Simulator – Machine Learning aplicado ao Futebol

Este projeto é um simulador de performance do Neymar usando Machine Learning.
A ideia é simples (Neymar hipotético): prever gols e assistências do Neymar em jogos futuros ou em cenários hipotéticos (como a temporada sem lesão), a partir de dados históricos.

O que tem aqui? 

Coleta e pré-processamento de dados de jogos (minuto jogado, mandante/visitante, diferença de gols, competição, etc.).

Feature engineering com one-hot encoding para torneios e métricas derivadas.

Modelos treinados:

🎯 Random Forest (principal)

❌ XGBoost (descartado por overfitting descomunal, até pro Ney)


Simulação de temporadas:

"E se o Neymar não tivesse se lesionado em 17/18?"

"Quantos gols/assist ele faria no Brasileirão 2025?"


Visualização: evolução de G/A ao longo das partidas simuladas.


👉 Resultado: um "Neymar Prime Simulator" que permite brincar com cenários e entender o impacto dos dados no desempenho.


---

🛠️ Como rodar

1. Clonar o repositório

git clone https://github.com/viniruggeri/neymar_ml.git
cd neymar_ml

2. Criar e ativar o ambiente

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

3. Instalar dependências

pip install -r requirements.txt

4. Estrutura de pastas esperada

data/
 ├─ X_games_injury.csv          # jogos simulados pós-lesão
 ├─ X_games_brasileirao_2025.csv # jogos reais/futuros do BR
 ├─ X_games_injury_pred.csv     # features preparadas
 └─ y_neymar_20XX.csv           # rótulos de treino (gols, assists)

models/
 └─ rf_neymar_2015_2018.pkl     # modelo RF treinado no auge
 └─ rf_neymar_2023_2025.pkl     # modelo RF treinado recente

5. Treinar modelo

python train_rf.py

6. Simular temporada

python simulate_rf.py (ou use os arquivos prontos para simular)

Saída esperada:

CSV com resultados de cada jogo (data/X_games_pred.csv)

Resumo da temporada no console (Gols/90, Assist/90, G+A/90)

Gráfico de evolução dos jogos (se plot ativado).



---

📊 Exemplo de saída

Resumo da temporada simulada:
  Model  Total Goals/90  Total Assists/90  Total G/A/90
0    RF       16.40              10.33         26.73

Evolução jogo a jogo:



---

📌 Observações

Este projeto é experimental e não oficial.

O objetivo é explorar Machine Learning aplicado ao esporte em cenários "what-if".

Resultados não devem ser interpretados como previsões reais.



---

⚡ Feito por @viniruggeri com muito Python, IA e amor ao futebol.
