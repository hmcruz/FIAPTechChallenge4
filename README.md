# FIAPTechChallenge4

## API para Previsão de Preço de Ações usando Modelo LSTM

Este projeto utiliza um modelo LSTM (Long Short-Term Memory) para prever o preço de fechamento de ações com base em dados históricos. O modelo é treinado e testado utilizando **PyTorch**, e as previsões são realizadas por meio de uma API desenvolvida com **Flask**.

---

## 📌 Funcionalidades

### Treinamento e Teste do Modelo LSTM:
- O modelo é treinado usando dados de ações obtidos via **Yahoo Finance** (`yfinance`).
- Utiliza a biblioteca **MLflow** para logar os experimentos e armazenar os modelos treinados.
- Escalonamento dos dados é feito com **MinMaxScaler** para normalizar os valores.

### API Flask para Previsão:
- O arquivo `predict_stock.py` contém a API que recebe solicitações de previsão.
- O endpoint `/predict` processa uma requisição `POST` com os seguintes parâmetros:
  - `symbol`: Símbolo da ação (ex.: `PETR4.SA`).
  - `reference_date`: Data de referência para a previsão (formato: `yyyy-mm-dd`).
- A API retorna o preço previsto para o próximo dia.

### Monitoramento:
- Métricas de uso da API são coletadas usando **PrometheusMetrics**, incluindo:
  - Contador de requisições.
  - Latência das requisições.

---

## 🏗️ Arquitetura

### 1. Treinador do Modelo (`lstm_hb.py`)
**Processos principais:**
- Download de dados históricos da ação.
- Preparação dos dados e escalonamento.
- Treinamento de um modelo LSTM com parâmetros ajustáveis.
- Armazenamento do modelo treinado no **MLflow**.

**Funções importantes:**
- `train_model`: Treina o modelo LSTM.
- `evaluate_model`: Avalia o modelo em dados de teste.
- `predict_next_day`: Realiza previsões para o próximo dia.
- `save_scaler` e `load_scaler`: Salva e carrega o escalonador (scaler).

### 2. API Flask (`predict_stock.py`)
**Endpoints:**
- `/predict`: Recebe uma requisição `POST` com dados de entrada e retorna a previsão para o próximo dia.

**Integração:**
- Chama o método `predict_stock` para realizar a previsão usando o modelo treinado.

---

## 🛠️ Instalação

### Pré-requisitos
- Python 3.8+

## ⚙️ Configuração

- Certifique-se de que o **MLflow** está configurado corretamente.
- Configure um servidor **Prometheus** para monitorar as métricas.

---

## 🚀 Uso

### Treinamento do Modelo
- O treinamento do modelo é feito automaticamente na primeira execução do método `predict_stock`, caso não haja um modelo disponível.
- O treinamento também pode ser iniciado manualmente chamando a função `train_model`.

### Execução da API
1. Execute o servidor Flask:
   ```bash
   python predict_stock.py
   ```
2.Envie uma solicitação POST ao endpoint /predict
  ```bash
   curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"symbol": "PETR4.SA", "reference_date": "2024-11-30"}'
  ```


