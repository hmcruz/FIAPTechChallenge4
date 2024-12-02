import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import mlflow
import mlflow.pytorch
import requests
from requests.adapters import HTTPAdapter
import joblib


# Configurações gerais
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sequence_length = 20
batch_size = 64
learning_rate = 0.001
num_epochs = 50
input_size = 3  # 'Open', 'High', 'Low', 'Volume'
hidden_size = 50
num_layers = 2
output_size = 1  # Prever 'Close'
id_model = ""

# Baixar dados do yfinance
def fetch_stock_data(symbol, start_date, end_date, session):
    df = yf.download(symbol, start=start_date, end=end_date, session=session)
    print(df.head())
    df = df[['Open', 'High', 'Low', 'Close']]  # Selecionar colunas relevantes
    df = df.dropna()  # Remover valores nulos, se houver
    print(df.head())
    return df

# Dataset real
class StockDataset(Dataset):
    def __init__(self, data, seq_length, target_col):
        self.seq_length = seq_length
        self.data = data
        self.target_col = target_col

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length, :-1]  # Features
        y = self.data[idx + self.seq_length, self.target_col]  # Target (Close)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Preparar dados
def prepare_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

# Modelo LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Última saída
        return out

# Treinamento
def train_model(train_loader, test_loader, scaler):
    model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    mlflow.set_experiment("Stock Price Prediction")
    with mlflow.start_run() as run:
        mlflow.log_params({
            "sequence_length": sequence_length,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs
        })

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i, (sequences, labels) in enumerate(train_loader):
                sequences, labels = sequences.to(device), labels.to(device)

                # Forward pass
                outputs = model(sequences).squeeze()
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        # Captura o run_id
        run_id = run.info.run_id
        id_model = run_id
        print(f"Modelo salvo com ID do run: {id_model}")
        # Salvar modelo
        mlflow.pytorch.log_model(model, "stock_lstm_model_hb")

        evaluate_model(model, test_loader, scaler)

# Avaliação
def evaluate_model(model, test_loader, scaler):
    model.eval()
    test_loss = 0.0
    predictions, actuals = [], []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences).squeeze()
            test_loss += nn.MSELoss()(outputs, labels).item()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    rmse = np.sqrt(avg_test_loss)
    print(f"Test RMSE: {rmse:.4f}")
    mlflow.log_metric("test_rmse", rmse)


def test_model(model_path, new_data, sequence_length, scaler):
    """
    Testa o modelo LSTM em novos dados.
    
    Args:
        model_path (str): Caminho do modelo salvo pelo MLflow.
        new_data (ndarray): Dados novos em formato escalarizado.
        sequence_length (int): Comprimento das sequências.
        scaler (MinMaxScaler): Escalador usado no treinamento para reverter os valores.
    """

    print("Tamanho de new_data:", len(new_data))
    print("sequence_length " + str(sequence_length))

    # Carregar o modelo salvo
    model = mlflow.pytorch.load_model(model_path).to(device)
    model.eval()

    print(model.eval())

    # Preparar os dados em sequências
    sequences = []
    for i in range(len(new_data) - sequence_length):
        seq = new_data[i:i + sequence_length, :-1]  # Pegando features
        sequences.append(seq)

    # Transformar para tensor 3D
    sequences = torch.tensor(sequences, dtype=torch.float32).to(device)
    print("Shape das sequências:", sequences.shape)  # Verificar a forma

    # Fazer previsões
    with torch.no_grad():
        predictions = model(sequences).cpu().numpy()

    # Reverter o escalonamento para o valor original
    predictions = scaler.inverse_transform(
        np.hstack([np.zeros((predictions.shape[0], new_data.shape[1] - 1)), predictions.reshape(-1, 1)])
    )[:, -1]

    return predictions

def predict_next_day(model_path, symbol, start_date, reference_date, sequence_length, scaler, session):
    """
    Preveja o valor da ação no dia seguinte usando o modelo treinado.

    Args:
        model_path (str): Caminho do modelo treinado no MLflow.
        symbol (str): Símbolo da ação (ex: "DIS").
        reference_date (str): Data de referência (formato "YYYY-MM-DD").
        sequence_length (int): Comprimento da sequência de dados.
        scaler (MinMaxScaler): Escalador usado no treinamento.
        session (requests.Session): Sessão de rede para fazer a requisição de dados.
    
    Returns:
        float: Previsão do valor da ação no dia seguinte.
    """
    # Baixar dados até a data de referência
    end_date = reference_date
    df = fetch_stock_data(symbol, start_date, end_date, session)

    # Preparar os dados (escalonando)
    data, _ = prepare_data(df)
    
    # Selecionar os últimos 'sequence_length' dados para fazer a previsão
    new_data = data[-sequence_length:]
    
    # Carregar o modelo treinado
    model = mlflow.pytorch.load_model(model_path).to(device)
    model.eval()

    # Preparar os dados em sequência para o modelo
    sequences = [new_data[:, :-1]]  # Pegando todas as features, menos a coluna de "Close"
    sequences = torch.tensor(sequences, dtype=torch.float32).to(device)
    
    # Fazer a previsão para o próximo dia
    with torch.no_grad():
        prediction = model(sequences).cpu().numpy()
    
    prediction_rescaled = scaler.inverse_transform(
        np.hstack([np.zeros((prediction.shape[0], new_data.shape[1] - 1)), prediction.reshape(-1, 1)])
    )[:, -1]

    # Retornar o valor previsto para o próximo dia
    return prediction_rescaled[0]

def save_scaler(scaler, file_path):
    """
    Salva o scaler em um arquivo para uso futuro.
    
    Args:
        scaler (MinMaxScaler): O scaler a ser salvo.
        file_path (str): Caminho do arquivo onde o scaler será salvo.
    """
    joblib.dump(scaler, file_path)
    print(f"Scaler salvo em {file_path}")

def load_scaler(file_path):
    """
    Carrega o scaler de um arquivo salvo anteriormente.
    
    Args:
        file_path (str): Caminho do arquivo onde o scaler foi salvo.
    
    Returns:
        MinMaxScaler: O scaler carregado.
    """
    scaler = joblib.load(file_path)
    print(f"Scaler carregado de {file_path}")
    return scaler

#metodo para uso da API
def predict_stock(symbol, reference_date):
    #1 - PRIMEIRO, TREINAR MODELo COM OS DADOS RECEBIDO
    #apenas para uso na rede CVP
    proxies = {
        "http": "http://CVP13876:Leinha231092!@192.168.100.110:80",
        "https": "http://CVP13876:Leinha231092!@192.168.100.110:80"
    }

    session = requests.Session()
    session.proxies.update(proxies)
    sequence_length = 20
    start_date = "2018-01-01"

    # Configurações do dataset
    end_date= str(reference_date)
    print(end_date)

    # Baixar e preparar dados
    df = fetch_stock_data(symbol, start_date, end_date, session)
    data, scaler = prepare_data(df)

    #salvando scaler para uso futuro
    save_scaler(scaler, "scaler.pkl")

    # Dividir dados em treino e teste
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    # Criar datasets e DataLoaders
    train_dataset = StockDataset(train_data, sequence_length, target_col=3)  # 'Close'
    test_dataset = StockDataset(test_data, sequence_length, target_col=3)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_model(train_loader, test_loader, scaler)

    #2 COM O MODELO TREINADO, PARA O NOVO SIMBOLO ENVIADO, REALIZA A PREVISAO
    if id_model != "":
        model_path = f"runs:/{id_model}/stock_lstm_model_hb"  # Caminho do modelo
    else:
        model_path = "runs:/58081ebef4fa438b8c71c9ed4622e564/stock_lstm_model_hb"  # Caminho do modelo

    #scaler = scaler  # Escalador do treinamento
    scaler = load_scaler("scaler.pkl")

    # Chamar o método para prever o próximo valor
    predicted_price = predict_next_day(model_path, symbol, start_date, reference_date, sequence_length, scaler, session)

    return predicted_price


# Main
if __name__ == "__main__":
    train_new_model = True
    test_model_trained = False  
    test_predict_next_day = True

    #apenas para uso na rede CVP
    proxies = {
        "http": "http://CVP13876:Leinha231092!@192.168.100.110:80",
        "https": "http://CVP13876:Leinha231092!@192.168.100.110:80"
    }

    session = requests.Session()
    session.proxies.update(proxies)

    symbol = "PETR4.SA"  # Exemplo de ação  
    start_date = "2018-01-01"
    reference_date = "2020-12-01"  # Data de referência

    if train_new_model:
        print("<<TREINANDO NOVO MODLELO>>")

        # Configurações do dataset
        end_date = reference_date
    
        # Baixar e preparar dados
        print("BAIXANDO DADOS")
        df = fetch_stock_data(symbol, start_date, end_date, session)
        print("PREPARANDO DADOS")
        data, scaler = prepare_data(df)

        #salvando scaler para uso futuro
        save_scaler(scaler, "scaler.pkl")

        # Dividir dados em treino e teste
        train_size = int(len(data) * 0.8)
        train_data, test_data = data[:train_size], data[train_size:]
        print(f"DADOS TREINO {len(train_data)}")
        print(f"DADOS TESTE {len(test_data)}")

        # Criar datasets e DataLoaders
        train_dataset = StockDataset(train_data, sequence_length, target_col=3)  # 'Close'
        test_dataset = StockDataset(test_data, sequence_length, target_col=3)
        print(f"DATASET TREINO {len(train_dataset)}")
        print(f"DATASET TESTE {len(test_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"LOADER TREINO {len(train_loader)}")
        print(f"LOADER TESTE {len(train_loader)}")

        train_model(train_loader, test_loader, scaler)

    if test_model_trained:
        print("<<TESTANDO MODELO TREINADO>>")

        # Configurações do dataset
        end_date = reference_date
    
        # Baixar e preparar dados
        print("BAIXANDO DADOS")
        new_df  = fetch_stock_data(symbol, start_date, end_date, session)
        print("PREPARANDO DADOS")
        new_data, scaler = prepare_data(new_df)

        # Caminho do modelo salvo
        #model_path = "runs:/b74e95ce39b84caf9316dd7261ac8421/stock_lstm_model"
        model_path = "runs:/58081ebef4fa438b8c71c9ed4622e564/stock_lstm_model_hb"

        sequence_length = 2
        # Testar o modelo
        predictions = test_model(model_path, new_data, sequence_length, scaler)

        # Exibir resultados
        print("Previsões para novos dados:")
        print(predictions)

        # Opcional: Comparar previsões com os valores reais (se disponíveis)
        if "Close" in new_df.columns:
            actuals = new_df["Close"].values[sequence_length:]
            print("\nValores reais:")
            print(actuals)

            print(new_df.describe())

            # Exibir métricas de desempenho
            rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
            print(f"\nRMSE das previsões: {rmse:.4f}")

    if test_predict_next_day:
        sequence_length = 20  # Tamanho da sequência
        if id_model != "":
            model_path = f"runs:/{id_model}/stock_lstm_model_hb"  # Caminho do modelo
        else:
            model_path = "runs:/58081ebef4fa438b8c71c9ed4622e564/stock_lstm_model_hb"  # Caminho do modelo

        #scaler = scaler  # Escalador do treinamento
        scaler = load_scaler("scaler.pkl")

        # Chamar o método para prever o próximo valor
        predicted_price = predict_next_day(model_path, symbol, start_date, reference_date, sequence_length, scaler, session)

        print(f"A previsão do valor da ação para o dia seguinte ({reference_date}) é: {predicted_price:.2f}")