import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import joblib
import mlflow
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def main():
    global device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') # Using acceleration on Mac
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment('household-energy-consumption')
    with mlflow.start_run():
        lr = 0.01
        mlflow.log_metric('lr', lr)
    
        training, validation = load_dataset('household_power_consumption.txt')
        seq_n = 24
        X_train, y_train = create_sequences(training, seq_n)
        X_test, y_test = create_sequences(validation, seq_n)
        global train_loader, test_loader
        train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test)
        
        global model
        model = EnergyConsumptionModel(X_train.shape[2]).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr = lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)
        
        predict = train_model(model, train_loader, criterion, optimizer, scheduler, epochs=20)
        
        model_name = "pytorch_prediction_model.pkl"

        joblib.dump(value = predict, filename = model_name)
        mlflow.end_run()

if __name__ == '__main__':
    main()

def load_dataset(dataset, frac_training = 0.8):
    df = pd.read_csv(
        dataset,
        sep = ';', parse_dates = {'datetime':['Date', 'Time']}, low_memory = False, 
        na_values = ['nan', '?']
    )

    df.set_index('datetime', inplace = True)
    df.isna().values.any()

    df.interpolate(inplace = True) # Filling missing values by interpolation

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    scaled_data = pd.DataFrame(scaled_data,
        columns=df.columns, 
        index=df.index
    )
    training = scaled_data.sample(frac = frac_training, random_state = 123456)
    validation = scaled_data.drop(training.index)
    
    return training, validation

# Creating sequences for time-series forecasting
def create_sequences(data, seq_length):
    '''
    _summary_
    This function creates sequences with a fixed length to use as input in a model.
    It allows us to use the past n (as set by seq_length) observations to predict the n+1 one.
    The n observations are stored in xs, the n+1 one is stored in ys.

    Args:
        data (pd.DataFrame): input data
        seq_length (int): length of data used to predict the next value

    Returns:
        np.array, np.array: two arrays containing the sequences
    '''    
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i+seq_length)].values
        y = data.iloc[i+seq_length]['Global_active_power']
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def create_data_loaders(X_train, y_train, X_test, y_test):
    X_train_tensors = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensors = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensors = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensors = torch.tensor(y_test, dtype=torch.float32).to(device)
    train = TensorDataset(X_train_tensors, y_train_tensors)
    test = TensorDataset(X_test_tensors, y_test_tensors)
    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    test_loader = DataLoader(test, batch_size=64, shuffle=False)
    return train_loader, test_loader

# Model construction
class EnergyConsumptionModel(nn.Module):
    def __init__(self, input_dim):
        super(EnergyConsumptionModel, self).__init__()
        self.fc0 = nn.Linear(input_dim, 128)
        self.fc1 = nn.Linear(128, 64)
        self.lstm1 = nn.LSTM(64, 64, batch_first=True)
        self.attention_fc = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.25)
        self.bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def attention(self, lstm_out):
        weights = F.softmax(self.attention_fc(lstm_out), dim=1)
        attended_output = (lstm_out * weights).sum(dim=1)
        return attended_output
    
    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        lstm_out, _ = self.lstm1(x)
        attended_output = self.attention(lstm_out)
        x = self.dropout(attended_output)
        x = self.bn(x)
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out

def train_model(model, train_loader, criterion, optimizer, scheduler, epochs = 20):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        mlflow.log_metric('avg_loss', avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}')
        scheduler.step(avg_loss)
        with torch.inference_mode():
            actuals, predictions, mse = get_model_metrics(model, test_loader)
            mlflow.log_metric('mse', mse)

def get_model_metrics(model, test_loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predictions.append(outputs.squeeze().cpu().numpy())
            actuals.append(y_batch.cpu().numpy())
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    mse = mean_squared_error(actuals, predictions)
    print(f'Test MSE: {mse}')
    return actuals, predictions, mse

def compute_delta(start, end, chunk_size):
    actuals, predictions, mse = get_model_metrics(model, test_loader)
    delta = predictions - actuals
    delta_slice = delta[start:end]
    actuals_slice = actuals[start:end]
    positive_delta = np.maximum(delta_slice, 0)
    negative_delta = np.minimum(delta_slice, 0)
    x_values = range(start, end)

    # Grouping data in chunks
    num_chunks = len(delta_slice) // chunk_size
    averaged_actuals = [np.mean(actuals_slice[i * chunk_size:(i+1) * chunk_size]) for i in range(num_chunks)]
    averaged_deltas = [np.mean(delta_slice[i*chunk_size:(i+1)*chunk_size]) for i in range(num_chunks)]
    averaged_positive_deltas = [np.mean(positive_delta[i * chunk_size:(i+1) * chunk_size]) for i in range(num_chunks)]
    averaged_negative_deltas = [np.mean(negative_delta[i * chunk_size:(i+1) * chunk_size]) for i in range(num_chunks)]
    x_values = range(start, start + num_chunks * chunk_size, chunk_size)
    
    return averaged_actuals, averaged_deltas, averaged_positive_deltas, averaged_negative_deltas, x_values

def plot_nn_loss(start, end, chunk_size = 360):
    averaged_actuals, averaged_deltas, averaged_positive_deltas, averaged_negative_deltas, x_values = compute_delta(start, end, chunk_size = chunk_size)
    plt.figure(figsize=(35, 6))
    plt.bar(x_values, averaged_actuals, width = chunk_size, align='edge', label = f'Delta (averaged over {chunk_size} minutes)', color = 'dodgerblue')
    plt.bar(x_values, averaged_positive_deltas, width = chunk_size, align='edge', bottom = averaged_actuals, label = 'Positive Delta', color = 'orange')
    plt.bar(x_values, averaged_negative_deltas, width=chunk_size, align='edge', bottom=np.array(averaged_actuals) + np.array(averaged_positive_deltas), label='Negative Delta', color='red')
    plt.legend()
    plt.title('Energy Consumption Forecasting')
    plt.xlabel('Time (in minutes)')
    plt.ylabel('Delta in Energy Consumption')
    plt.savefig('nn_loss.png')

def plot_deltas(start, end, chunk_size = 360):
    averaged_actuals, averaged_deltas, averaged_positive_deltas, averaged_negative_deltas, x_values = compute_delta(start, end)
    plt.figure(figsize=(35, 6))
    plt.bar(x_values, averaged_deltas, width=chunk_size, align='edge', label=f'Delta (averaged over {chunk_size} minutes)')
    plt.legend()
    plt.title('Energy Consumption Forecasting')
    plt.xlabel('Time (in minutes)')
    plt.ylabel('Delta in Energy Consumption')
    plt.savefig('nn_loss.png')