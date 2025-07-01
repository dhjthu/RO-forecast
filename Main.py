import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from Tools import load_and_extract_data, preprocess_data, setup_seed
from Models import CNN, RNN, GRU, LSTM, ConvLSTM


setup_seed(420)
file_path = 'Data-KDE-f1.xlsx'
sequence_length = 150
prediction_length = 7
hidden_size = 100
num_layers = 3
output_size = prediction_length
num_epochs = 500
learning_rate = 0.001

data_x, data_y, dates = load_and_extract_data(file_path)
data_x, data_y_scaled, scaler_x, scaler_y = preprocess_data(data_x, data_y, standardize=True)

split_index = int(0.7 * len(data_x))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_forecast(model, optimizer, criterion, data_x, data_y, start_idx, end_idx):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        count = 0
        for i in range(start_idx, end_idx - sequence_length - prediction_length + 1):
            x = data_x[i : i + sequence_length]
            y = data_y[i + sequence_length : i + sequence_length + prediction_length]
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            y_t = torch.tensor(y, dtype=torch.float32).view(1, -1).to(device)

            pred = model(x_t)
            loss = criterion(pred, y_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            count += 1

        if (epoch + 1) % 50 == 0:
            print(f'  Epoch [{epoch+1}/{num_epochs}]  Avg Loss: {epoch_loss/count:.4f}')

def eval_forecast(model, data_x, data_y, start_idx, end_idx):
    model.eval()
    preds, trues, dt = [], [], []
    with torch.no_grad():
        for i in range(start_idx, end_idx - sequence_length - prediction_length + 1, prediction_length):
            x = data_x[i : i + sequence_length]
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            p = model(x_t).cpu().numpy().flatten()

            trues.extend(data_y[i + sequence_length : i + sequence_length + prediction_length])
            preds.extend(p)
            dt.extend(dates[i + sequence_length : i + sequence_length + prediction_length])

            for j in range(prediction_length):
                idx = i + sequence_length + j
                if idx < len(data_x):
                    data_x[idx, -1] = p[j]

    preds = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    trues = scaler_y.inverse_transform(np.array(trues).reshape(-1, 1)).flatten()
    r2 = r2_score(trues, preds)
    return pd.DataFrame({'Date': dt, 'Actual': trues, 'Predicted': preds}), r2

models = {
    'CNN':      CNN(data_x.shape[1], hidden_size, num_layers, sequence_length, output_size, 1, device),
    'RNN':      RNN(data_x.shape[1], hidden_size, num_layers, output_size, 1, device),
    'GRU':      GRU(data_x.shape[1], hidden_size, num_layers, output_size, 1, device),
    'LSTM':     LSTM(data_x.shape[1], hidden_size, num_layers, output_size, 1, device),
    'ConvLSTM': ConvLSTM(data_x.shape[1], hidden_size, num_layers, output_size, 1, device),
}

criterion = nn.MSELoss()

all_results = {}
for name, model in models.items():
    print(f'==== Training {name} ====')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_forecast(model, optimizer, criterion,
                   data_x.copy(), data_y_scaled.copy(),
                   start_idx=0, end_idx=split_index)
    
    results_df, r2 = eval_forecast(model,
                                   data_x.copy(), data_y_scaled.copy(),
                                   start_idx=split_index,
                                   end_idx=len(data_x))
    print(f'{name} Test R2: {r2:.4f}\n')
    all_results[name] = results_df

with pd.ExcelWriter('model_predictions_train_test_split.xlsx') as writer:
    for name, df in all_results.items():
        df.to_excel(writer, sheet_name=f'{name}_Test', index=False)