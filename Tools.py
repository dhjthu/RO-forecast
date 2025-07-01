import torch
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from typing import Dict
from sklearn.metrics import r2_score, mean_absolute_error

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_and_extract_data(file_path):
    data = pd.read_excel(file_path)
    data_x = data.iloc[:, 1:-1].values
    data_y = data.iloc[:, -1].values
    dates = data.iloc[:, 0].values
    return data_x, data_y, dates

def preprocess_data(data_x, data_y, standardize=True):
    if standardize:
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        data_x = scaler_x.fit_transform(data_x)
        data_y = scaler_y.fit_transform(data_y.reshape(-1, 1)).flatten()
    else:
        scaler_x, scaler_y = None, None
    return data_x, data_y, scaler_x, scaler_y

def compare_models(model_results: Dict[str, pd.DataFrame]) -> str:

    metrics = {}

    for name, df in model_results.items():
        tmp = df.copy()
        tmp['Error'] = (tmp['Predicted'] - tmp['Actual']).abs()
        print(f'\n=== {name} Error ===')
        print(tmp[['Date','Actual','Predicted','Error']].to_string(index=False))

    for name, df in model_results.items():
        r2  = r2_score(df['Actual'], df['Predicted'])
        mae = mean_absolute_error(df['Actual'], df['Predicted'])
        metrics[name] = {'R2': r2, 'MAE': mae}

    print('\n=== Total ===')
    for name, m in metrics.items():
        print(f"{name:10s}  R2: {m['R2']:.4f}  MAE: {m['MAE']:.4f}")

    best_model = max(metrics.items(), key=lambda x: x[1]['R2'])[0]
    best_r2  = metrics[best_model]['R2']
    best_mae = metrics[best_model]['MAE']
    print(f'\n>>> Best model：{best_model}（R2={best_r2:.4f}，MAE={best_mae:.4f}）')

    return best_model