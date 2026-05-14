import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(7, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 7), nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading Clean Data for Scaler...")
    df_clean = pd.read_csv("clean_telemetry.csv", parse_dates=[0], index_col=0)
    
    df_clean_stg2 = df_clean.copy()
    df_clean_stg2['E_total_calc'] = df_clean_stg2['E_grid'] + df_clean_stg2['E_DG']
    df_clean_stg2['EPR'] = df_clean_stg2['Q_prod'] / (df_clean_stg2['E_total_calc'] + 1e-5)
    df_clean_stg2['hour_sin'] = np.sin(2 * np.pi * df_clean_stg2.index.hour / 24)
    df_clean_stg2['hour_cos'] = np.cos(2 * np.pi * df_clean_stg2.index.hour / 24)
    df_clean_stg2['day_of_week'] = df_clean_stg2.index.dayofweek
    features_stg2 = ['E_grid', 'E_DG', 'Q_prod', 'EPR', 'hour_sin', 'hour_cos', 'day_of_week']
    scaler_stg2 = MinMaxScaler().fit(df_clean_stg2[features_stg2])

    print("Loading Autoencoder...")
    autoencoder = Autoencoder().to(device)
    autoencoder.load_state_dict(torch.load("autoencoder_verifier.pth", weights_only=True))
    autoencoder.eval()

    print("\n--- STAGE 1: Bypassing BiLSTM (Using Linear Interpolation) ---")
    df_faulted = pd.read_csv("faulted_telemetry.csv", parse_dates=[0], index_col=0)
    
    # ซ่อมข้อมูลด้วย Linear Interpolation แทน AI
    df_imputed = df_faulted.copy()
    nan_mask = df_imputed[['E_grid', 'E_DG']].isna()
    df_imputed.interpolate(method='linear', inplace=True)
    df_imputed.bfill(inplace=True)
    df_imputed.ffill(inplace=True)
    
    df_imputed['E_grid'] = df_imputed['E_grid'].clip(lower=0)
    df_imputed['E_DG'] = df_imputed['E_DG'].clip(lower=0)
    print(f"Repaired {nan_mask.sum().sum()} missing values using Math Baseline.")

    print("\n--- STAGE 2: Autoencoder Verification ---")
    df_imputed['E_total_calc'] = df_imputed['E_grid'] + df_imputed['E_DG']
    df_imputed['EPR'] = df_imputed['Q_prod'] / (df_imputed['E_total_calc'] + 1e-5)
    df_imputed['hour_sin'] = np.sin(2 * np.pi * df_imputed.index.hour / 24)
    df_imputed['hour_cos'] = np.cos(2 * np.pi * df_imputed.index.hour / 24)
    df_imputed['day_of_week'] = df_imputed.index.dayofweek

    x_scaled_stg2 = scaler_stg2.transform(df_imputed[features_stg2])
    x_tensor_stg2 = torch.tensor(x_scaled_stg2, dtype=torch.float32).to(device)

    with torch.no_grad():
        reconstructed = autoencoder(x_tensor_stg2)
        errors = torch.mean((x_tensor_stg2 - reconstructed)**2, dim=1).cpu().numpy()

    # ระบุค่า Threshold ของคุณ
    TAU = 0.005857  
    
    df_labels = pd.read_csv("anomaly_labels.csv", parse_dates=["timestamp"], index_col="timestamp")
    y_true = df_labels['anomaly_label'].values

    y_pred = (errors > TAU).astype(int)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n" + "="*50)
    print("🎯 BASELINE TEST: LINEAR INTERPOLATION + AUTOENCODER")
    print("="*50)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()