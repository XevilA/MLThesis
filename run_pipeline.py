import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# ==========================================
# 1. โครงสร้างโมเดล Stage 1 & 2
# ==========================================

class BiLSTMImputer(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=2, output_dim=2, dropout=0.2):
        super(BiLSTMImputer, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, 
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)

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

# ==========================================
# 2. ฟังก์ชันหลัก
# ==========================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Final Pipeline (Strict Academic Mode) on {device}...\n")

    # --- 1. เตรียม Scaler ---
    df_clean = pd.read_csv("clean_telemetry.csv", parse_dates=[0], index_col=0)
    
    features_stg1 = ['E_grid', 'E_DG', 'Q_prod', 'T_amb', 'GHI']
    scaler_x_stg1 = MinMaxScaler().fit(df_clean[features_stg1])
    scaler_y_stg1 = MinMaxScaler().fit(df_clean[['E_grid', 'E_DG']])

    df_clean_stg2 = df_clean.copy()
    df_clean_stg2['E_total_calc'] = df_clean_stg2['E_grid'] + df_clean_stg2['E_DG']
    df_clean_stg2['EPR'] = df_clean_stg2['Q_prod'] / (df_clean_stg2['E_total_calc'] + 1e-5)
    df_clean_stg2['hour_sin'] = np.sin(2 * np.pi * df_clean_stg2.index.hour / 24)
    df_clean_stg2['hour_cos'] = np.cos(2 * np.pi * df_clean_stg2.index.hour / 24)
    df_clean_stg2['day_of_week'] = df_clean_stg2.index.dayofweek
    features_stg2 = ['E_grid', 'E_DG', 'Q_prod', 'EPR', 'hour_sin', 'hour_cos', 'day_of_week']
    scaler_stg2 = MinMaxScaler().fit(df_clean_stg2[features_stg2])

    # --- 2. โหลดโมเดล ---
    print("Loading Trained Models...")
    bilstm = BiLSTMImputer().to(device)
    bilstm.load_state_dict(torch.load("bilstm_imputer.pth", weights_only=True))
    bilstm.eval()

    autoencoder = Autoencoder().to(device)
    autoencoder.load_state_dict(torch.load("autoencoder_verifier.pth", weights_only=True))
    autoencoder.eval()

    # --- 3. ซ่อมข้อมูลด้วย BiLSTM (Sliding Window) ---
    print("--- STAGE 1: BiLSTM Imputation ---")
    df_faulted = pd.read_csv("faulted_telemetry.csv", parse_dates=[0], index_col=0)
    df_for_bilstm = df_faulted.copy()
    nan_mask = df_for_bilstm[['E_grid', 'E_DG']].isna()
    df_for_bilstm.fillna(0, inplace=True) 

    x_scaled_stg1 = scaler_x_stg1.transform(df_for_bilstm[features_stg1])
    
    seq_len = 168
    preds_list = []
    for i in range(0, len(x_scaled_stg1), seq_len):
        chunk = x_scaled_stg1[i : i + seq_len]
        if len(chunk) < seq_len:
            pad_size = seq_len - len(chunk)
            chunk = np.pad(chunk, ((0, pad_size), (0, 0)), mode='constant')
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = bilstm(chunk_tensor)
        actual_len = min(seq_len, len(x_scaled_stg1) - i)
        preds_list.append(pred.squeeze(0).cpu().numpy()[:actual_len])
    
    preds_stg1_np = np.concatenate(preds_list, axis=0)
    preds_original_scale = scaler_y_stg1.inverse_transform(preds_stg1_np)

    df_imputed = df_faulted.copy()
    df_imputed.loc[nan_mask['E_grid'], 'E_grid'] = preds_original_scale[nan_mask['E_grid'], 0]
    df_imputed.loc[nan_mask['E_DG'], 'E_DG'] = preds_original_scale[nan_mask['E_DG'], 1]
    df_imputed['E_grid'] = df_imputed['E_grid'].clip(lower=0)
    df_imputed['E_DG'] = df_imputed['E_DG'].clip(lower=0)
    print(f"Successfully repaired {nan_mask.sum().sum()} missing values.")

    # --- 4. จับผิดข้อมูลด้วย Autoencoder ---
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

    # --- 5. ประเมินผลลัพธ์ (Strict Mode) ---
    # ใช้ค่า 99th Percentile Threshold จากตอนรัน train_autoencoder.py เป๊ะๆ
    TAU = 0.003192 
    
    df_labels = pd.read_csv("anomaly_labels.csv", parse_dates=["timestamp"], index_col="timestamp")
    y_true = df_labels['anomaly_label'].values

    y_pred = (errors > TAU).astype(int)
    
    type_1 = np.sum((errors > TAU) & (errors <= 2 * TAU))
    type_2 = np.sum((errors > 2 * TAU) & (errors <= 5 * TAU))
    type_3 = np.sum(errors > 5 * TAU)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n" + "="*50)
    print("🎯 FINAL RESULTS: INTEGRATED PIPELINE (ACADEMIC STRICT)")
    print("="*50)
    print(f"Total Timesteps:     {len(y_true)}")
    print(f"Actual Anomalies:    {np.sum(y_true)} (จงใจโกง)")
    print(f"Detected Anomalies:  {np.sum(y_pred)} (AI จับได้)")
    print("-" * 50)
    print("Anomaly Classification (Taxonomy):")
    print(f"  [Type I] Marginal (TAU < RE <= 2TAU):     {type_1} ชั่วโมง")
    print(f"  [Type II] Structural (2TAU < RE <= 5TAU): {type_2} ชั่วโมง")
    print(f"  [Type III] Critical (RE > 5TAU):          {type_3} ชั่วโมง")
    print("-" * 50)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*50)
    print("\nNote: Threshold (TAU) is strictly fixed at 0.003192 based on the 99th percentile of validation error.")

if __name__ == "__main__":
    main()