import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from evt_threshold import calculate_evt_threshold

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

def evaluate_preds(y_true, y_pred, name="Model"):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "Model": name,
        "Precision": f"{p:.4f}",
        "Recall": f"{r:.4f}",
        "F1-Score": f"{f1:.4f}",
    }

def main():
    print("==================================================")
    print("🥊 RUNNING BENCHMARK BASELINES (STAGES 1 & 2)")
    print("==================================================\n")

    # 1. โหลดข้อมูล
    df_clean = pd.read_csv("clean_telemetry.csv", parse_dates=[0], index_col=0)
    df_faulted = pd.read_csv("faulted_telemetry.csv", parse_dates=[0], index_col=0)
    df_labels = pd.read_csv(
        "anomaly_labels.csv", parse_dates=["timestamp"], index_col="timestamp"
    )
    y_true = df_labels["anomaly_label"].values

    results = []

    # ---------------------------------------------------------
    # BASELINE 1: Linear Interpolation + Z-Score Threshold
    # ---------------------------------------------------------
    print("Running Baseline 1: Linear Interp + Z-Score...")
    df_b1 = df_faulted.copy()
    df_b1.interpolate(method="linear", inplace=True)
    df_b1.bfill(inplace=True)
    df_b1.ffill(inplace=True)

    # คำนวณ EPR แล้วหา Z-Score (ถ้าแกว่งเกิน 3 SD ถือว่าโกง)
    epr_b1 = df_b1["Q_prod"] / (df_b1["E_grid"] + df_b1["E_DG"] + 1e-5)
    z_scores = np.abs((epr_b1 - epr_b1.mean()) / epr_b1.std())
    thresh_b1 = np.percentile(z_scores, 99)
    pred_b1 = (z_scores > thresh_b1).astype(int)

    results.append(evaluate_preds(y_true, pred_b1, "Linear Interp + Z-Score (Stat)"))

    # ---------------------------------------------------------
    # BASELINE 2: k-NN Imputer + Isolation Forest (ML Classic)
    # ---------------------------------------------------------
    print("Running Baseline 2: k-NN Imputer + Isolation Forest...")
    features = ["E_grid", "E_DG", "Q_prod", "T_amb", "GHI"]

    # ใช้ k-NN เดาค่าแหว่งจาก 5 ชั่งโมงที่คล้ายกันที่สุด
    knn = KNNImputer(n_neighbors=5)
    imputed_matrix = knn.fit_transform(df_faulted[features])
    df_b2 = pd.DataFrame(imputed_matrix, columns=features, index=df_faulted.index)

    # ส่งเข้า Isolation Forest ตรวจจับสิ่งแปลกปลอม
    iso = IsolationForest(contamination='auto', random_state=42)
    scaler = StandardScaler()
    scaled_b2 = scaler.fit_transform(df_b2[["E_grid", "E_DG", "Q_prod"]])

    iso_scores = iso.fit(scaled_b2).decision_function(scaled_b2)
    thresh_b2 = np.percentile(iso_scores, 1)
    pred_b2 = (iso_scores < thresh_b2).astype(int)

    results.append(
        evaluate_preds(y_true, pred_b2, "k-NN + Isolation Forest (Classic ML)")
    )

    # ---------------------------------------------------------
    # OUR PROPOSED MODEL (BiLSTM + Autoencoder + EVT/POT)
    # ---------------------------------------------------------
    print("Running Proposed Model: BiLSTM + Autoencoder (EVT/POT)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Fit Scalers on clean data
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

    # 2. Load Models
    bilstm = BiLSTMImputer().to(device)
    bilstm.load_state_dict(torch.load("bilstm_imputer.pth", map_location=device, weights_only=True))
    bilstm.eval()

    autoencoder = Autoencoder().to(device)
    autoencoder.load_state_dict(torch.load("autoencoder_verifier.pth", map_location=device, weights_only=True))
    autoencoder.eval()

    # 3. Impute
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

    # 4. Autoencoder verification
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

    # 5. EVT Threshold
    TAU = calculate_evt_threshold("autoencoder_verifier.pth", q_u=0.98, p_target=0.01)
    pred_proposed = (errors > TAU).astype(int)

    results.append(
        evaluate_preds(y_true, pred_proposed, "★ Proposed (BiLSTM + Autoencoder POT)")
    )

    # ปรินต์ตารางเปรียบเทียบ
    df_res = pd.DataFrame(results)
    print("\n" + "=" * 65)
    print("🎯 BENCHMARK COMPARISON TABLE (GAP CLOSURE 2/4)")
    print("=" * 65)
    print(df_res.to_string(index=False))
    print("=" * 65)
    print("\n✅ Reviewer Note: We now have justifiable baselines for Section 4.")


if __name__ == "__main__":
    main()
