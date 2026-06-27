import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import genpareto
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler

from evt_threshold import calculate_evt_threshold

# ==========================================
# 1. โครงสร้างโมเดล Stage 1 & 2
# ==========================================


class BiLSTMImputer(nn.Module):
    def __init__(
        self, input_dim=5, hidden_dim=128, num_layers=2, output_dim=2, dropout=0.2
    ):
        super(BiLSTMImputer, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(7, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 7),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ==========================================
# 2. ฟังก์ชันคณิตศาสตร์ POT (Section 4.2.3 ในเปเปอร์)
# ==========================================


def calculate_pot_threshold(normal_errors, q=1e-3):
    """
    คำนวณหา Dynamic Threshold ด้วยทฤษฎี Extreme Value Theory (POT)
    - normal_errors: ค่า Error จากข้อมูล Clean ที่ไม่เคยโดนโกง
    - q: ระดับความเสี่ยงที่ยอมรับได้ (1e-3 คือยอมให้เตือนผิดพลาดได้แค่ 0.1%)
    """
    print(
        f"\nNote: Threshold is dynamically calibrated using Extreme Value Theory (EVT/POT) based on clean_telemetry.csv validation error."
    )
    # 1. หา Base Threshold (u) ที่เปอร์เซ็นไทล์ 95
    u = np.percentile(normal_errors, 95)

    # 2. ดึงเฉพาะส่วนที่ "ล้น" เกิน u ออกมา (Excesses)
    excesses = normal_errors[normal_errors > u] - u

    if len(excesses) < 10:
        return np.percentile(normal_errors, 99)  # Fallback ป้องกันกราฟแคบเกินไป

    # 3. Fit พารามิเตอร์ของ Generalized Pareto Distribution (GPD)
    c, loc, scale = genpareto.fit(excesses, floc=0)

    # 4. เข้าสมการคำนวณ EVT Quantile (z_q)
    n = len(normal_errors)
    n_u = len(excesses)

    if abs(c) < 1e-6:  # กรณีค่า c เข้าใกล้ศูนย์ (Exponential Tail)
        z_q = u - scale * np.log((q * n) / n_u)
    else:  # กรณี GPD ปกติ
        z_q = u + (scale / c) * (((q * n) / n_u) ** (-c) - 1)

    return max(float(z_q), u)


# ==========================================
# 3. ฟังก์ชันหลัก
# ==========================================


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Final Pipeline [EVT / POT Mode] on {device}...\n")

    # --- 1. เตรียม Scaler ---
    df_clean = pd.read_csv("clean_telemetry.csv", parse_dates=[0], index_col=0)

    features_stg1 = ["E_grid", "E_DG", "Q_prod", "T_amb", "GHI"]
    scaler_x_stg1 = MinMaxScaler().fit(df_clean[features_stg1])
    scaler_y_stg1 = MinMaxScaler().fit(df_clean[["E_grid", "E_DG"]])

    df_clean_stg2 = df_clean.copy()
    df_clean_stg2["E_total_calc"] = df_clean_stg2["E_grid"] + df_clean_stg2["E_DG"]
    df_clean_stg2["EPR"] = df_clean_stg2["Q_prod"] / (
        df_clean_stg2["E_total_calc"] + 1e-5
    )
    df_clean_stg2["hour_sin"] = np.sin(2 * np.pi * df_clean_stg2.index.hour / 24)
    df_clean_stg2["hour_cos"] = np.cos(2 * np.pi * df_clean_stg2.index.hour / 24)
    df_clean_stg2["day_of_week"] = df_clean_stg2.index.dayofweek
    features_stg2 = [
        "E_grid",
        "E_DG",
        "Q_prod",
        "EPR",
        "hour_sin",
        "hour_cos",
        "day_of_week",
    ]
    scaler_stg2 = MinMaxScaler().fit(df_clean_stg2[features_stg2])

    # --- 2. โหลดโมเดล ---
    print("Loading Trained Models...")
    bilstm = BiLSTMImputer().to(device)
    bilstm.load_state_dict(torch.load("bilstm_imputer.pth", weights_only=True))
    bilstm.eval()

    autoencoder = Autoencoder().to(device)
    autoencoder.load_state_dict(
        torch.load("autoencoder_verifier.pth", weights_only=True)
    )
    autoencoder.eval()

    # --- 3. [NEW] คำนวณ POT THRESHOLD จาก Clean Data ---
    print("Calibrating EVT / POT Threshold on Clean Baseline...")
    x_scaled_clean_stg2 = scaler_stg2.transform(df_clean_stg2[features_stg2])
    x_tensor_clean_stg2 = torch.tensor(x_scaled_clean_stg2, dtype=torch.float32).to(
        device
    )

    with torch.no_grad():
        recon_clean = autoencoder(x_tensor_clean_stg2)
        clean_errors = (
            torch.mean((x_tensor_clean_stg2 - recon_clean) ** 2, dim=1).cpu().numpy()
        )

    TAU_POT = calculate_pot_threshold(clean_errors, q=1e-3)
    print(f"*** STRICT POT THRESHOLD (tau_evt): {TAU_POT:.6f} ***\n")

    # --- 4. ซ่อมข้อมูลด้วย BiLSTM (Sliding Window) & ประเมินผล (Multi-Seed Benchmark) ---
    SEEDS = [42, 43, 44, 45, 46]
    metrics = {"precision": [], "recall": [], "f1": []}
    
    for seed in SEEDS:
        print(f"\n{'='*50}\n🚀 RUNNING PIPELINE WITH SEED: {seed}\n{'='*50}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        print("--- STAGE 1: BiLSTM Imputation (Monte Carlo Dropout) ---")
        df_faulted = pd.read_csv("faulted_telemetry.csv", parse_dates=[0], index_col=0)
        df_for_bilstm = df_faulted.copy()
        nan_mask = df_for_bilstm[["E_grid", "E_DG"]].isna()
        df_for_bilstm.fillna(0, inplace=True)

        x_scaled_stg1 = scaler_x_stg1.transform(df_for_bilstm[features_stg1])

        seq_len = 168
        K = 50
        preds_list_mean = []
        preds_list_lower = []
        preds_list_upper = []
        
        bilstm.train()  # Enable stochastic dropout for MC Dropout
        
        for i in range(0, len(x_scaled_stg1), seq_len):
            chunk = x_scaled_stg1[i : i + seq_len]
            if len(chunk) < seq_len:
                pad_size = seq_len - len(chunk)
                chunk = np.pad(chunk, ((0, pad_size), (0, 0)), mode="constant")
            chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
            
            mc_preds = []
            with torch.no_grad():
                for _ in range(K):
                    pred = bilstm(chunk_tensor)
                    mc_preds.append(pred.squeeze(0).cpu().numpy())
            
            mc_preds = np.array(mc_preds)
            mean_pred = np.mean(mc_preds, axis=0)
            lower_pred = np.percentile(mc_preds, 2.5, axis=0)
            upper_pred = np.percentile(mc_preds, 97.5, axis=0)
            
            actual_len = min(seq_len, len(x_scaled_stg1) - i)
            preds_list_mean.append(mean_pred[:actual_len])
            preds_list_lower.append(lower_pred[:actual_len])
            preds_list_upper.append(upper_pred[:actual_len])

        bilstm.eval()

        preds_stg1_np = np.concatenate(preds_list_mean, axis=0)
        preds_lower_np = np.concatenate(preds_list_lower, axis=0)
        preds_upper_np = np.concatenate(preds_list_upper, axis=0)
        
        preds_original_scale = scaler_y_stg1.inverse_transform(preds_stg1_np)
        preds_lower_scale = scaler_y_stg1.inverse_transform(preds_lower_np)
        preds_upper_scale = scaler_y_stg1.inverse_transform(preds_upper_np)

        df_imputed = df_faulted.copy()
        df_imputed.loc[nan_mask["E_grid"], "E_grid"] = preds_original_scale[nan_mask["E_grid"], 0]
        df_imputed.loc[nan_mask["E_DG"], "E_DG"] = preds_original_scale[nan_mask["E_DG"], 1]
        df_imputed["E_grid"] = df_imputed["E_grid"].clip(lower=0)
        df_imputed["E_DG"] = df_imputed["E_DG"].clip(lower=0)
        
        print(f"Successfully repaired {nan_mask.sum().sum()} missing values.")

        # Calculate PICP and ACE
        picp_list = []
        ace_list = []
        for col_idx, col in enumerate(["E_grid", "E_DG"]):
            mask = nan_mask[col].values
            if np.sum(mask) == 0:
                continue
            # Use ground truth from df_clean for artificially faulted indices
            y_true = df_clean.loc[df_faulted.index, col].values[mask]
            lower_bounds = preds_lower_scale[mask, col_idx]
            upper_bounds = preds_upper_scale[mask, col_idx]
            
            in_bounds = (y_true >= lower_bounds) & (y_true <= upper_bounds)
            picp = np.mean(in_bounds)
            picp_list.append(picp)
            
            # Nominal Confidence is 95% (0.95)
            ace = picp - 0.95
            ace_list.append(ace)

        avg_picp = np.mean(picp_list) if picp_list else 0
        avg_ace = np.mean(ace_list) if ace_list else 0
        print(f"MC Dropout PICP (95% CI): {avg_picp:.4f}")
        print(f"MC Dropout ACE:           {avg_ace:.4f}")

        # --- 5. จับผิดข้อมูลด้วย Autoencoder ---
        print("\n--- STAGE 2: Autoencoder Verification ---")
        df_imputed["E_total_calc"] = df_imputed["E_grid"] + df_imputed["E_DG"]
        df_imputed["EPR"] = df_imputed["Q_prod"] / (df_imputed["E_total_calc"] + 1e-5)
        df_imputed["hour_sin"] = np.sin(2 * np.pi * df_imputed.index.hour / 24)
        df_imputed["hour_cos"] = np.cos(2 * np.pi * df_imputed.index.hour / 24)
        df_imputed["day_of_week"] = df_imputed.index.dayofweek

        x_scaled_stg2 = scaler_stg2.transform(df_imputed[features_stg2])
        x_tensor_stg2 = torch.tensor(x_scaled_stg2, dtype=torch.float32).to(device)

        with torch.no_grad():
            reconstructed = autoencoder(x_tensor_stg2)
            errors = torch.mean((x_tensor_stg2 - reconstructed) ** 2, dim=1).cpu().numpy()

        # --- 6. ประเมินผลลัพธ์ (ใช้ TAU_POT) ---
        TAU = calculate_evt_threshold(
            "autoencoder_verifier.pth", q_u=0.98, p_target=0.01, verbose=True
        )

        df_labels = pd.read_csv(
            "anomaly_labels.csv", parse_dates=["timestamp"], index_col="timestamp"
        )
        y_true_labels = df_labels["anomaly_label"].values

        y_pred = (errors > TAU).astype(int)

        type_1 = np.sum((errors > TAU) & (errors <= 2 * TAU))
        type_2 = np.sum((errors > 2 * TAU) & (errors <= 5 * TAU))
        type_3 = np.sum(errors > 5 * TAU)

        precision = precision_score(y_true_labels, y_pred, zero_division=0)
        recall = recall_score(y_true_labels, y_pred, zero_division=0)
        f1 = f1_score(y_true_labels, y_pred, zero_division=0)

        print("\n" + "-" * 50)
        print("Anomaly Classification (Taxonomy):")
        print(f"  [Type I] Marginal:     {type_1} ชั่วโมง")
        print(f"  [Type II] Structural:  {type_2} ชั่วโมง")
        print(f"  [Type III] Critical:   {type_3} ชั่วโมง")
        print("-" * 50)
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("-" * 50)
        
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1"].append(f1)

    print("\n" + "=" * 50)
    print("🎯 FINAL MULTI-SEED BENCHMARK (5 RUNS)")
    print("=" * 50)
    print(f"Precision: {np.mean(metrics['precision']):.4f} ± {np.std(metrics['precision']):.4f}")
    print(f"Recall:    {np.mean(metrics['recall']):.4f} ± {np.std(metrics['recall']):.4f}")
    print(f"F1-Score:  {np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
