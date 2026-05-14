import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# ── 1. ดึงโครงสร้าง Autoencoder มาใช้ ──
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

def prepare_features(df):
    df = df.copy()
    # จัดการค่า NaN ก่อน (สมมติว่าผ่าน Stage 1 BiLSTM มาแล้ว หรือเติม 0 ชั่วคราวสำหรับการทดสอบ)
    df.fillna(0, inplace=True) 
    
    df['E_total_calc'] = df['E_grid'] + df['E_DG']
    df['EPR'] = df['Q_prod'] / (df['E_total_calc'] + 1e-5)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['day_of_week'] = df.index.dayofweek
    return df[['E_grid', 'E_DG', 'Q_prod', 'EPR', 'hour_sin', 'hour_cos', 'day_of_week']]

def main():
    # ค่า Threshold ที่ได้จากการเทรน
    TAU = 0.005857 

    print("Loading datasets...")
    # โหลด Clean Data เพื่อใช้ Fit Scaler (ต้องใช้มาตรวัดเดียวกับตอน Train)
    df_clean = pd.read_csv("clean_telemetry.csv", parse_dates=[0], index_col=0)
    features_clean = prepare_features(df_clean)
    
    scaler = MinMaxScaler()
    scaler.fit(features_clean.values)
    
    # โหลด Faulted Data (ข้อมูลที่โดนโกง) และ Labels (เฉลย)
    df_faulted = pd.read_csv("faulted_telemetry.csv", parse_dates=[0], index_col=0)
    features_faulted = prepare_features(df_faulted)
    data_scaled = scaler.transform(features_faulted.values)
    
    df_labels = pd.read_csv("anomaly_labels.csv", parse_dates=["timestamp"], index_col="timestamp")
    y_true = df_labels['anomaly_label'].values

    # โหลดโมเดล
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load("autoencoder_verifier.pth", weights_only=True))
    model.eval()

    print("Running Anomaly Detection...")
    x_tensor = torch.tensor(data_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        reconstructed = model(x_tensor)
        # คำนวณ Reconstruction Error (MSE) ของแต่ละบรรทัด
        errors = torch.mean((x_tensor - reconstructed)**2, dim=1).cpu().numpy()

    # ── การจัดหมวดหมู่ 3 ระดับ (Three-Tier Taxonomy) ──
    # ทายว่าโกง (1) ถ้า Error มากกว่า TAU
    y_pred = (errors > TAU).astype(int)
    
    type_1 = np.sum((errors > TAU) & (errors <= 2 * TAU))
    type_2 = np.sum((errors > 2 * TAU) & (errors <= 5 * TAU))
    type_3 = np.sum(errors > 5 * TAU)

    # ── คำนวณความแม่นยำ (Metrics) ──
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n" + "="*50)
    print("🎯 RESULTS: AUTOENCODER VERIFICATION ENGINE")
    print("="*50)
    print(f"Total Timesteps:     {len(y_true)}")
    print(f"Actual Anomalies:    {np.sum(y_true)} (จากเฉลย)")
    print(f"Detected Anomalies:  {np.sum(y_pred)} (ที่ AI จับได้)")
    print("-" * 50)
    print("Three-Tier Taxonomy Breakdown (จากที่ AI จับได้):")
    print(f"  [Type I] Marginal (TAU < RE <= 2TAU):     {type_1} ชั่วโมง")
    print(f"  [Type II] Structural (2TAU < RE <= 5TAU): {type_2} ชั่วโมง")
    print(f"  [Type III] Critical (RE > 5TAU):          {type_3} ชั่วโมง")
    print("-" * 50)
    print(f"Precision (ความแม่นยำเมื่อจับผิด): {precision:.4f}")
    print(f"Recall (ความสามารถในการกวาดจับ):   {recall:.4f}")
    print(f"F1-Score:                          {f1:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()