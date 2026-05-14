import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# ── 1. กำหนด Architecture ของ Autoencoder ──
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder: 7 -> 32 -> 16 -> 8
        self.encoder = nn.Sequential(
            nn.Linear(7, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8) # Latent space (Bottleneck)
        )
        # Decoder: 8 -> 16 -> 32 -> 7
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 7),
            nn.Sigmoid() # คืนค่าให้อยู่ในช่วง 0-1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ── 2. สร้าง Dataset แบบ Point-based (ไม่ใช้ Sequence) ──
class PointDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

# ── 3. ฟังก์ชันหลัก ──
def main():
    print("Loading clean_telemetry.csv for Autoencoder...")
    df = pd.read_csv("clean_telemetry.csv", parse_dates=[0], index_col=0)
    
    # Feature Engineering (สร้างฟีเจอร์ตามเปเปอร์)
    df['E_total_calc'] = df['E_grid'] + df['E_DG']
    # ป้องกันการหารด้วยศูนย์ด้วยการบวกค่าเล็กๆ (1e-5)
    df['EPR'] = df['Q_prod'] / (df['E_total_calc'] + 1e-5) 
    
    # แปลงเวลาให้เป็น Cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['day_of_week'] = df.index.dayofweek # 0=Monday, 6=Sunday
    
    features = ['E_grid', 'E_DG', 'Q_prod', 'EPR', 'hour_sin', 'hour_cos', 'day_of_week']
    data_to_scale = df[features].values
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_to_scale)
    
    # แบ่งข้อมูล Train 80%, Val 20%
    n = len(data_scaled)
    train_end = int(n * 0.8)
    
    train_data = data_scaled[:train_end]
    val_data = data_scaled[train_end:]
    
    train_dataset = PointDataset(train_data)
    val_dataset = PointDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    # Setup โมเดล
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4) # Learning rate ตามเปเปอร์
    
    epochs = 150
    
    print(f"Start Training Autoencoder on {device}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x in train_loader:
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x) # Autoencoder เทียบ Output กับ Input ของตัวเอง
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_x)
                val_loss += loss.item()
                
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {val_loss/len(val_loader):.6f}")
            
    print("Training Complete! Save model weights...")
    torch.save(model.state_dict(), "autoencoder_verifier.pth")
    
    # ── 4. คำนวณ Threshold สำหรับใช้จับผิด ──
    print("Calculating Anomaly Threshold (99th Percentile)...")
    model.eval()
    all_errors = []
    val_loader_thresh = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    with torch.no_grad():
        for batch_x in val_loader_thresh:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            # คำนวณ Error รายบรรทัด
            error = torch.mean((batch_x - outputs)**2, dim=1).cpu().numpy()
            all_errors.extend(error)
            
    threshold = np.percentile(all_errors, 99)
    print(f"*** CALIBRATED THRESHOLD (tau): {threshold:.6f} ***")
    print("(จดค่า Threshold นี้ไว้เพื่อใช้ในขั้นตอนทดสอบระบบ)")

if __name__ == "__main__":
    main()