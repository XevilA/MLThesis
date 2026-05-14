import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

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

class TelemetryDataset(Dataset):
    def __init__(self, data, target, seq_len=168):
        self.data = data
        self.target = target
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.target[idx : idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading data & training BiLSTM on {device}...")
    
    df = pd.read_csv("clean_telemetry.csv", parse_dates=[0], index_col=0)
    
    features = ['E_grid', 'E_DG', 'Q_prod', 'T_amb', 'GHI']
    targets = ['E_grid', 'E_DG']
    
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x_scaled = scaler_x.fit_transform(df[features])
    y_scaled = scaler_y.fit_transform(df[targets])
    
    n = len(df)
    train_end, val_end = int(n * 0.8), int(n * 0.9)
    
    train_dataset = TelemetryDataset(x_scaled[:train_end], y_scaled[:train_end])
    val_dataset = TelemetryDataset(x_scaled[train_end:val_end], y_scaled[train_end:val_end])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = BiLSTMImputer().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 150
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # --- MASKED MODELING: แกล้งปิดข้อมูล 20% ให้เป็น 0 ---
            mask = (torch.rand(batch_x.shape[0], batch_x.shape[1], 1).to(device) > 0.20).float()
            masked_x = batch_x.clone()
            masked_x[:, :, 0:2] = masked_x[:, :, 0:2] * mask 
            
            optimizer.zero_grad()
            outputs = model(masked_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                mask = (torch.rand(batch_x.shape[0], batch_x.shape[1], 1).to(device) > 0.20).float()
                masked_x = batch_x.clone()
                masked_x[:, :, 0:2] = masked_x[:, :, 0:2] * mask
                outputs = model(masked_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.5f} | Val Loss: {val_loss/len(val_loader):.5f}")
        
    torch.save(model.state_dict(), "bilstm_imputer.pth")
    print("Saved Smart BiLSTM!")

if __name__ == "__main__":
    main()