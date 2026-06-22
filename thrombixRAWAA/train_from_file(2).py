import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding='utf-8')

# تحديد الجهاز المستخدم للتدريب (GPU أو CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ThrombiXLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ThrombiXLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        return self.fc(out) # 🔴 تم إزالة الـ Softmax لأن CrossEntropyLoss تحسبه تلقائياً

def load_data_from_csv(filename="thrombix_dataset.csv"):
    print(f"📖 جاري قراءة البيانات من {filename}...")
    df = pd.read_csv(filename)
    
    X_list, y_list = [], []
    grouped = df.groupby('Sample_ID')
    
    for sample_id, group in grouped:
        sorted_group = group.sort_values('Time_Step')
        features = sorted_group.iloc[:, 2:-1].values
        label = sorted_group['Label'].iloc[0]
        
        X_list.append(features)
        y_list.append(label)
    
    X_np = np.array(X_list)
    y_np = np.array(y_list)
    
    # 🔴 إضافة نقص: معايرة البيانات (Scaling)
    # نقوم بعمل تسطيح للمصفوفة لمعايرتها ثم إعادتها لأبعادها الأصلية
    nsamples, nx, ny = X_np.shape
    X_flat = X_np.reshape(-1, ny)
    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)
    X_np = X_flat_scaled.reshape(nsamples, nx, ny)
    
    return X_np, y_np

# حلقة التدريب والتقييم المحترفة
def train_model(model, train_loader, val_loader, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\n🚀 بدء تدريب النموذج...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # 🔴 إضافة نقص: تقييم النموذج على بيانات التحقق أثناء التدريب
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                
        val_loss /= len(val_loader.dataset)
        accuracy = (correct / len(val_loader.dataset)) * 100
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"🔄 الدورة [{epoch+1}/{epochs}] | خطأ التدريب: {train_loss:.4f} | خطأ التحقق: {val_loss:.4f} | دقة التحقق: {accuracy:.2f}%")
            
    print("✅ انتهى التدريب بنجاح!")
    return model

if __name__ == "__main__":
    try:
        X_data, y_data = load_data_from_csv("thrombix_dataset.csv")
    except FileNotFoundError:
        print("❌ خطأ: لم يتم العثور على ملف البيانات الحيوية.")
        exit()

    # 🔴 إضافة نقص: تقسيم البيانات إلى 80% تدريب و 20% اختبار
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # تحويل التوزيعات إلى Pytorch Tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    # 🔴 إضافة نقص: إنشاء الـ DataLoaders لمعالجة البيانات على دفعات (Batches)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    NUM_FEATURES = X_data.shape[2]
    model = ThrombiXLSTM(input_size=NUM_FEATURES, hidden_size=64, num_layers=2, num_classes=4).to(device)
    
    trained_model = train_model(model, train_loader, val_loader, epochs=60)

    torch.save(trained_model.state_dict(), 'thrombix_weights.pth')
    print("💾 تم حفظ أوزان النموذج الجاهز في ملف thrombix_weights.pth")