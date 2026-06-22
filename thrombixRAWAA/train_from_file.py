import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')
# استدعاء نفس معمارية الـ LSTM التي اعتمدناها سابقاً
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
        return torch.softmax(self.fc(out), dim=1)

def load_data_from_csv(filename="thrombix_dataset.csv", seq_length=50):
    """
    قراءة ملف الـ CSV وتحويله إلى مصفوفات PyTorch بالهيكل الزمني الصحيح.
    """
    print(f"📖 جاري قراءة البيانات من {filename}...")
    df = pd.read_csv(filename)
    
    X_list = []
    y_list = []
    
    # تجميع البيانات بناءً على رقم المريض (Sample_ID) لإعادة بناء المصفوفة ثلاثية الأبعاد
    grouped = df.groupby('Sample_ID')
    
    for sample_id, group in grouped:
        # ترتيب الخطوات الزمنية لضمان التسلسل الصحيح
        sorted_group = group.sort_values('Time_Step')
        
        # استخراج أعمدة المستشعرات فقط (من العمود الثالث وحتى ما قبل الأخير)
        features = sorted_group.iloc[:, 2:-1].values
        
        # استخراج التشخيص (وهو ثابت لكل خطوات المريض الزمنية في هذا المثال)
        label = sorted_group['Label'].iloc[0]
        
        X_list.append(features)
        y_list.append(label)
        
    # تحويل القوائم إلى Tensors الخاصة بـ PyTorch
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_tensor = torch.tensor(y_list, dtype=torch.long)
    
    print(f"✅ تم تحميل وتحويل البيانات بنجاح!")
    print(f"   أبعاد بيانات الإدخال (X): {X_tensor.shape} -> (عدد المرضى, الخطوات الزمنية, المستشعرات)")
    print(f"   أبعاد التسميات (y): {y_tensor.shape}")
    
    return X_tensor, y_tensor

# حلقة التدريب المعتمدة
def train_model(model, X_train, y_train, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    print("\n🚀 بدء تدريب النموذج باستخدام بيانات الملف لحفظ الأنماط...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"🔄 الدورة [{epoch+1}/{epochs}] | نسبة الخطأ (Loss): {loss.item():.4f}")
            
    print("✅ انتهى التدريب بنجاح وحفظ النموذج الأنماط الحقيقية المستخلصة من الملف.")
    return model

if __name__ == "__main__":
    # 1. تحميل البيانات من الملف المكتوب
    try:
        X_train, y_train = load_data_from_csv("thrombix_dataset.csv")
    except FileNotFoundError:
        print("❌ خطأ: لم يتم العثور على ملف البيانات الحيوية. يرجى تشغيل سكربت التوليد أولاً.")
        exit()

    # 2. تهيئة وتدريب النموذج
    NUM_FEATURES = 10
    SEQ_LENGTH = 50
    model = ThrombiXLSTM(input_size=NUM_FEATURES, hidden_size=64, num_layers=2, num_classes=4)
    trained_model = train_model(model, X_train, y_train, epochs=60)

    # كود حفظ النموذج يضاف في نهاية السكربت
torch.save(trained_model.state_dict(), 'thrombix_weights.pth')
print("💾 تم حفظ أوزان النموذج الجاهز في ملف thrombix_weights.pth")