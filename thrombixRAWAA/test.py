import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

sys.stdout.reconfigure(encoding='utf-8')

# 1. تحديد الجهاز (GPU أو CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. تعريف نفس المعمارية المعتمدة
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
        out = out[:, -1, :] # أخذ الخطوة الزمنية الأخيرة
        return self.fc(out)

# 3. دالة لتوليد بيانات مرضى جُدد تماماً لمحاكاة الواقع الطبي
def generate_unseen_mock_data(num_patients=30, seq_length=50, num_features=10):
    print(f"🎲 جاري توليد بيانات محاكاة لـ {num_patients} مريض جديد لم يروهم النموذج من قبل...")
    
    X_list = []
    y_list = []
    
    for p_id in range(1000, 1000 + num_patients): # معرفات مرضى جديدة تماماً
        # اختيار فئة عشوائية للمريض من الفئات الـ 4 (0, 1, 2, 3)
        label = np.random.randint(0, 4)
        
        # توليد نمط أساسي يعتمد على الفئة مع إضافة "ضوضاء عشوائية" لجعل الاختبار حقيقياً وصعباً
        base_pattern = label * 1.5 
        noise = np.random.normal(0, 0.8, size=(seq_length, num_features))
        patient_features = base_pattern + noise
        
        X_list.append(patient_features)
        y_list.append(label)
        
    X_np = np.array(X_list)
    y_np = np.array(y_list)
    
    # تطبيق المعايرة (Scaling) بشكل معزول
    nsamples, nx, ny = X_np.shape
    X_flat = X_np.reshape(-1, ny)
    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)
    X_np = X_flat_scaled.reshape(nsamples, nx, ny)
    
    return torch.tensor(X_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.long)

# 4. السكربت التنفيذي للاختبار
if __name__ == "__main__":
    # أبعاد النموذج الثابتة
    NUM_FEATURES = 10
    NUM_CLASSES = 4
    
    # أ. توليد البيانات الجديدة
    X_test, y_test = generate_unseen_mock_data(num_patients=50) # اختبار على 50 مريضاً جديداً
    
    # ب. بناء هيكل النموذج ونقله للـ device
    model = ThrombiXLSTM(input_size=NUM_FEATURES, hidden_size=64, num_layers=2, num_classes=NUM_CLASSES).to(device)
    
    # جـ. تحميل الأوزان المحفوظة من السكربت الأول
    try:
        model.load_state_dict(torch.load('thrombix_weights.pth', map_location=device))
        print("💾 تم شحن أوزان النموذج المحفوظة `thrombix_weights.pth` بنجاح.")
    except FileNotFoundError:
        print("❌ خطأ: لم يتم العثور على ملف الأوزان 'thrombix_weights.pth'. يرجى تشغيل سكربت التدريب أولاً.")
        exit()
        
    # د. وضع النموذج في نمط التقييم (Evaluation Mode)
    model.eval()
    
    # هـ. البدء في التنبؤ بدون حساب التدرجات لتوفير الذاكرة
    print("\n🔍 جاري إدخال المرضى الجدد للنموذج واستخلاص التشخيصات...")
    with torch.no_grad():
        X_test = X_test.to(device)
        outputs = model(X_test)
        
        # استخراج الفئة صاحبة أعلى احتمالية
        _, predicted = torch.max(outputs, 1)
        
    # نقل النتائج إلى CPU لتحليلها بمكتبة scikit-learn
    y_true = y_test.numpy()
    y_pred = predicted.cpu().numpy()
    
    # و. طباعة النتائج الصارمة والتحكيم النهائي
    print("\n" + "="*50)
    print("📊 تقرير الأداء النهائي على البيانات الجديدة (Unseen Data)")
    print("="*50)
    
    # حساب الدقة الإجمالية
    accuracy = (y_pred == y_true).mean() * 100
    print(f"🎯 الدقة الإجمالية الحقيقية للنموذج: {accuracy:.2f}%\n")
    
    # تقرير تفصيلي لكل فئة مرضية
    print("📝 التقرير التفصيلي لكل فئة (Classification Report):")
    print(classification_report(y_true, y_pred, target_names=[f'Class {i}' for i in range(NUM_CLASSES)]))
    
    print("🧱 مصفوفة الارتباك (Confusion Matrix):")
    print(confusion_matrix(y_true, y_pred))
    print("="*50)