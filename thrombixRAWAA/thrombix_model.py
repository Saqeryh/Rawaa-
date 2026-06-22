import torch
import torch.nn as nn
import torch.nn.functional as F

class ThrombiXLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        """
        تهيئة نموذج ThrombiX للتنبؤ بحالات التجلط الدموي.
        
        المتغيرات:
        - input_size: عدد المستشعرات/الميزات (Features) في كل خطوة زمنية.
        - hidden_size: عدد الخلايا العصبية في طبقة الـ LSTM لاستخلاص الأنماط.
        - num_layers: عدد طبقات الـ LSTM (يفضل 1-2 لتجنب Overfitting في الأجهزة القابلة للارتداء).
        - num_classes: عدد الحالات التشخيصية (4 حالات).
        """
        super(ThrombiXLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # طبقة المعالجة الزمنية (LSTM Layer)
        # batch_first=True ضرورية ليكون شكل الإدخال: (Batch Size, Sequence Length, Features)
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True,
                            dropout=0.2 if num_layers > 1 else 0) # Dropout لمنع الحفظ الأعمى للبيانات
        
        # طبقة التصنيف (Dense / Linear Layer)
        # تأخذ الفهم الزمني العميق من آخر حالة للـ LSTM وتطابقه مع الحالات الأربع
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        مسار تمرير البيانات (Forward Pass).
        """
        # تهيئة الحالة المخفية (Hidden State) وحالة الخلية (Cell State) بأصفار
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # تمرير البيانات عبر الـ LSTM
        # المخرج 'out' يحتوي على جميع المخرجات الزمنية، ونحن نحتاج النتيجة النهائية فقط
        out, _ = self.lstm(x, (h0, c0))
        
        # استخلاص مخرجات آخر خطوة زمنية (Last Time-step)
        out = out[:, -1, :]
        
        # تمرير النتيجة إلى طبقة التصنيف (Dense Layer) للحصول على الـ Logits
        logits = self.fc(out)
        
        # تحويل المخرجات إلى نسب مئوية (Probabilities) عبر دالة Softmax
        # dim=1 يعني تطبيقها عبر الحالات الأربع لكل مريض في الـ Batch
        probabilities = F.softmax(logits, dim=1)
        
        return probabilities

        # القائمة المعتمدة لتصنيفات المشروع
CLASSES = [
    "طبيعي (Normal)", 
    "نشاط متزايد (Rising Activity)", 
    "ما قبل التجلط (Pre-thrombotic state)", 
    "خطر حرج (Critical Risk)"
]

def generate_dummy_sensor_data(batch_size, seq_length, num_features):
    """
    محاكاة قراءات المستشعرات:
    (D-dimer, PPG, IMU, Thermistor, P-selectin, TAT)
    تُرجع مصفوفة Tensor عشوائية بالأبعاد المطلوبة لاختبار النموذج.
    """
    # الأبعاد: (Batch Size, Sequence Length, Number of Features)
    dummy_data = torch.rand(batch_size, seq_length, num_features)
    print(f"تم توليد بيانات وهمية بأبعاد: {dummy_data.shape}")
    return dummy_data

def run_thrombix_inference(model, input_data):
    """
    دالة الاستنتاج الفعلي (Inference).
    تستقبل النموذج والقراءات الجديدة، وتطبع التشخيص والنسب المئوية.
    """
    # تفعيل وضع الاستنتاج (Evaluation Mode) لتعطيل خواص التدريب مثل Dropout
    model.eval()
    
    # تعطيل حساب التدرجات (Gradients) لتسريع الأداء وتوفير الذاكرة
    with torch.no_grad():
        # تمرير البيانات للنموذج للحصول على الاحتمالات
        predictions = model(input_data)
    
    # استخراج أول عينة في الـ Batch لغرض العرض
    patient_probs = predictions[0]
    
    # اختيار أعلى نسبة مئوية (التصنيف الفائز)
    predicted_idx = torch.argmax(patient_probs).item()
    predicted_class = CLASSES[predicted_idx]
    
    # طباعة التقرير النهائي
    print("\n" + "="*40)
    print(" 🩸 تقرير نظام رَواء (ThrombiX) الذكي 🩸")
    print("="*40)
    print("النسب المئوية للتشخيص:")
    
    for i, class_name in enumerate(CLASSES):
        # ضرب الاحتمال بـ 100 للحصول على نسبة مئوية مفهومة
        percentage = patient_probs[i].item() * 100
        print(f"  - {class_name}: {percentage:.2f}%")
        
    print("-" * 40)
    print(f"📌 التشخيص النهائي: {predicted_class}")
    
    # نظام إنذار مبكر برمجي
    if predicted_idx >= 2: # إذا كانت الحالة "ما قبل التجلط" أو "خطر حرج"
        print("⚠️ تنبيه نظام: تم رصد مؤشرات حيوية تستدعي تدخلاً طبياً فورياً!")
    print("="*40)


if __name__ == "__main__":
    # لاحظي المسافة هنا قبل كل سطر (4 مسافات أو ضغطة Tab)
    # 1. إعداد المعطيات (Hyperparameters)
    NUM_FEATURES = 10     # إجمالي القراءات (مثلاً: 3 لـ IMU، 2 لـ PPG، 1 للحرارة، 4 لباقي المؤشرات)
    SEQ_LENGTH = 50       # طول النافذة الزمنية (مثلاً 50 قراءة متتالية لتكوين نمط)
    BATCH_SIZE = 1        # قراءة المريض الحالي
    HIDDEN_SIZE = 64      # حجم الطبقة المخفية
    NUM_LAYERS = 2        # عدد طبقات الـ LSTM المكدسة
    NUM_CLASSES = 4       # عدد الحالات التشخيصية

    # 2. بناء النموذج
    thrombix_model = ThrombiXLSTM(input_size=NUM_FEATURES, 
                                  hidden_size=HIDDEN_SIZE, 
                                  num_layers=NUM_LAYERS, 
                                  num_classes=NUM_CLASSES)

    # 3. توليد بيانات جهاز المريض (محاكاة)
    live_sensor_readings = generate_dummy_sensor_data(BATCH_SIZE, SEQ_LENGTH, NUM_FEATURES)

    # 4. تشغيل نظام التنبؤ
    run_thrombix_inference(thrombix_model, live_sensor_readings)
    import torch
import torch.nn as nn
import torch.optim as optim

def generate_smart_training_data(num_samples, seq_length, num_features):
    """
    توليد بيانات تدريب 'ذكية' تحتوي على أنماط تميز كل حالة تشخيصية.
    """
    X_data = []
    y_labels = []
    
    for _ in range(num_samples):
        # اختيار حالة تشخيصية عشوائية لهذا المريض (0 إلى 3)
        label = torch.randint(0, 4, (1,)).item()
        
        # الأساس: مصفوفة عشوائية بقيم صغيرة
        base_signal = torch.rand(seq_length, num_features) * 0.3
        
        # تعديل البيانات بناءً على الحالة التشخيصية لتكوين "أنماط"
        if label == 0: # طبيعي
            # لا تغيير، قراءات مستقرة ومنخفضة
            signal = base_signal
        
        elif label == 1: # نشاط متزايد
            # إضافة تصاعد تدريجي (Trend) بمرور الوقت
            trend = torch.linspace(0, 0.5, seq_length).unsqueeze(1).repeat(1, num_features)
            signal = base_signal + trend
            
        elif label == 2: # ما قبل التجلط
            # قيم أعلى بشكل عام مع بعض التذبذبات
            signal = base_signal + 0.5 + (torch.randn(seq_length, num_features) * 0.1)
            
        elif label == 3: # خطر حرج
            # قفزات عالية جداً في القراءات تحاكي الـ D-dimer والـ TAT
            signal = base_signal + 0.8
            
        X_data.append(signal)
        y_labels.append(label)
        
    # تجميع القوائم في Tensors نهائية
    X_tensor = torch.stack(X_data)
    y_tensor = torch.tensor(y_labels, dtype=torch.long)
    
    return X_tensor, y_tensor

def train_thrombix_model(model, X_train, y_train, epochs=50, learning_rate=0.001):
    """
    حلقة التدريب (Training Loop) لتعليم نموذج رَواء.
    """
    print("\n🚀 جاري بدء عملية تدريب نموذج ThrombiX...\n")
    
    # 1. تعريف دالة الخسارة والمُحسّن
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # تفعيل وضع التدريب
    model.train()
    
    for epoch in range(epochs):
        # أ. تصفير التدرجات (Gradients) من الدورة السابقة
        optimizer.zero_grad()
        
        # ب. التمرير الأمامي (Forward Pass) - استخراج التوقعات
        # ملاحظة: النموذج يخرج احتمالات (Softmax)، لكن CrossEntropyLoss في PyTorch
        # تفضل استقبال الـ Logits الخام. لأغراض هذا النموذج البسيط، سنمررها كما هي.
        outputs = model(X_train)
        
        # ج. حساب مقدار الخطأ (Loss)
        loss = criterion(outputs, y_train)
        
        # د. التمرير العكسي (Backward Pass) - حساب التدرجات
        loss.backward()
        
        # هـ. تحديث الأوزان (Weights Update)
        optimizer.step()
        
        # طباعة تقرير التقدم كل 10 دورات
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"🔄 دورة التدريب (Epoch) [{epoch+1}/{epochs}] | نسبة الخطأ (Loss): {loss.item():.4f}")
            
    print("\n✅ اكتمل تدريب النموذج بنجاح! الآلة الآن قادرة على التفريق بين الأنماط.")
    return model

# ==========================================
# كتلة التشغيل (التدريب ثم الاستنتاج)
# ==========================================
if __name__ == "__main__":
    # إعدادات المعمارية
    NUM_FEATURES = 10
    SEQ_LENGTH = 50
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    NUM_CLASSES = 4
    
    # إعدادات التدريب
    NUM_SAMPLES = 200 # عدد المرضى الوهميين في قاعدة البيانات
    EPOCHS = 100      # عدد دورات التعلم

    # 1. بناء المعمارية
    # (تأكد من وجود كلاس ThrombiXLSTM معرفاً في الأعلى)
    thrombix_model = ThrombiXLSTM(input_size=NUM_FEATURES, 
                                  hidden_size=HIDDEN_SIZE, 
                                  num_layers=NUM_LAYERS, 
                                  num_classes=NUM_CLASSES)

    # 2. توليد بيانات التدريب الذكية
    print(f"📊 جاري توليد {NUM_SAMPLES} سجل طبي وهمي للتدريب...")
    X_train, y_train = generate_smart_training_data(NUM_SAMPLES, SEQ_LENGTH, NUM_FEATURES)

    # 3. إطلاق حلقة التدريب
    trained_model = train_thrombix_model(thrombix_model, X_train, y_train, epochs=EPOCHS)

    # 4. اختبار النموذج المدرب على قراءة جديدة (مريض في حالة خطر حرج مثلاً)
    print("\n🧪 اختبار النموذج المدرب على بيانات 'خطر حرج' جديدة:")
    # نولد قراءة لمريض واحد، ونرفع القيم يدوياً لمحاكاة الخطر الحرج
    test_sample = (torch.rand(1, SEQ_LENGTH, NUM_FEATURES) * 0.3) + 0.8 
    
    # (تأكد من وجود دالة run_thrombix_inference معرفة في الأعلى)
    run_thrombix_inference(trained_model, test_sample)