import pandas as pd
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

def create_and_save_dataset(filename="thrombix_dataset.csv", num_samples=500, seq_length=50, num_features=10):
    print(f"📊 جاري توليد {num_samples} سجل طبي ذكي...")
    
    all_rows = []
    
    for sample_id in range(num_samples):
        # اختيار حالة عشوائية لكل مريض (0: طبيعي، 1: نشاط متزايد، 2: ما قبل التجلط، 3: خطر حرج)
        label = np.random.randint(0, 4)
        
        # القيمة الأساسية (إشارات منخفضة ومستقرة)
        base_signal = np.random.rand(seq_length, num_features) * 0.3
        
        # تطبيق الأنماط الطبية بناءً على الحالة ليكون التدريب منطقياً
        if label == 1:    # نشاط متزايد
            trend = np.linspace(0, 0.5, seq_length).reshape(-1, 1)
            base_signal += trend
        elif label == 2:  # ما قبل التجلط
            base_signal += 0.5 + (np.random.randn(seq_length, num_features) * 0.1)
        elif label == 3:  # خطر حرج
            base_signal += 0.8
            
        # تحويل المصفوفة ثلاثية الأبعاد إلى أسطر ثنائية الأبعاد لحفظها في CSV
        for t in range(seq_length):
            # السطر يحتوي على: رقم المريض، الخطوة الزمنية، القراءات الـ 10، والتشخيص النهائي
            row = [sample_id, t] + list(base_signal[t]) + [label]
            all_rows.append(row)
            
    # أسماء الأعمدة لتكون واضحة ومطابقة للهاردوير والمؤشرات الحيوية
    columns = [
        'Sample_ID', 'Time_Step', 
        'D_dimer', 'P_selectin', 'TAT',          # المؤشرات الحيوية
        'PPG_Red', 'PPG_IR',                     # مستشعر MAX30102
        'Accel_X', 'Accel_Y', 'Accel_Z',         # مستشعر MPU-6050
        'Temperature', 'Immobility_Timer',       # الحرارة والعداد البرمجي
        'Label'                                  # التشخيص النهائي
    ]
    
    # تحويل البيانات إلى dataframe وحفظها
    df = pd.DataFrame(all_rows, columns=columns)
    df.to_csv(filename, index=False)
    print(f"💾 تم حفظ البيانات بنجاح في الملف: {filename} (يحتوي على {len(df)} سطر)")

if __name__ == "__main__":
    # لتشغيل السكربت وتوليد الملف، ستحتاجين لمكتبة pandas و numpy
    # pip install pandas numpy
    create_and_save_dataset()