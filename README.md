# 🎓 مساعد التعليم الذكي - Llama 3.1

مساعد تعليمي ذكي مدعوم بنموذج Llama 3.1 مع واجهة احترافية للتعلم التفاعلي.

## 🌟 المميزات

- **🤖 نموذج Llama 3.1**: أحدث نموذج ذكاء اصطناعي للتعليم
- **🌐 API منفصل**: تشغيل النموذج في Colab والواجهة محلياً
- **🎨 واجهة جميلة**: واجهة احترافية مع Streamlit
- **📊 إحصائيات**: تتبع المحادثات والإحصائيات
- **💾 تصدير**: حفظ المحادثات بصيغة JSON
- **⚡ سريع**: استجابة سريعة مع تحسين الذاكرة

## 🏗️ الهيكل

```
├── llama_api_server.py     # خادم API للنموذج (Colab)
├── local_app.py           # الواجهة المحلية (Streamlit)
├── colab_notebook.ipynb   # النسخة الأصلية (notebook)
├── requirements.txt       # متطلبات المشروع
└── README.md             # هذا الملف
```

## 🚀 طريقة التشغيل

### الخطوة 1: تشغيل API في Google Colab

1. **افتح Google Colab**
2. **انسخ كود `llama_api_server.py`** في خلية جديدة
3. **أضف Hugging Face Token:**
   ```python
   HF_TOKEN = "hf_your_token_here"
   ```
4. **شغل الخلية** وانتظر التحميل
5. **انسخ رابط ngrok** الذي سيظهر

### الخطوة 2: تشغيل الواجهة المحلية

1. **ثبت المتطلبات:**
   ```bash
   pip install streamlit requests plotly pandas
   ```

2. **شغل التطبيق:**
   ```bash
   streamlit run local_app.py
   ```

3. **أدخل رابط API** في الشريط الجانبي

4. **استمتع بالمحادثة!** 🎉

## 📋 متطلبات النظام

### للـ API (Google Colab):
- **GPU**: T4 أو أفضل
- **ذاكرة**: 15GB+ RAM
- **Python**: 3.7+
- **مكتبات**: transformers, torch, flask, pyngrok

### للواجهة المحلية:
- **Python**: 3.7+
- **مكتبات**: streamlit, requests, plotly, pandas

## 🔧 الإعداد المتقدم

### تخصيص النموذج:

```python
# في llama_api_server.py
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # نموذج Llama
# أو
MODEL_NAME = "microsoft/DialoGPT-medium"  # نموذج أصغر
```

### تخصيص الإعدادات:

```python
# في دالة chat_with_llama
max_new_tokens=512,    # طول الإجابة
temperature=0.7,       # الإبداع
do_sample=True         # التنويع
```

## 🛠️ استكشاف الأخطاء

### مشاكل شائعة:

1. **خطأ الذاكرة:**
   ```python
   load_in_8bit=True  # ضغط النموذج
   ```

2. **خطأ التوكن:**
   ```python
   HF_TOKEN = "your_valid_token"  # تأكد من التوكن
   ```

3. **انقطاع ngrok:**
   ```python
   # أعد تشغيل الخلية
   public_url = ngrok.connect(5000)
   ```

## 📊 API Reference

### GET `/`
الصفحة الرئيسية - معلومات API

**Response:**
```json
{
  "status": "success",
  "message": "Llama 3.1 API is running!",
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "timestamp": "2024-09-04T12:00:00"
}
```

### POST `/chat`
إرسال رسالة للنموذج

**Request:**
```json
{
  "message": "ما هو الذكاء الاصطناعي؟"
}
```

**Response:**
```json
{
  "success": true,
  "response": "الذكاء الاصطناعي هو...",
  "user_input": "ما هو الذكاء الاصطناعي؟",
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "timestamp": "2024-09-04T12:00:00",
  "status": "success"
}
```

### GET `/health`
فحص حالة API

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-09-04T12:00:00"
}
```

## 🎯 أمثلة الاستخدام

### باستخدام curl:
```bash
curl -X POST https://xxxxx.ngrok.io/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "اشرح لي الفيزياء النووية"}'
```

### باستخدام Python:
```python
import requests

response = requests.post(
    "https://xxxxx.ngrok.io/chat",
    json={"message": "كيف أتعلم البرمجة؟"}
)

print(response.json()["response"])
```

### باستخدام JavaScript:
```javascript
fetch('https://xxxxx.ngrok.io/chat', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        message: 'ما هي فوائد الرياضيات؟'
    })
})
.then(response => response.json())
.then(data => console.log(data.response));
```

## 🔒 الأمان

- **لا تشارك رابط ngrok** مع أشخاص غير مخولين
- **استخدم HTTPS** فقط
- **لا تضع التوكن** في الكود المرفوع
- **استخدم متغيرات البيئة** للإنتاج

## 🚧 التطوير

### إضافة ميزات جديدة:

1. **TTS/STT**: إضافة تحويل الصوت
2. **ذاكرة المحادثة**: حفظ السياق
3. **تخصيص البرومبت**: برومبت ديناميكي
4. **قاعدة البيانات**: حفظ المحادثات

### المساهمة:

1. Fork المشروع
2. إنشاء branch جديد
3. التطوير والاختبار
4. إرسال Pull Request

## 📄 الترخيص

هذا المشروع مرخص تحت رخصة MIT.

## 🆘 الدعم

للحصول على المساعدة:
- **Issues**: افتح issue جديد
- **Discussions**: شارك في النقاشات
- **Email**: راسلنا للدعم

## 🎉 شكر خاص

- **Meta**: لنموذج Llama 3.1
- **Hugging Face**: للمكتبات والاستضافة
- **Google**: لـ Colab المجاني
- **Streamlit**: للواجهة الجميلة

---

**مطور بحب ❤️ للتعليم الذكي**
