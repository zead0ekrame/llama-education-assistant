# 🚀 Llama 3.1 API Server للنسخ في Colab
# انسخ هذا الكود في Google Colab

# تثبيت المكتبات المطلوبة
"""
%pip install transformers torch accelerate bitsandbytes
%pip install flask flask-cors pyngrok
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
import threading
import json
from datetime import datetime
import time

# إعداد Hugging Face Token
HF_TOKEN = "YOUR_HUGGING_FACE_TOKEN_HERE"  # ضع التوكن هنا
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

print(f"🤗 Hugging Face Token: يرجى إضافة التوكن في الكود")
print(f"🤖 النموذج: {MODEL_NAME}")

# تحميل نموذج Llama 3.1
def load_llama_model():
    """تحميل نموذج Llama 3.1 مع التحسينات"""
    print("🔄 تحميل نموذج Llama 3.1...")
    
    # تحميل Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        trust_remote_code=True
    )
    
    # تحميل النموذج مع تحسينات الذاكرة
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        load_in_8bit=True  # ضغط النموذج لتوفير الذاكرة
    )
    
    # إنشاء pipeline للتفاعل
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("✅ تم تحميل النموذج بنجاح!")
    return pipe, tokenizer

# دالة للتفاعل مع النموذج
def chat_with_llama(user_input, max_length=512):
    """التفاعل مع نموذج Llama 3.1"""
    
    # إعداد prompt للتعليم
    system_prompt = """أنت مساعد تعليمي ذكي يساعد الطلاب في التعلم. 
    أجب على أسئلة الطلاب بطريقة واضحة ومفيدة. 
    استخدم أمثلة عملية واشرح المفاهيم بطريقة مبسطة."""
    
    # استخدام تنسيق Llama 3.1 الصحيح
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    try:
        # تحويل الرسائل إلى تنسيق النموذج
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # توليد الإجابة
        inputs = tokenizer(prompt, return_tensors="pt").to(pipe.device)
        
        with torch.no_grad():
            outputs = pipe.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # استخراج الإجابة
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        return response.strip()
    except Exception as e:
        return f"عذراً، حدث خطأ: {str(e)}"

# تحميل النموذج
print("🔄 بدء تحميل النموذج...")
pipe, tokenizer = load_llama_model()

# إنشاء Flask API
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'success',
        'message': 'Llama 3.1 API is running!',
        'model': MODEL_NAME,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # الحصول على البيانات من الطلب
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'لم يتم إرسال رسالة',
                'status': 'error'
            }), 400
        
        user_input = data['message'].strip()
        if not user_input:
            return jsonify({
                'success': False,
                'error': 'الرسالة فارغة',
                'status': 'error'
            }), 400
        
        # الحصول على الإجابة من النموذج
        response = chat_with_llama(user_input)
        
        return jsonify({
            'success': True,
            'response': response,
            'user_input': user_input,
            'model': MODEL_NAME,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'timestamp': datetime.now().isoformat()
    })

# تشغيل API مع ngrok
def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False)

# تشغيل Flask في thread منفصل
flask_thread = threading.Thread(target=run_flask)
flask_thread.daemon = True
flask_thread.start()

# انتظار قليل حتى يبدأ الخادم
time.sleep(3)

# إنشاء ngrok tunnel
try:
    public_url = ngrok.connect(5000)
    print("🎉 تم تشغيل API بنجاح!")
    print(f"🌐 الرابط العام: {public_url.public_url}")
    print(f"🔗 استخدم هذا الرابط في التطبيق المحلي")
    print("\n📋 طرق الاستخدام:")
    print(f"GET  {public_url.public_url}/        - الصفحة الرئيسية")
    print(f"POST {public_url.public_url}/chat    - إرسال رسالة")
    print(f"GET  {public_url.public_url}/health  - فحص الحالة")
    print("\n💡 مثال على الاستخدام:")
    print(f"curl -X POST {public_url.public_url}/chat \\")
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"message": "ما هو الذكاء الاصطناعي?"}\'\n')
    
    # حفظ الرابط في ملف
    with open('/content/api_url.txt', 'w') as f:
        f.write(public_url.public_url)
    print("💾 تم حفظ الرابط في api_url.txt")
    
except Exception as e:
    print(f"❌ خطأ في تشغيل ngrok: {e}")

# للحفاظ على تشغيل API
print("\n🔄 للحفاظ على تشغيل API:")
print("شغل الكود التالي إذا توقف ngrok:")
print("""
try:
    ngrok.disconnect(public_url.public_url)
except:
    pass

public_url = ngrok.connect(5000)
print(f"🔄 تم إعادة تشغيل ngrok: {public_url.public_url}")
""")
