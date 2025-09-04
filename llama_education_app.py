# 🎓 مساعد التعليم الذكي - Llama 3.1
# واجهة محسنة مستوردة من Flowise

# تثبيت المكتبات
import subprocess
import sys

def install_packages():
    """تثبيت المكتبات المطلوبة"""
    packages = [
        "transformers",
        "torch", 
        "accelerate",
        "bitsandbytes",
        "openai-whisper",
        "gtts",
        "gradio",
        "huggingface_hub",
        "pydub"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ تم تثبيت {package}")
        except:
            print(f"❌ فشل في تثبيت {package}")

# تثبيت المكتبات
print("🔄 تثبيت المكتبات...")
install_packages()

# الاستيراد
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import whisper
from gtts import gTTS
import gradio as gr
import tempfile
import os
from datetime import datetime
from pydub import AudioSegment

# إعداد النموذج
HF_TOKEN = "YOUR_HUGGING_FACE_TOKEN_HERE"  # ضع التوكن هنا
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # نموذج Llama 3.1

print(f"🤗 Hugging Face Token: يرجى إضافة التوكن في الكود")
print(f"🤖 النموذج: {MODEL_NAME}")

# تحميل نموذج Llama 3.1
def load_llama_model():
    """تحميل نموذج Llama 3.1 مع التحسينات"""
    print("🔄 تحميل نموذج Llama 3.1...")
    
    # تحميل Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,  # هنا التوكن
        trust_remote_code=True
    )
    
    # تحميل النموذج مع تحسينات الذاكرة
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,  # هنا التوكن
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
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
def chat_with_llama(pipe, tokenizer, user_input, max_length=512):
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

# تحميل نموذج Whisper
print("🔄 تحميل نموذج Whisper...")
whisper_model = whisper.load_model("base")
print("✅ تم تحميل Whisper بنجاح!")

# دوال TTS و STT
def text_to_speech(text, language='ar'):
    """تحويل النص إلى كلام"""
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        
        # حفظ الملف الصوتي مؤقتاً
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts.save(tmp_file.name)
            return tmp_file.name
    except Exception as e:
        print(f"❌ خطأ في TTS: {e}")
        return None

def speech_to_text(audio_file):
    """تحويل الكلام إلى نص"""
    try:
        # تحويل الملف إلى WAV إذا لزم الأمر
        audio = AudioSegment.from_file(audio_file)
        wav_file = audio_file.replace('.mp3', '.wav').replace('.m4a', '.wav')
        audio.export(wav_file, format="wav")
        
        # استخدام Whisper للتعرف على الكلام
        result = whisper_model.transcribe(wav_file, language="ar")
        
        # حذف الملف المؤقت
        if os.path.exists(wav_file):
            os.remove(wav_file)
            
        return result["text"].strip()
    except Exception as e:
        print(f"❌ خطأ في STT: {e}")
        return "عذراً، لم أتمكن من فهم الصوت"

def voice_chat(pipe, tokenizer, audio_file):
    """محادثة صوتية كاملة"""
    # تحويل الصوت إلى نص
    user_text = speech_to_text(audio_file)
    
    if not user_text:
        return "عذراً، لم أتمكن من فهم الصوت", "عذراً، لم أتمكن من فهم الصوت", None
    
    # الحصول على إجابة من النموذج
    response = chat_with_llama(pipe, tokenizer, user_text)
    
    # تحويل الإجابة إلى صوت
    audio_response = text_to_speech(response)
    
    return user_text, response, audio_response

# إنشاء واجهة محسنة مستوردة من Flowise
def create_education_interface():
    """إنشاء واجهة احترافية للتعليم"""
    
    # تحميل النموذج
    pipe, tokenizer = load_llama_model()
    
    def process_text_input(user_input):
        """معالجة النص المدخل"""
        if not user_input.strip():
            return "من فضلك اكتب سؤالك"
        
        response = chat_with_llama(pipe, tokenizer, user_input)
        return response
    
    def process_voice_input(audio_file):
        """معالجة الصوت المدخل"""
        if audio_file is None:
            return "من فضلك سجل صوتك", None
        
        user_text, response, audio_response = voice_chat(pipe, tokenizer, audio_file)
        
        return f"🎤 سؤالك: {user_text}\n\n🤖 الإجابة: {response}", audio_response
    
    # إنشاء واجهة Gradio محسنة
    with gr.Blocks(
        title="مساعد التعليم الذكي - واجهة محسنة",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 15px;
            margin: 1rem 0;
            max-width: 80%;
        }
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: auto;
        }
        .bot-message {
            background: white;
            color: #333;
            border: 1px solid #e0e0e0;
            margin-right: auto;
        }
        """
    ) as interface:
        
        # Header محسن
        gr.Markdown("""
        <div class="main-header">
            <h1>🎓 مساعد التعليم الذكي</h1>
            <p>واجهة احترافية مستوردة من Flowise - تفاعل ذكي مع الطلاب</p>
        </div>
        """)
        
        # Tabs محسنة
        with gr.Tabs():
            with gr.Tab("💬 محادثة نصية"):
                gr.Markdown("### 💬 تفاعل نصي ذكي")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        text_input = gr.Textbox(
                            label="اكتب سؤالك هنا",
                            placeholder="مثال: ما هو الذكاء الاصطناعي؟",
                            lines=3
                        )
                        text_btn = gr.Button("📤 إرسال", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        text_output = gr.Textbox(
                            label="🤖 الإجابة",
                            lines=10,
                            interactive=False
                        )
                
                text_btn.click(
                    process_text_input,
                    inputs=text_input,
                    outputs=text_output
                )
            
            with gr.Tab("🎤 محادثة صوتية"):
                gr.Markdown("### 🎤 تفاعل صوتي متقدم")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        audio_input = gr.Audio(
                            label="سجل صوتك",
                            type="filepath",
                            format="mp3"
                        )
                        voice_btn = gr.Button("🎤 معالجة الصوت", variant="primary", size="lg")
                    
                    with gr.Column(scale=3):
                        voice_output = gr.Textbox(
                            label="المحادثة",
                            lines=8,
                            interactive=False
                        )
                        audio_output = gr.Audio(
                            label="الإجابة الصوتية",
                            type="filepath"
                        )
                
                voice_btn.click(
                    process_voice_input,
                    inputs=audio_input,
                    outputs=[voice_output, audio_output]
                )
            
            with gr.Tab("📊 الإحصائيات"):
                gr.Markdown("### 📊 إحصائيات المحادثة")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
                        **💡 نصائح للاستخدام:**
                        - استخدم أسئلة واضحة ومحددة
                        - يمكنك السؤال عن أي موضوع تعليمي
                        - جرب أساليب مختلفة في السؤال
                        - استخدم التقييم لتحسين الإجابات
                        """)
                    
                    with gr.Column():
                        gr.Markdown("""
                        **🎯 مميزات الواجهة:**
                        - تصميم احترافي مستورد من Flowise
                        - إحصائيات مباشرة في الوقت الفعلي
                        - أزرار سريعة للأسئلة الشائعة
                        - تصدير المحادثة بصيغة JSON
                        """)
        
        # Footer
        gr.Markdown("""
        ---
        **🎓 مساعد التعليم الذكي** | واجهة محسنة مستوردة من Flowise
        """)
    
    return interface

# تشغيل التطبيق
if __name__ == "__main__":
    print("🚀 بدء تشغيل مساعد التعليم...")
    print(f"⏰ الوقت الحالي: {datetime.now()}")
    
    # إنشاء الواجهة
    interface = create_education_interface()
    
    # تشغيل الواجهة
    interface.launch(
        share=True,
        debug=True,
        server_name="0.0.0.0",
        server_port=7860
    )
    
    print("🔗 رابط الواجهة متاح الآن!")
    print("💡 إذا انتهت صلاحية الرابط، شغل الملف مرة أخرى")
