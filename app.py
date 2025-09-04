#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مساعد التعليم الذكي - Hugging Face Space
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import whisper
from gtts import gTTS
import tempfile
import os
from datetime import datetime

# إعداد النموذج
MODEL_NAME = "microsoft/DialoGPT-medium"
# HF_TOKEN = "YOUR_HUGGING_FACE_TOKEN_HERE"

# تحميل النموذج
print("🔄 تحميل النموذج...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# تحميل Whisper
print("🔄 تحميل Whisper...")
whisper_model = whisper.load_model("base")

print("✅ تم تحميل جميع النماذج!")

# دوال TTS و STT
def text_to_speech(text, language='ar'):
    """تحويل النص إلى كلام"""
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts.save(tmp_file.name)
            return tmp_file.name
    except Exception as e:
        print(f"❌ خطأ في TTS: {e}")
        return None

def speech_to_text(audio_file):
    """تحويل الكلام إلى نص"""
    try:
        result = whisper_model.transcribe(audio_file, language="ar")
        return result["text"].strip()
    except Exception as e:
        print(f"❌ خطأ في STT: {e}")
        return "لم أتمكن من فهم الكلام"

def chat_with_model(user_input):
    """التفاعل مع النموذج"""
    try:
        # إعداد prompt للتعليم
        prompt = f"أنت مساعد تعليمي ذكي. أجب على سؤال الطالب: {user_input}"
        
        # توليد الإجابة
        response = pipe(
            prompt,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # استخراج الإجابة
        generated_text = response[0]['generated_text']
        answer = generated_text.replace(prompt, "").strip()
        
        return answer
    except Exception as e:
        return f"عذراً، حدث خطأ: {str(e)}"

def voice_chat(audio_file):
    """التفاعل الصوتي"""
    if audio_file is None:
        return "من فضلك سجل صوتك", None
    
    # تحويل الصوت إلى نص
    user_text = speech_to_text(audio_file)
    
    # الحصول على إجابة
    response = chat_with_model(user_text)
    
    # تحويل الإجابة إلى كلام
    audio_response = text_to_speech(response)
    
    return f"🎤 سؤالك: {user_text}\n\n🤖 الإجابة: {response}", audio_response

# إنشاء واجهة Gradio
def create_interface():
    with gr.Blocks(
        title="مساعد التعليم الذكي",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        """
    ) as interface:
        
        gr.Markdown("""
        # 🎓 مساعد التعليم الذكي
        
        مساعد تعليمي ذكي يساعد الطلاب في التعلم والتفاعل عبر النص والصوت.
        
        **المميزات:**
        - 💬 محادثة نصية ذكية
        - 🎤 محادثة صوتية
        - 📚 شرح المفاهيم التعليمية
        - 🤖 إجابات فورية
        """)
        
        with gr.Tabs():
            with gr.Tab("💬 محادثة نصية"):
                with gr.Row():
                    with gr.Column(scale=3):
                        text_input = gr.Textbox(
                            label="اكتب سؤالك هنا",
                            placeholder="مثال: ما هو الذكاء الاصطناعي؟",
                            lines=3
                        )
                        text_btn = gr.Button("إرسال", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        text_output = gr.Textbox(
                            label="الإجابة",
                            lines=10,
                            interactive=False
                        )
                
                text_btn.click(
                    chat_with_model,
                    inputs=text_input,
                    outputs=text_output
                )
            
            with gr.Tab("🎤 محادثة صوتية"):
                with gr.Row():
                    with gr.Column(scale=2):
                        audio_input = gr.Audio(
                            label="سجل صوتك",
                            type="filepath",
                            format="mp3"
                        )
                        voice_btn = gr.Button("معالجة الصوت", variant="primary", size="lg")
                    
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
                    voice_chat,
                    inputs=audio_input,
                    outputs=[voice_output, audio_output]
                )
        
        gr.Markdown("""
        ---
        **💡 نصائح:**
        - استخدم أسئلة واضحة ومحددة
        - يمكنك السؤال عن أي موضوع تعليمي
        - الواجهة تدعم اللغة العربية والإنجليزية
        """)
    
    return interface

# تشغيل الواجهة
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # في Hugging Face Spaces
    )
