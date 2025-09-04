#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مساعد التعليم الذكي - واجهة مستوردة من Flowise
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
import base64

# إعداد الصفحة
st.set_page_config(
    page_title="مساعد التعليم الذكي",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS مخصص للواجهة المستوردة
st.markdown("""
<style>
    /* Header Styles */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Chat Container */
    .chat-container {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        min-height: 500px;
        max-height: 600px;
        overflow-y: auto;
    }
    
    /* Message Styles */
    .message {
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 15px;
        max-width: 80%;
        word-wrap: break-word;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
        text-align: right;
    }
    
    .bot-message {
        background: white;
        color: #333;
        border: 1px solid #e0e0e0;
        margin-right: auto;
    }
    
    /* Input Area */
    .input-area {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin-top: 1rem;
    }
    
    /* Sidebar Styles */
    .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Stats Cards */
    .stat-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Loading Animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎓 مساعد التعليم الذكي</h1>
    <p>واجهة مستوردة من Flowise - تفاعل ذكي مع الطلاب</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ إعدادات Flowise")
    
    # Flowise Configuration
    flowise_url = st.text_input(
        "🔗 Flowise URL", 
        value="http://localhost:3000",
        help="رابط Flowise الخاص بك"
    )
    
    flow_id = st.text_input(
        "🆔 Flow ID", 
        value="your_flow_id_here",
        help="معرف Flow المستورد"
    )
    
    api_key = st.text_input(
        "🔑 API Key", 
        type="password",
        value="your_api_key_here",
        help="مفتاح API الخاص بك"
    )
    
    st.markdown("---")
    
    # Model Settings
    st.markdown("### 🤖 إعدادات النموذج")
    
    temperature = st.slider(
        "🌡️ Temperature", 
        0.0, 1.0, 0.7,
        help="مستوى الإبداع في الإجابات"
    )
    
    max_tokens = st.slider(
        "📝 Max Tokens", 
        100, 2000, 1000,
        help="الحد الأقصى للكلمات"
    )
    
    st.markdown("---")
    
    # Statistics
    st.markdown("### 📊 الإحصائيات")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    total_messages = len(st.session_state.messages)
    user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
    bot_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
    
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{total_messages}</div>
        <div class="stat-label">إجمالي الرسائل</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{user_messages}</div>
        <div class="stat-label">رسائل المستخدم</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{bot_messages}</div>
        <div class="stat-label">ردود البوت</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Actions
    st.markdown("---")
    st.markdown("### 🛠️ الإجراءات")
    
    if st.button("🗑️ مسح المحادثة", type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("📥 تصدير المحادثة"):
        export_chat_data(st.session_state.messages)

# Main Content
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 💬 المحادثة")
    
    # Chat Container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="message user-message">
                <strong>أنت:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="message bot-message">
                <strong>🤖 المساعد:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input Area
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    
    # Chat input
    user_input = st.text_input(
        "اكتب رسالتك هنا...",
        key="chat_input",
        placeholder="مثال: ما هو الذكاء الاصطناعي؟"
    )
    
    col_send, col_voice = st.columns([1, 1])
    
    with col_send:
        if st.button("📤 إرسال", type="primary"):
            if user_input:
                # Add user message
                st.session_state.messages.append({
                    "role": "user", 
                    "content": user_input,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Get bot response
                with st.spinner("🤔 جاري التفكير..."):
                    try:
                        response = call_flowise_api(
                            user_input, 
                            flowise_url, 
                            flow_id, 
                            api_key, 
                            temperature, 
                            max_tokens
                        )
                        
                        # Add bot response
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ خطأ: {str(e)}")
    
    with col_voice:
        if st.button("🎤 تسجيل صوتي"):
            st.info("🎤 ميزة التسجيل الصوتي قيد التطوير")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### 🎯 نصائح للاستخدام")
    
    tips = [
        "💡 استخدم أسئلة واضحة ومحددة",
        "📚 يمكنك السؤال عن أي موضوع تعليمي",
        "🔍 اطرح أسئلة متابعة للحصول على تفاصيل أكثر",
        "⭐ استخدم التقييم لتحسين الإجابات",
        "🔄 جرب أساليب مختلفة في السؤال"
    ]
    
    for tip in tips:
        st.markdown(f"<div style='padding: 0.5rem; margin: 0.5rem 0; background: #e3f2fd; border-radius: 5px;'>{tip}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("### ⚡ إجراءات سريعة")
    
    quick_questions = [
        "ما هو الذكاء الاصطناعي؟",
        "اشرح لي الرياضيات",
        "ما هو الفرق بين التعلم الآلي والذكاء الاصطناعي؟",
        "كيف أتعلم البرمجة؟"
    ]
    
    for question in quick_questions:
        if st.button(f"❓ {question}", key=f"quick_{question}"):
            st.session_state.messages.append({
                "role": "user", 
                "content": question,
                "timestamp": datetime.now().isoformat()
            })
            
            with st.spinner("🤔 جاري التفكير..."):
                try:
                    response = call_flowise_api(
                        question, 
                        flowise_url, 
                        flow_id, 
                        api_key, 
                        temperature, 
                        max_tokens
                    )
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ خطأ: {str(e)}")

def call_flowise_api(message, flowise_url, flow_id, api_key, temperature, max_tokens):
    """استدعاء Flowise API"""
    try:
        url = f"{flowise_url}/api/v1/prediction/{flow_id}"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "question": message,
            "overrideConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens,
                "systemMessage": "أنت مساعد تعليمي ذكي. أجب على أسئلة الطلاب بطريقة واضحة ومفيدة. استخدم أمثلة عملية واشرح المفاهيم بطريقة مبسطة."
            }
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("text", "عذراً، لم أتمكن من الإجابة")
        else:
            return f"خطأ في الاتصال: {response.status_code}"
            
    except requests.exceptions.Timeout:
        return "⏰ انتهت مهلة الاتصال. يرجى المحاولة مرة أخرى."
    except Exception as e:
        return f"❌ خطأ: {str(e)}"

def export_chat_data(messages):
    """تصدير بيانات المحادثة"""
    chat_data = {
        "export_timestamp": datetime.now().isoformat(),
        "total_messages": len(messages),
        "messages": messages
    }
    
    json_data = json.dumps(chat_data, ensure_ascii=False, indent=2)
    
    st.download_button(
        label="📥 تحميل المحادثة",
        data=json_data,
        file_name=f"education_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>🎓 مساعد التعليم الذكي</strong> | واجهة مستوردة من Flowise</p>
    <p>💡 تم تصميم هذه الواجهة لتحسين تجربة التعلم التفاعلية</p>
    <p>🔄 آخر تحديث: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
