#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مساعد التعليم الذكي - Streamlit App
"""

import streamlit as st
import requests
import json
from datetime import datetime

# إعداد الصفحة
st.set_page_config(
    page_title="مساعد التعليم الذكي",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS مخصص
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎓 مساعد التعليم الذكي</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ الإعدادات")
    
    # إعدادات Flowise
    st.subheader("🔗 Flowise Settings")
    flowise_url = st.text_input("Flowise URL", value="http://localhost:3000")
    flow_id = st.text_input("Flow ID", value="your_flow_id_here")
    api_key = st.text_input("API Key", type="password", value="your_api_key_here")
    
    # إعدادات النموذج
    st.subheader("🤖 Model Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 100, 2000, 1000)
    
    # معلومات
    st.subheader("ℹ️ معلومات")
    st.info("مساعد تعليمي ذكي للتفاعل مع الطلاب")

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 المحادثة")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("اكتب سؤالك هنا..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("جاري التفكير..."):
                try:
                    # Call Flowise API
                    response = call_flowise_api(prompt, flowise_url, flow_id, api_key, temperature, max_tokens)
                    st.markdown(response)
                    
                    # Add bot response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"عذراً، حدث خطأ: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

with col2:
    st.header("📊 الإحصائيات")
    
    # Statistics
    total_messages = len(st.session_state.messages)
    user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
    bot_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
    
    st.metric("إجمالي الرسائل", total_messages)
    st.metric("رسائل المستخدم", user_messages)
    st.metric("ردود البوت", bot_messages)
    
    # Clear chat button
    if st.button("🗑️ مسح المحادثة", type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    # Export chat
    if st.button("📥 تصدير المحادثة"):
        export_chat(st.session_state.messages)

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
        return "انتهت مهلة الاتصال. يرجى المحاولة مرة أخرى."
    except Exception as e:
        return f"خطأ: {str(e)}"

def export_chat(messages):
    """تصدير المحادثة"""
    chat_data = {
        "timestamp": datetime.now().isoformat(),
        "messages": messages,
        "total_messages": len(messages)
    }
    
    st.download_button(
        label="تحميل المحادثة",
        data=json.dumps(chat_data, ensure_ascii=False, indent=2),
        file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎓 مساعد التعليم الذكي | تم بناؤه باستخدام Streamlit و Flowise</p>
    <p>💡 نصائح: استخدم أسئلة واضحة ومحددة للحصول على أفضل النتائج</p>
</div>
""", unsafe_allow_html=True)
