# 🎓 مساعد التعليم الذكي - الواجهة المحلية
# تشغيل: streamlit run local_app.py

import streamlit as st
import requests
import json
from datetime import datetime
import time

# إعداد الصفحة
st.set_page_config(
    page_title="مساعد التعليم الذكي",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS مخصص للواجهة الجميلة
st.markdown("""
<style>
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
        text-align: right;
    }
    
    .bot-message {
        background: #f0f2f6;
        color: #333;
        border: 1px solid #e0e0e0;
        margin-right: auto;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# الشريط الجانبي
with st.sidebar:
    st.markdown("## ⚙️ إعدادات API")
    
    # رابط API
    api_url = st.text_input(
        "🌐 رابط Colab API:",
        placeholder="https://xxxxx.ngrok.io",
        help="انسخ الرابط من Google Colab"
    )
    
    # فحص الاتصال
    if st.button("🔍 فحص الاتصال"):
        if api_url:
            try:
                response = requests.get(f"{api_url}/health", timeout=5)
                if response.status_code == 200:
                    st.markdown('<div class="status-success">✅ الاتصال ناجح!</div>', unsafe_allow_html=True)
                    data = response.json()
                    st.json(data)
                else:
                    st.markdown('<div class="status-error">❌ فشل الاتصال</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="status-error">❌ خطأ: {str(e)}</div>', unsafe_allow_html=True)
        else:
            st.warning("يرجى إدخال رابط API")
    
    st.markdown("---")
    
    # إعدادات الرسائل
    st.markdown("## 📝 إعدادات الرسائل")
    max_length = st.slider("طول الإجابة القصوى:", 50, 1000, 512)
    
    # أمثلة سريعة
    st.markdown("## 💡 أمثلة سريعة")
    quick_questions = [
        "ما هو الذكاء الاصطناعي؟",
        "اشرح لي الفيزياء النووية",
        "كيف أتعلم البرمجة؟",
        "ما هي فوائد الرياضيات؟",
        "اشرح لي تاريخ مصر"
    ]
    
    selected_question = st.selectbox("اختر سؤال سريع:", [""] + quick_questions)

# الرأس الرئيسي
st.markdown("""
<div class="main-header">
    <h1>🎓 مساعد التعليم الذكي</h1>
    <p>مدعوم بنموذج Llama 3.1 - واجهة احترافية للتعلم التفاعلي</p>
</div>
""", unsafe_allow_html=True)

# التحقق من وجود API
if not api_url:
    st.error("⚠️ يرجى إدخال رابط Colab API في الشريط الجانبي")
    st.info("""
    **خطوات الإعداد:**
    1. افتح Google Colab
    2. شغل ملف `llama_api_server.py`
    3. انسخ الرابط من ngrok
    4. الصقه في الشريط الجانبي
    """)
    st.stop()

# التبويبات الرئيسية
tab1, tab2, tab3 = st.tabs(["💬 محادثة", "📊 الإحصائيات", "🔧 اختبار API"])

with tab1:
    # إعداد حالة المحادثة
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # عرض المحادثات السابقة
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🧑‍🎓 أنت:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 المساعد:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # خانة الرسالة
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_area(
            "اكتب سؤالك هنا:",
            value=selected_question if selected_question else "",
            placeholder="مثال: ما هو الذكاء الاصطناعي؟",
            height=100,
            key="user_input"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        send_button = st.button("📤 إرسال", type="primary", use_container_width=True)
        clear_button = st.button("🗑️ مسح المحادثة", use_container_width=True)
    
    # معالجة الرسالة
    if send_button and user_input.strip():
        # إضافة رسالة المستخدم
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # عرض رسالة المستخدم
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🧑‍🎓 أنت:</strong><br>
            {user_input}
        </div>
        """, unsafe_allow_html=True)
        
        # إرسال الطلب للAPI
        with st.spinner("🤖 المساعد يفكر..."):
            try:
                response = requests.post(
                    f"{api_url}/chat",
                    json={"message": user_input},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        bot_response = data["response"]
                        
                        # إضافة رد المساعد
                        st.session_state.messages.append({"role": "assistant", "content": bot_response})
                        
                        # عرض رد المساعد
                        st.markdown(f"""
                        <div class="chat-message bot-message">
                            <strong>🤖 المساعد:</strong><br>
                            {bot_response}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # إعادة تحديث الصفحة لإظهار الرسائل الجديدة
                        st.rerun()
                    else:
                        st.error(f"❌ خطأ من API: {data.get('error', 'خطأ غير محدد')}")
                else:
                    st.error(f"❌ خطأ في الاتصال: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                st.error("⏰ انتهت مهلة الاتصال. يرجى المحاولة مرة أخرى.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 خطأ في الاتصال. تأكد من أن API يعمل.")
            except Exception as e:
                st.error(f"❌ خطأ: {str(e)}")
    
    # مسح المحادثة
    if clear_button:
        st.session_state.messages = []
        st.rerun()

with tab2:
    st.markdown("### 📊 إحصائيات الاستخدام")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("عدد الرسائل", len(st.session_state.get("messages", [])))
    
    with col2:
        user_messages = len([m for m in st.session_state.get("messages", []) if m["role"] == "user"])
        st.metric("أسئلة المستخدم", user_messages)
    
    with col3:
        bot_messages = len([m for m in st.session_state.get("messages", []) if m["role"] == "assistant"])
        st.metric("إجابات المساعد", bot_messages)
    
    # رسم بياني للرسائل
    if st.session_state.get("messages"):
        st.markdown("#### 📈 توزيع الرسائل")
        
        import pandas as pd
        import plotly.express as px
        
        message_counts = {
            "النوع": ["أسئلة المستخدم", "إجابات المساعد"],
            "العدد": [user_messages, bot_messages]
        }
        
        df = pd.DataFrame(message_counts)
        fig = px.pie(df, values="العدد", names="النوع", title="توزيع الرسائل")
        st.plotly_chart(fig, use_container_width=True)
    
    # تصدير المحادثة
    if st.session_state.get("messages"):
        st.markdown("#### 💾 تصدير المحادثة")
        
        if st.button("📥 تصدير كـ JSON"):
            chat_data = {
                "timestamp": datetime.now().isoformat(),
                "total_messages": len(st.session_state.messages),
                "messages": st.session_state.messages
            }
            
            st.download_button(
                label="💾 تحميل ملف JSON",
                data=json.dumps(chat_data, ensure_ascii=False, indent=2),
                file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

with tab3:
    st.markdown("### 🔧 اختبار API")
    
    # معلومات API
    if api_url:
        st.markdown(f"**رابط API:** {api_url}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🏠 اختبار الصفحة الرئيسية"):
                try:
                    response = requests.get(f"{api_url}/", timeout=5)
                    st.json(response.json())
                except Exception as e:
                    st.error(f"خطأ: {e}")
        
        with col2:
            if st.button("❤️ اختبار فحص الصحة"):
                try:
                    response = requests.get(f"{api_url}/health", timeout=5)
                    st.json(response.json())
                except Exception as e:
                    st.error(f"خطأ: {e}")
        
        # اختبار رسالة مخصصة
        st.markdown("#### 📝 اختبار رسالة مخصصة")
        test_message = st.text_input("رسالة الاختبار:", "مرحبا")
        
        if st.button("🧪 إرسال اختبار"):
            try:
                response = requests.post(
                    f"{api_url}/chat",
                    json={"message": test_message},
                    timeout=30
                )
                st.json(response.json())
            except Exception as e:
                st.error(f"خطأ: {e}")
    
    # معلومات تقنية
    st.markdown("#### 🛠️ معلومات تقنية")
    st.info("""
    **متطلبات التشغيل:**
    - Python 3.7+
    - Streamlit
    - Requests
    - Google Colab مع ngrok
    
    **للتشغيل:**
    ```bash
    pip install streamlit requests plotly pandas
    streamlit run local_app.py
    ```
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🎓 مساعد التعليم الذكي | مدعوم بنموذج Llama 3.1</p>
    <p>تم تطويره بحب ❤️ للتعليم الذكي</p>
</div>
""", unsafe_allow_html=True)
