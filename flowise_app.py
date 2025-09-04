#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مساعد التعليم الذكي - Flowise Integration
"""

import requests
import json
from datetime import datetime

class FlowiseEducationBot:
    def __init__(self, flowise_url="http://localhost:3000"):
        self.flowise_url = flowise_url
        self.api_key = "your_api_key_here"
    
    def send_message(self, message):
        """إرسال رسالة للبوت"""
        try:
            url = f"{self.flowise_url}/api/v1/prediction/your_chatflow_id"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "question": message,
                "overrideConfig": {
                    "systemMessage": "أنت مساعد تعليمي ذكي. أجب على أسئلة الطلاب بطريقة واضحة ومفيدة."
                }
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("text", "عذراً، لم أتمكن من الإجابة")
            else:
                return f"خطأ في الاتصال: {response.status_code}"
                
        except Exception as e:
            return f"خطأ: {str(e)}"
    
    def create_education_flow(self):
        """إنشاء Flow تعليمي"""
        flow_config = {
            "name": "مساعد التعليم الذكي",
            "description": "مساعد تعليمي ذكي للطلاب",
            "nodes": [
                {
                    "id": "1",
                    "type": "ChatOpenAI",
                    "data": {
                        "label": "OpenAI Chat",
                        "name": "openai_chat",
                        "model": "gpt-3.5-turbo",
                        "temperature": 0.7,
                        "maxTokens": 1000,
                        "systemMessage": "أنت مساعد تعليمي ذكي. أجب على أسئلة الطلاب بطريقة واضحة ومفيدة. استخدم أمثلة عملية واشرح المفاهيم بطريقة مبسطة."
                    }
                },
                {
                    "id": "2",
                    "type": "ChatInput",
                    "data": {
                        "label": "Chat Input",
                        "name": "chat_input"
                    }
                },
                {
                    "id": "3",
                    "type": "ChatOutput",
                    "data": {
                        "label": "Chat Output",
                        "name": "chat_output"
                    }
                }
            ],
            "edges": [
                {
                    "source": "2",
                    "target": "1",
                    "sourceHandle": "output",
                    "targetHandle": "input"
                },
                {
                    "source": "1",
                    "target": "3",
                    "sourceHandle": "output",
                    "targetHandle": "input"
                }
            ]
        }
        
        return flow_config

# مثال على الاستخدام
if __name__ == "__main__":
    bot = FlowiseEducationBot()
    
    # إنشاء Flow
    flow = bot.create_education_flow()
    print("✅ تم إنشاء Flow تعليمي")
    print(json.dumps(flow, indent=2, ensure_ascii=False))
    
    # اختبار البوت
    test_message = "ما هو الذكاء الاصطناعي؟"
    response = bot.send_message(test_message)
    print(f"السؤال: {test_message}")
    print(f"الإجابة: {response}")
