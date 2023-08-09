# Demo 용 웹 서버 (8501 포트 사용)

import streamlit as st
from streamlit_chat import message
import pandas as pd
import requests
import json

class llama_web():
    def __init__(self):
        self.URL = "http://127.0.0.1:8000/chat" #언어 모델 API 서버 주소(8000 포트) 
        self.count = 0
        st.set_page_config(
                page_title="Streamlit Chat -Demo",
                page_icon="robot"
                )

    def get_answer(self, message):  #유저 input 을 언어모델 API로 보내어 응답 받아오는 함수
        param = {'user_message': message}
        resp = requests.post(self.URL, json=param)
        output = json.loads(resp.content)
        output = output['message']
        return output

    def update_log(self, user_message, bot_message):
        if 'chat_log' not in st.session_state:
            st.session_state.chat_log = {"user_message": [], "bot_message": []}

        st.session_state.chat_log['user_message'].append(user_message)
        st.session_state.chat_log['bot_message'].append(bot_message)

        return st.session_state.chat_log

    def window(self):
        st.title("IMR chatbot")
        st.header("Custom instruction tunned polyglot-ko 5.8B model")
        st.caption("created by IMR 인턴 양주영")
        st.write("Hello!")
        st.text_input("type a message..", key="user_message")
        
        #message = st.session_state.user_message
        #st.write(f"user message: {message}")
        

        if user_message := st.session_state['user_message']:
            output = self.get_answer(user_message)
            #st.write(f"user message: {message}")
            #st.write(f"bot message: {output}")
            st.success(f"bot message: {output}")

            chat_log = self.update_log(user_message, output)
            st.write(pd.DataFrame(st.session_state.chat_log))


    def chat(self):
        st.title("IMR chatbot")
        st.header("Custom fine-tuned polyglot-ko 5.8B model")
        st.caption("created by IMR 인턴 양주영")

        #message("Hello. I'm summer", is_user=True)
        #message("Hello, I'm bot")

        st.text_input("Ask a Question..", key="user_message")

        if user_message := st.session_state['user_message']:
            output = self.get_answer(user_message)
            # message(user_message, is_user=True)
            # message(output)
            
            chat_log = self.update_log(user_message, output)
            bot_messages = chat_log['bot_message'][::-1]
            user_messages = chat_log['user_message'][::-1]
            
            for idx, (bot, user) in enumerate(zip(bot_messages, user_messages)):
                message(bot, key=f"{idx}_bot")
                message(user, key=str(idx), is_user=True)

            

if __name__ == '__main__':
    web = llama_web()
    web.chat()
