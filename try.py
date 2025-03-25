import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
GROQ_API_KEY = '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'
# Groq API URL
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Streamlit App Title
st.title("💬 AI Chatbot Powered by Groq")

# Store chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("You:", key="user_input")

if st.button("Send"):
    if not GROQ_API_KEY:
        st.error("❌ API Key is missing! Set GROQ_API_KEY in .env")
    elif not user_input:
        st.warning("⚠️ Please enter a message.")
    else:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        # Append conversation history for context
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
        messages += st.session_state.chat_history
        messages.append({"role": "user", "content": user_input})

        payload = {
            "model": "llama3-8b-8192",  # Update model if needed
            "temperature": 0.7,
            "messages": messages
        }

        response = requests.post(GROQ_URL, json=payload, headers=headers)

        if response.status_code == 200:
            response_data = response.json()
            bot_reply = response_data["choices"][0]["message"]["content"]

            # Update chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

            # Display conversation
            for msg in st.session_state.chat_history:
                role = "🧑‍💻 You" if msg["role"] == "user" else "🤖 AI"
                st.write(f"**{role}:** {msg['content']}")
        else:
            st.error(f"API request failed! Status Code: {response.status_code}, Response: {response.text}")
