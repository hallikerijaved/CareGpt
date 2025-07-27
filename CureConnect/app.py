import json
import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime
import numpy as np
import pickle
import re
import random
import tensorflow as tf
from keras.models import load_model
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os

# Load model and assets
model = load_model('chatbot_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pickle', 'rb') as handle:
    lbl_enc = pickle.load(handle)
with open('intents.json', 'r') as file:
    data = json.load(file)
intents = data["intents"]

# Preprocess user text
def preprocess_text(text):
    text = re.sub('[^a-zA-Z\']', ' ', text)
    return " ".join(text.lower().split())

# Get model response
def get_model_response(query, intents):
    preprocessed_query = preprocess_text(query)
    x_test = tokenizer.texts_to_sequences([preprocessed_query])
    if not x_test[0]:
        return "I'm sorry, I couldn't understand that."
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, padding='post', maxlen=18)
    y_pred = model.predict(x_test)
    predicted_class = np.argmax(y_pred)
    tag = lbl_enc.inverse_transform([predicted_class])[0]
    responses = [intent["responses"] for intent in intents if intent["tag"] == tag]
    return random.choice(responses[0]) if responses else "I'm not sure how to respond."

# Render messages
def render_message(message, is_user):
    alignment = "margin-left:auto;" if is_user else "margin-right:auto;"
    bg_color = "linear-gradient(135deg, #6D9EEB, #3A75C4)" if is_user else "linear-gradient(135deg, #F0F0F0, #D5D5D5)"
    text_color = "#fefefe" if is_user else "#111"
    border_radius = "20px 20px 0 20px" if is_user else "20px 20px 20px 0"
    box_shadow = "0 4px 12px rgba(0, 0, 0, 0.1)"
    icon = "🧍" if is_user else "🤖"
    st.markdown(f"""
        <div style="background: {bg_color}; color: {text_color}; padding: 12px 18px; border-radius: {border_radius};
        max-width: 75%; {alignment} margin-bottom: 12px; font-size: 16px; font-family: 'Segoe UI', sans-serif; box-shadow: {box_shadow};">
            {icon} {message}
        </div>
    """, unsafe_allow_html=True)

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def conversation_to_txt(conversation):
    lines = []
    for m in conversation:
        sender = "You" if m["sender"] == "user" else "Bot"
        text = remove_emojis(m["text"])
        lines.append(f"{sender}: {text}")
    return "\n".join(lines)

# Speech recognizer
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak now.")
        audio = recognizer.listen(source, phrase_time_limit=5)
    try:
        return recognizer.recognize_google(audio)
    except Exception:
        st.error("Sorry, I couldn't understand the audio.")
        return None

import time  # Add this at the top with other imports

def play_audio(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        audio_path = fp.name

    audio_bytes = open(audio_path, 'rb').read()
    st.audio(audio_bytes, format='audio/mp3')

    # Delay deletion to avoid "file in use" error
    time.sleep(20)  # Adjust if needed
    try:
        os.remove(audio_path)
    except PermissionError:
        pass  # Silently ignore if still in use


# Main app
def main():
    st.set_page_config(page_title="Mental Health Chatbot", page_icon=None, layout="wide")

    st.markdown("""<h1 style='font-size: 48px; font-weight: 800;
        background: linear-gradient(90deg, #6D00F6, #FF4ECD, #00E0FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: "Segoe UI", sans-serif; margin-bottom: 0.2em;'>CareGPT: Mental Care, Anytime</h1>
        <p style='font-size:14px;color:#666;'>Feel free to chat. I'm here to listen.</p>""", unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'speech_mode' not in st.session_state:
        st.session_state.speech_mode = False

    with st.sidebar:
     st.title("🧠 Mental Health Bot")
     

     st.subheader("📌 Emergency Contacts")
     st.markdown("**📞 Suicide Prevention Lifeline**")
     st.markdown("📱 1-800-273-8255")
     st.markdown("**📱 Crisis Text Line**")
     st.markdown("💬 Text 'HELLO' to 741741")
     st.markdown("**🌐 Tele MANAS**")
     st.markdown("[Visit Website](https://telemanas.mohfw.gov.in/home)")
     st.markdown("---")  # Divider
      # Mood Tracker
     st.markdown("#### 📊 Mood Tracker")
     mood = st.selectbox("How are you feeling?", ["😊 Happy", "😢 Sad", "😰 Anxious", "😡 Angry", "😌 Calm", "🥱 Tired"])
     note = st.text_area("Optional note:")
     if st.button("Log Mood"):
        if "mood_log" not in st.session_state:
            st.session_state.mood_log = []
        st.session_state.mood_log.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "mood": mood,
            "note": note
        })
        st.success("Mood logged!")

     if "mood_log" in st.session_state and st.session_state.mood_log:
        df = pd.DataFrame(st.session_state.mood_log)
        st.markdown("**Mood History:**")
        st.dataframe(df[::-1], use_container_width=True)
     st.markdown("---")  # Divider
     st.subheader("💬 Chat History")
     if st.session_state.conversation:
        for msg in st.session_state.conversation:
            sender = "🧍 You" if msg["sender"] == "user" else "🤖 Bot"
            st.markdown(f"**{sender}:** {msg['text']}")
        chat_txt = conversation_to_txt(st.session_state.conversation)
        st.download_button("📥 Download Chat", chat_txt, file_name="chat_history.txt")
     else:
        st.warning("No chat history yet.")

     st.markdown("---")

     if st.button("🧹 Clear Chat"):
        st.session_state.conversation = []
        st.rerun()

    with st.container():
        for msg in st.session_state.conversation:
            render_message(msg["text"], is_user=(msg["sender"] == "user"))

    st.markdown("<div id='bottom'></div><script>document.getElementById('bottom').scrollIntoView({behavior: 'smooth'});</script>", unsafe_allow_html=True)

    user_input = st.text_input("Type your message here...", key="text_input", placeholder="Type your message here...")

    col1, col2 = st.columns([1, 0.15])
    with col1:
        if st.button("Send", use_container_width=True):
            input_text = user_input.strip()
            if input_text:
                response = get_model_response(input_text, intents)
                st.session_state.conversation.append({"sender": "user", "text": input_text})
                st.session_state.conversation.append({"sender": "bot", "text": response})
                st.session_state.speech_mode = False
                st.rerun()
            else:
                st.error("Please enter a message.")
    with col2:
        if st.button("🎤", help="Click to speak"):
            spoken = recognize_speech()
            if spoken:
                st.session_state.speech_mode = True
                response = get_model_response(spoken, intents)
                st.session_state.conversation.append({"sender": "user", "text": spoken})
                st.session_state.conversation.append({"sender": "bot", "text": response})
                play_audio(response)
                st.rerun()

if __name__ == "__main__":
    main()
