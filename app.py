import streamlit as st

st.title("Customer Support Chatbot")

user_input = st.text_input("Ask your question:")

if user_input:
    st.write("Bot:", chatbot_reply(user_input))
