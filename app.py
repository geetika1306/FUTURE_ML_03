import streamlit as st
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Support Chatbot",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🤖 Customer Support Chatbot")
st.write("AI-powered chatbot using NLP and Machine Learning")
st.divider()

# -----------------------------
# Training Data
# -----------------------------
data = {
    "question": [
        "How can I reset my password?",
        "I forgot my password",
        "What is your refund policy?",
        "How do I get a refund?",
        "How can I contact customer support?",
        "Is customer care available 24/7?",
        "Where is my order?",
        "Track my order"
    ],
    "intent": [
        "password_reset",
        "password_reset",
        "refund",
        "refund",
        "contact_support",
        "contact_support",
        "order_tracking",
        "order_tracking"
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# Text Preprocessing
# -----------------------------
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

df["question"] = df["question"].apply(preprocess)

# -----------------------------
# Vectorization
# -----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["question"])
y = df["intent"]

# -----------------------------
# Train Model
# -----------------------------
model = LogisticRegression()
model.fit(X, y)

# -----------------------------
# Responses
# -----------------------------
responses = {
    "password_reset": "You can reset your password using the 'Forgot Password' option.",
    "refund": "Our refund policy allows refunds within 7 days of purchase.",
    "contact_support": "You can contact customer support via email or live chat.",
    "order_tracking": "Please provide your order ID to track your order."
}

# -----------------------------
# Chatbot Function (FIXED)
# -----------------------------
def chatbot_reply(user_input):
    user_input = preprocess(user_input)
    vector = vectorizer.transform([user_input])
    intent = model.predict(vector)[0]
    return responses.get(
        intent,
        "Sorry, I didn't understand that. Please contact support."
    )

# -----------------------------
# Streamlit UI
# -----------------------------
user_input = st.text_input("Ask your question:")

if user_input:
    st.success(f"🤖 Bot: {chatbot_reply(user_input)}")

st.divider()
st.caption("Built using Python, NLP, and Machine Learning")
