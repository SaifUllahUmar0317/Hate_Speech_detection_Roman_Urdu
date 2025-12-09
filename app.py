# streamlit_app.py

import streamlit as st
import joblib
import re
import string

# ============================
# Load Model and Vectorizer
# ============================

model = joblib.load("roman_urdu_hate_model.pkl")
vectorizer = joblib.load("roman_urdu_vectorizer.pkl")

# ============================
# Roman Urdu Cleaning & Normalization
# ============================

roman_stopwords = set([
    "ka","ki","ke","ko","mein","me","mai","hain","hai","tha","thi","hy",
    "kya","kon","koi","ye","wo","hon","hun","tha","tak","se","to","par",
    "aur","ya","wala","wali","wale","bhi","bhai","agar","magar"
])

normalization_map = {
    "boht": "bohot",
    "bohut": "bohot",
    "buht": "bohot",
    "bhut": "bohot",
    "acha": "achha",
    "gnda": "ganda",
    "gndu": "gandu",
    "kamina": "kameena",
    "pagal": "paagal",
    "lanat": "laanat"
}

def normalize_spellings(text):
    for wrong, correct in normalization_map.items():
        text = re.sub(r"\b" + wrong + r"\b", correct, text)
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = normalize_spellings(text)
    text = " ".join([word for word in text.split() if word not in roman_stopwords])
    return text

# ============================
# Streamlit UI
# ============================

st.title("Roman Urdu Hate Speech Classifier")
st.write("Enter a comment below to check if it contains hate speech:")

user_input = st.text_area("Enter comment here:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a comment!")
    else:
        cleaned_input = clean_text(user_input)
        input_vec = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_vec)[0]
        if prediction == 0:
            st.success(f"The comment is classified as: **Normal**")
        else:
            st.error(f"⚠️ This comment is classified as **Hate Speech**!")
        
        
