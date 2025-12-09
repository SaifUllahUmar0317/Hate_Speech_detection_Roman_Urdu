# ============================
# Roman Urdu Hate Speech Classifier (Binary)
# ============================

import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ============================
# 1. Load Dataset
# ============================

df = pd.read_csv("Roman_Urdu_dataset.csv")

# Ensure correct column names
df.columns = ["text", "label"]

# Remove nulls
print(df.isnull().sum())
print(df.duplicated().sum())
df.drop_duplicates()
df.dropna(inplace=True)

df["label"] = df["label"].map({ "N": 1, "H": 0, "O": 0 })

# ============================
# 2. Roman Urdu Normalization
# ============================

# Roman Urdu stopwords (short version; you can expand)
roman_stopwords = set([
    "ka","ki","ke","ko","mein","me","mai","hain","hai","tha","thi","hy",
    "kya","kon","koi","ye","wo","hon","hun","tha","tak","se","to","par",
    "aur","ya","wala","wali","wale","bhi","bhai","agar","magar"
])

# Slang / spelling normalization dictionary
normalization_map = {
    "boht": "bohot",
    "bohut": "bohot",
    "buht": "bohot",
    "bhut": "bohot",
    "acha": "achha",
    "acha": "achha",
    "acha": "achha",
    "gnda": "ganda",
    "gndu": "gandu",
    "kamina": "kameena",
    "pagal": "paagal",
    "pagal": "paagal",
    "lanat": "laanat"
}

def normalize_spellings(text):
    for wrong, correct in normalization_map.items():
        text = re.sub(r"\b" + wrong + r"\b", correct, text)
    return text

# ============================
# 3. Text Cleaning Function
# ============================

def clean_text(text):
    text = text.lower()

    # remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # remove numbers
    text = re.sub(r"\d+", "", text)

    # normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # apply spelling normalization
    text = normalize_spellings(text)

    # remove stopwords
    text = " ".join([word for word in text.split() if word not in roman_stopwords])

    return text

# Apply cleaning
df["clean_text"] = df["text"].apply(clean_text)


# ============================
# 4. Split Data
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
)

# ============================
# 5. Vectorization (TF-IDF)
# ============================

vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ============================
# 6. Model Training
# ============================

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ============================
# 7. Evaluation
# ============================

predictions = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))

# ============================
# 8. Save Model and Vectorizer
# ============================

joblib.dump(model, "roman_urdu_hate_model.pkl")
joblib.dump(vectorizer, "roman_urdu_vectorizer.pkl")

print("\nModel Saved Successfully!")