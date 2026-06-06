# ============================
# Roman Urdu Hate Speech Classifier (Binary)
# ============================

import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


# ============================
# 1. Load Dataset
# ============================

df = pd.read_csv("dataset.csv")

# Ensure correct column names
df.columns = ["text", "label"]


# ============================
# 2. Checking imbalancing
# ============================

sns.countplot(data=df, x="label")
plt.title("Label Distribution")
plt.show()


# =============================
# 3. Preprocessing 
# =============================

print(df.isnull().sum())
print(df.duplicated().sum())
df.drop_duplicates()
df.dropna(inplace=True)


# ============================
# 4. Roman Urdu Normalization
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

import re

def remove_emojis(text):
    emoji_pattern = re.compile(
        "[" 
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002500-\U00002BEF"  # Chinese symbols
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"                # dingbats
        u"\u3030"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)


# ============================
# 5. Text Cleaning Function
# ============================

def clean_text(text):
    text = text.lower()

    # remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # remove numbers
    text = re.sub(r"\d+", "", text)

    # remove emojis, stickers etc
    text = remove_emojis(text)

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
# 6. Split Data
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
)


# ============================
# 7. Vectorization (TF-IDF)
# ============================

vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ============================
# 8. Model Training
# ============================

models = {
    "RandomForest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [100,150,200],
            "max_depth": [5,7,9,10],
            "min_samples_split": [2,4,6,8]
        }
    },
    
    "LogisticRegression":{
        "model": LogisticRegression(),
        "params":{
            "max_iter": [100,500,1000]
        }
    }
}

scores = []
final_model = None
best_score = 0
for name,model_params in models.items():
    # Model Training and prediction
    best_model_params = RandomizedSearchCV(estimator = model_params["model"], param_distributions=model_params["params"], cv=5)
    best_model_params.fit(X_train_vec, y_train)
    
    if best_model_params.best_score_ > best_score:
        best_score = best_model_params.best_score_
        final_model = best_model_params.best_estimator_
   
    scores.append({
        "model": name,
        "best parameters": best_model_params.best_params_,
        "score": best_model_params.best_score_
    })

params_scores_df = pd.DataFrame(scores)
params_scores_df


# ============================
# 9. Evaluation
# ============================

predictions = final_model.predict(X_test_vec)

acc = accuracy_score(y_test, predictions)
pre = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

cm = confusion_matrix(y_test, predictions)
print("Accuracy:", acc)
print("Precision:", pre)
print("Recall:", recall)
print("f1_score:", f1)

print("\nConfusion Matrix:\n")

sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Normal", "Hate"],
           yticklabels=["Normal", "Hate"])
plt.title("Confusion Matrix")
plt.show()


# ============================
# 10. Save Model and Vectorizer
# ============================

joblib.dump(final_model, "roman_urdu_hate_model.pkl")
joblib.dump(vectorizer, "roman_urdu_vectorizer.pkl")

print("\nModel Saved Successfully!")