# 🛡️ Roman Urdu Hate Speech Detection System

A Machine Learning powered web application designed to detect and filter Hate Speech in Roman Urdu text. This tool uses Natural Language Processing (NLP) to classify comments as either **Normal** or **Hate Speech**, helping to create a safer online environment. The system includes a complete training pipeline with multiple models, hyperparameter tuning, and an interactive Streamlit dashboard.

# Project Title

[![Watch the demo](https://hatespeechrecognition.streamlit.app/)

---

# 📖 Table of Contents

* [Project Overview](#-project-overview)
* [Key Features](#-key-features)
* [How It Works](#-how-it-works)
* [Dataset & Preprocessing](#-dataset--preprocessing)
* [Model Training & Selection](#-model-training--selection)
* [Installation & Local Setup](#-installation--local-setup)
* [Project Structure](#-project-structure)
* [Technologies Used](#-technologies-used)
* [Team](#-team)
* [License](#-license)

---

# 🔍 Project Overview

Roman Urdu (Urdu written in English script) is widely used on social media in South Asia. Standard English NLP models fail to detect toxicity in this script due to spelling variations, code-mixing, and lack of training data.

This project solves that problem by:

* Normalizing Roman Urdu slang and spelling variations.
* Removing noise (stopwords, punctuation, emojis).
* Training and comparing four ML models (Logistic Regression, Random Forest, SVM, Naive Bayes) with hyperparameter tuning.
* Selecting the best model based on F1-score and training efficiency.
* Providing an interactive web app for real-time detection, batch processing, chat simulation, and analytics.

---

# ✨ Key Features

| Feature                 | Description                                                                                           |
| ----------------------- | ----------------------------------------------------------------------------------------------------- |
| **Real-time Analysis**  | Type or paste Roman Urdu text and get instant prediction with confidence score and word highlighting. |
| **Batch Processing**    | Upload a CSV file or paste multiple comments for bulk analysis and download results.                  |
| **Chat Simulator**      | Simulate a conversation where the bot detects hate speech in user messages.                           |
| **Analytics Dashboard** | View statistics, trends, confidence distribution, and user feedback.                                  |
| **Model Performance**   | Sidebar displays accuracy, F1-score, and a gauge chart for model confidence.                          |
| **Feedback System**     | Users can flag correct or incorrect predictions to improve future models.                             |
| **Explainable AI**      | Visualization of preprocessing steps, hate-word highlighting, and feature importance.                 |

---

# 🚀 How It Works

## 1. User Input

The user enters a Roman Urdu sentence.

### Example

```text
Tu boht gnda insan hai
```

## 2. Text Cleaning

The preprocessing pipeline performs:

* Convert text to lowercase
* Remove URLs
* Remove punctuation
* Remove numbers
* Remove emojis
* Normalize slang and spelling variations
* Remove Roman Urdu stopwords

### Example

```text
boht → bohot
gnda → ganda
```

## 3. Vectorization

The cleaned text is converted into numerical features using **TF-IDF Vectorization**.

```python
ngram_range=(1,3)
```

This includes:

* Unigrams
* Bigrams
* Trigrams

## 4. Prediction

The trained machine learning model predicts whether the text contains hate speech.

## 5. Output

### 🟢 Normal

Safe and non-offensive content.

### 🔴 Hate Speech

Toxic or offensive content with highlighted hate-related words.

---

# 🧹 Dataset & Preprocessing

The model was trained on a labeled dataset containing Roman Urdu comments categorized as:

* Normal
* Hate Speech

## Custom Slang Normalization

| Original | Normalized |
| -------- | ---------- |
| boht     | bohot      |
| bohut    | bohot      |
| buht     | bohot      |
| bhut     | bohot      |
| acha     | achha      |
| gnda     | ganda      |
| gndu     | gandu      |
| kamina   | kameena    |
| pagal    | paagal     |
| chutiya  | stupid     |
| bsdk     | abusive    |

## Noise Removal

The preprocessing pipeline removes:

* Stopwords
* Punctuation
* Numbers
* Emojis
* URLs
* Extra spaces

### Example Stopwords

```text
ka
ki
hai
tha
thi
hain
```

---

# 🧠 Model Training & Selection

The training pipeline is implemented in:

```text
main.py
```

## Exploratory Data Analysis (EDA)

The system performs:

* Class distribution analysis
* Text length distribution
* Boxplots and visualizations

## Train/Test Split

```python
80% Training Data
20% Testing Data
```

Stratified sampling is used to preserve class balance.

## TF-IDF Configuration

```python
TfidfVectorizer(
    ngram_range=(1,3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
```

## Models Evaluated

### Logistic Regression

```python
LogisticRegression()
```

### Random Forest

```python
RandomForestClassifier()
```

### Support Vector Machine (SVM)

```python
SVC()
```

### Multinomial Naive Bayes

```python
MultinomialNB()
```

## Hyperparameter Optimization

```python
RandomizedSearchCV
```

Used for automated parameter tuning and model optimization.

## Evaluation Metrics

The following metrics are calculated:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* 5-Fold Cross Validation F1 Score

## Model Selection Strategy

1. Calculate F1-score for all models.
2. Select all models within 1% of the highest F1-score.
3. Choose the model with the fastest training time.

This ensures a balance between performance and efficiency.

---

# 💻 Installation & Local Setup

## Prerequisites

* Python 3.8 or higher
* pip package manager

## Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

## Step 2: Create Virtual Environment (Recommended)

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Launch Streamlit Application

```bash
streamlit run app.py
```

## Step 5: Open Browser

Visit:

```text
http://localhost:8501
```

---

# 📂 Project Structure

```text
├── app.py                          # Main Streamlit application
├── main.py                         # Training pipeline
├── requirements.txt                # Python dependencies
├── dataset.csv                     # Raw dataset
├── best_roman_urdu_hate_model.pkl  # Best trained model
├── roman_urdu_vectorizer.pkl       # TF-IDF vectorizer
└── README.md                       # Documentation
```

---

# 🛠 Technologies Used

| Area                  | Tools                                                  |
| --------------------- | ------------------------------------------------------ |
| Language              | Python 3                                               |
| Web Framework         | Streamlit                                              |
| Machine Learning      | scikit-learn                                           |
| Models                | Logistic Regression, Random Forest, SVC, MultinomialNB |
| NLP                   | TF-IDF Vectorization                                   |
| Hyperparameter Tuning | RandomizedSearchCV                                     |
| Data Handling         | pandas, numpy                                          |
| Visualization         | matplotlib, seaborn, plotly, wordcloud                 |
| Model Persistence     | joblib                                                 |

---

# 👥 Team

### 👩 Izza Mustafa

**Role:** Pipeline Designing

### 👨 Ahtisham Ul Haq

**Role:** Dataset Creation

### 👨‍💻 Saifullah Umar

**Role:** Frontend Designing

---

# 📄 License

This project is developed for **educational and research purposes**.

If you reuse, modify, or distribute any part of this project, please provide appropriate credit to the original authors.

```text
© Roman Urdu Hate Speech Detection System
All Rights Reserved.
```
