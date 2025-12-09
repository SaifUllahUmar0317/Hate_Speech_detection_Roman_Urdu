# 🛡️ Roman Urdu Hate Speech Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-Machine%20Learning-orange)
![Status](https://img.shields.io/badge/Status-Live-success)

A Machine Learning powered web application designed to detect and filter **Hate Speech** in **Roman Urdu** text. This tool utilizes Natural Language Processing (NLP) techniques to classify comments as either "Normal" or "Hate Speech," helping to create a safer online environment.

---

## 🔗 Live Demo
### 👉 [Click Here to Test the App](YOUR_STREAMLIT_APP_LINK_HERE)
*(Note: Replace the text above with your generated Streamlit link after deployment)*

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [How It Works](#-how-it-works)
- [Dataset & Preprocessing](#-dataset--preprocessing)
- [Model Architecture](#-model-architecture)
- [Installation & Local Setup](#-installation--local-setup)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Author](#-author)

---

## 🔍 Project Overview
Roman Urdu (Urdu written in English script) is widely used on social media in South Asia. However, standard English NLP models fail to detect toxicity in this script due to spelling variations and code-mixing.

This project solves that problem by:
1.  Normalizing Roman Urdu text.
2.  Removing noise (stopwords, punctuation).
3.  Using a trained Logistic Regression model to predict the nature of the text.

---

## 🚀 How It Works
1.  **User Input:** The user enters a sentence in the text area (e.g., *"Tu boht gnda insan hai"*).
2.  **Text Cleaning:** The app applies custom regex functions to:
    -   Remove URLs and special characters.
    -   Normalize slang (e.g., correcting `boht` -> `bohot`, `gnda` -> `ganda`).
    -   Remove Roman Urdu stopwords.
3.  **Vectorization:** The text is converted into numbers using a **TF-IDF Vectorizer** (trained on 1-gram and 2-gram combinations).
4.  **Prediction:** The Machine Learning model predicts the class.
5.  **Output:** -   🟢 **Normal:** Safe content.
    -   🔴 **Hate Speech:** Toxic or offensive content.

---

## 🧹 Dataset & Preprocessing
The model was trained on a labeled dataset of Roman Urdu comments. The preprocessing pipeline includes:

### Custom Slang Normalization
To handle the inconsistency of Roman Urdu typing, a dictionary map is used:
-   `acha` / `achha` → **achha**
-   `bhut` / `boht` → **bohot**
-   `gnda` → **ganda**
*(And many others)*

### Stopword Removal
Common words that add no semantic meaning (like `ka`, `ke`, `tha`, `hai`) are filtered out to improve model accuracy.

---

## 🧠 Model Architecture
-   **Algorithm:** Logistic Regression (Max Iterations: 1000)
-   **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency)
-   **N-Grams:** Unigrams and Bigrams (1,2)
-   **Split:** 75% Training, 25% Testing

The model was selected for its efficiency and high performance on text classification tasks with sparse data.

---

## 💻 Installation & Local Setup
If you want to run this app on your own computer, follow these steps:

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    cd YOUR_REPO_NAME
    ```

2.  **Install Dependencies**
    Ensure you have Python installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    streamlit run app.py
    ```

4.  **View in Browser**
    The app will open automatically at `http://localhost:8501`.

---

## 📂 Project Structure

```text
├── app.py                      # Main Streamlit application file
├── requirements.txt            # List of python libraries needed
├── roman_urdu_hate_model.pkl   # The trained Machine Learning model
├── roman_urdu_vectorizer.pkl   # Saved TF-IDF Vectorizer
├── training_script.py          # (Optional) The script used to train the model
├── README.md                   # Project documentation
└── dataset.csv                 # (Optional) The dataset used for training

---

## 🛠 Technologies UsedProject Structure

```text
* **[Python](https://www.python.org/)**: Core programming language.
* **[Streamlit](https://streamlit.io/)**: For building the web interface.
* **[Scikit-Learn](https://scikit-learn.org/)**: For model training and vectorization.
* **[Pandas](https://pandas.pydata.org/)**: For data manipulation.
* **[Joblib](https://joblib.readthedocs.io/)**: For saving and loading the model.

---

## 👤 Author
**[SAIF ULLAH UMAR]**
**[AI & ML Engineer]**


- Check out my [GitHub](https://github.com/SaifUllahUmar0317/)
