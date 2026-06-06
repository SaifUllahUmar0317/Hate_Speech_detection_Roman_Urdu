#!/usr/bin/env python
# coding: utf-8

"""
Enhanced Roman Urdu Hate Speech Classifier
Features: Real-time analysis, word highlighting, batch processing, feedback system
"""

import streamlit as st
import joblib
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import numpy as np
from datetime import datetime
import json
import os

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Roman Urdu Hate Speech Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
    }
    
    /* Result cards */
    .result-card-safe {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        animation: fadeIn 0.5s ease-in;
    }
    
    .result-card-hate {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Confidence meter */
    .confidence-meter {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Chat bubble styling */
    .chat-bubble-user {
        background: #667eea;
        color: white;
        padding: 1rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.5rem 0;
        max-width: 80%;
        float: right;
        clear: both;
    }
    
    .chat-bubble-bot {
        background: #f0f2f6;
        color: #333;
        padding: 1rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.5rem 0;
        max-width: 80%;
        float: left;
        clear: both;
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Warning text highlighting */
    .hate-word {
        background-color: #ff6b6b;
        color: white;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# Load Model and Vectorizer
# ============================
@st.cache_resource
def load_models():
    model = joblib.load("best_roman_urdu_hate_model.pkl")
    vectorizer = joblib.load("roman_urdu_vectorizer.pkl")
    results = joblib.load("model_results.pkl")
    
    # Try to load selection metadata (for backward compatibility)
    try:
        selection_metadata = joblib.load("selection_metadata.pkl")
    except:
        # If file doesn't exist, create it from results
        selected_model_row = results[results.get('Selected', False)]
        if len(selected_model_row) > 0:
            selection_metadata = {
                'selected_model': selected_model_row['Model'].iloc[0],
                'selected_f1': selected_model_row['F1-Score'].iloc[0],
                'selected_accuracy': selected_model_row['Accuracy'].iloc[0],
                'selection_criteria': 'Legacy: Based on saved model selection'
            }
        else:
            # Fallback: use model with highest F1
            best_row = results.loc[results['F1-Score'].idxmax()]
            selection_metadata = {
                'selected_model': best_row['Model'],
                'selected_f1': best_row['F1-Score'],
                'selected_accuracy': best_row['Accuracy'],
                'selection_criteria': 'Legacy: Default to highest F1-Score'
            }
    
    return model, vectorizer, results, selection_metadata

model, vectorizer, results_df, selection_metadata = load_models()

# ============================
# Roman Urdu Resources
# ============================
roman_stopwords = set([
    "ka","ki","ke","ko","mein","me","mai","hain","hai","tha","thi","hy",
    "kya","kon","koi","ye","wo","hon","hun","tha","tak","se","to","par",
    "aur","ya","wala","wali","wale","bhi","bhai","agar","magar","na","ne"
])

normalization_map = {
    "boht": "bohot", "bohut": "bohot", "buht": "bohot", "bhut": "bohot",
    "acha": "achha", "gnda": "ganda", "gndu": "gandu", "kamina": "kameena",
    "pagal": "paagal", "lanat": "laanat", "teri": "tumhari", "maa": "mother",
    "behan": "sister", "bhen": "sister", "chutiya": "stupid", "bsdk": "abusive",
    # ADD THESE:
    "kharab": "kharab", "bura": "bura", "bure": "bure", "buri": "buri",
    "soch": "soch", "ghalat": "ghalat", "bewaqoof": "bewaqoof"
}

# Expand hate indicators list
hate_indicators = [
    "ganda", "gandu", "kamina", "kameena", "paagal", "chutiya", 
    "bsdk", "lanat", "badtameez", "bewaqoof", "ahmaq",
    # ADD THESE:
    "bura", "bure", "buri", "kharab", "ghalat", "soch kharab"
]

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

def highlight_hate_words(text):
    """Highlight hate speech indicators in the text"""
    highlighted = text
    for word in hate_indicators:
        if word in highlighted.lower():
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            highlighted = pattern.sub(f'<span class="hate-word">{word}</span>', highlighted)
    return highlighted

def analyze_text_sentiment(text):
    """Simple sentiment analysis based on word patterns"""
    positive_words = ["acha", "bohot acha", "maza", "umda", "khoobsurat", "pyara"]
    negative_words = ["bura", "ganda", "kharab", "badtameez", "bewaqoof"]
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return "Positive", "😊", "#84fab0"
    elif neg_count > pos_count:
        return "Negative", "😠", "#f5576c"
    else:
        return "Neutral", "😐", "#ffd93d"

# ============================
# Session State Initialization
# ============================
if 'history' not in st.session_state:
    st.session_state.history = []
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# ============================
# Sidebar - Enhanced
# ============================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.title("🛡️ Hate Shield")
    st.markdown("*AI-Powered Roman Urdu Moderation*")
    
    st.markdown("---")
    
    # Model Performance Dashboard
    with st.expander("📊 Model Performance", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{results_df['Accuracy'].max():.1%}")
        with col2:
            st.metric("F1-Score", f"{results_df['F1-Score'].max():.1%}")
        
        # Show SELECTED model (not just highest F1)
        selected_model_name = selection_metadata['selected_model']
        selected_f1 = selection_metadata['selected_f1']
        st.caption(f"🏆 Best Model: **{selected_model_name}**")
        
        # Gauge chart for confidence - use SELECTED model's F1
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=selected_f1 * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Model Confidence"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "#667eea"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgray"},
                       {'range': [50, 80], 'color': "gray"},
                       {'range': [80, 100], 'color': "darkgray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 90}}))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, width="stretch")
    
    st.markdown("---")
    
    # Quick Stats
    st.subheader("📈 Today's Stats")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="stat-card"><div class="stat-number">' + 
                   str(len(st.session_state.history)) + '</div><div>Checks</div></div>', 
                   unsafe_allow_html=True)
    with col2:
        hate_count = sum(1 for h in st.session_state.history if h.get('prediction') == 1)
        st.markdown('<div class="stat-card"><div class="stat-number" style="color:#f5576c;">' + 
                   str(hate_count) + '</div><div>Flagged</div></div>', 
                   unsafe_allow_html=True)
    with col3:
        safe_count = len(st.session_state.history) - hate_count
        st.markdown('<div class="stat-card"><div class="stat-number" style="color:#84fab0;">' + 
                   str(safe_count) + '</div><div>Safe</div></div>', 
                   unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Settings
    with st.expander("⚙️ Settings"):
        st.session_state.dark_mode = st.toggle("🌙 Dark Mode", st.session_state.dark_mode)
        auto_detect = st.checkbox("Auto-detect hate words", value=True)
        show_confidence = st.checkbox("Show confidence details", value=True)
    
    st.markdown("---")

# ============================
# Main Header
# ============================
st.markdown(f"""
<div class="main-header">
    <h1>🛡️ Roman Urdu Hate Speech Detector</h1>
    <p>AI-powered content moderation for Roman Urdu text | Real-time detection with explainable AI</p>
</div>
""", unsafe_allow_html=True)

# ============================
# Tabs for Different Modes
# ============================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Single Analysis", 
    "📁 Batch Processing", 
    "💬 Chat Simulator",
    "📊 Analytics Dashboard",
    "ℹ️ About & Guide"
])

# ============================================
# TAB 1: Single Analysis (Enhanced)
# ============================================
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Comment for Analysis")
        
        # Example quick-select buttons
        st.markdown("**Quick Examples:**")
        example_cols = st.columns(4)
        examples = {
            "😊 Normal Example": "Yeh mausam bohot acha hai",  # Will be classified as NORMAL
            "🚫 Hate Speech": "Tum bohot bure ho, tum se nafrat hai",  # Will be classified as HATE
            "😊 Another Normal": "Aaj ka din bohot umda hai",  # Will be classified as NORMAL
            "🚫 Another Hate": "Dafa ho jao, badtameez"  # Will be classified as HATE
        }
        
        for idx, (label, text) in enumerate(examples.items()):
            with example_cols[idx % 4]:
                if st.button(label, key=f"ex_{idx}", width="stretch"):
                    st.session_state.current_text = text
        
        # Main text input
        if 'current_text' not in st.session_state:
            st.session_state.current_text = ""
        
        user_input = st.text_area(
            "**Type or paste Roman Urdu text:**",
            value=st.session_state.current_text,
            placeholder="Example: Tum bohat bure ho, tumhari soch kharab hai...",
            height=150,
            key="main_input"
        )
        
        # Analysis options
        col_a, col_b, col_c = st.columns([1, 1, 2])
        with col_a:
            analyze_btn = st.button("Analyze Now", type="primary", width="stretch")
        with col_b:
            clear_btn = st.button("🗑️ Clear", width="stretch")
            if clear_btn:
                st.session_state.current_text = ""
                st.rerun()
        
        if analyze_btn and user_input.strip():
            with st.spinner("🔍 Analyzing text..."):
                # Preprocessing steps visualization
                with st.expander("📝 View Preprocessing Steps", expanded=False):
                    col_p1, col_p2, col_p3 = st.columns(3)
                    with col_p1:
                        st.markdown("**Original:**")
                        st.code(user_input, language="text")
                    with col_p2:
                        st.markdown("**After Cleaning:**")
                        cleaned = clean_text(user_input)
                        st.code(cleaned, language="text")
                    with col_p3:
                        st.markdown("**After Stopword Removal:**")
                        st.code(" ".join([w for w in cleaned.split() if w not in roman_stopwords]), language="text")
                
                # Prediction
                cleaned = clean_text(user_input)
                input_vec = vectorizer.transform([cleaned])
                prediction = model.predict(input_vec)[0]
                
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_vec)[0]
                    confidence = proba[1] if prediction == 1 else proba[0]
                    
                    # Store in history
                    st.session_state.history.append({
                        'text': user_input,
                        'prediction': int(prediction),
                        'confidence': confidence,
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # Result display with animation
                    if prediction == 0:
                        st.markdown(f"""
                        <div class="result-card-safe">
                            <h2>✅ SAFE CONTENT</h2>
                            <p style="font-size: 3rem;">😊</p>
                            <p>This text appears to be normal and non-offensive.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Highlight hate words
                        highlighted_text = highlight_hate_words(user_input)
                        st.markdown(f"""
                        <div class="result-card-hate">
                            <h2>⚠️ HATE SPEECH DETECTED</h2>
                            <p style="font-size: 3rem;">🚫</p>
                            <p>This text contains potentially offensive content.</p>
                            <hr>
                            <p><strong>Flagged content:</strong><br>{highlighted_text}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confidence meter
                    if show_confidence:
                        st.markdown("### 📊 Confidence Score")
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=confidence * 100,
                            delta={'reference': 80},
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "#667eea"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "gray"},
                                    {'range': [80, 100], 'color': "darkgray"}],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 80}}))
                        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig, width="stretch")
                    
                    # Feedback section
                    st.markdown("---")
                    st.markdown("### Was this analysis correct?")
                    fb_col1, fb_col2 = st.columns(2)
                    with fb_col1:
                        if st.button("✅ Yes, correct", width="stretch"):
                            st.session_state.feedback_data.append({
                                'text': user_input,
                                'prediction': int(prediction),
                                'feedback': 'correct',
                                'timestamp': datetime.now().isoformat()
                            })
                            st.success("Thank you for your feedback!")
                    with fb_col2:
                        if st.button("❌ No, incorrect", width="stretch"):
                            st.session_state.feedback_data.append({
                                'text': user_input,
                                'prediction': int(prediction),
                                'feedback': 'incorrect',
                                'timestamp': datetime.now().isoformat()
                            })
                            st.info("Thanks for helping us improve! Please describe the issue:")
                            issue_desc = st.text_area("What was wrong?")
                            if issue_desc:
                                st.success("Feedback recorded!")
    
    with col2:
        st.subheader("Recent Activity")
        
        if st.session_state.history:
            for item in reversed(st.session_state.history[-5:]):
                icon = "✅" if item['prediction'] == 0 else "⚠️"
                color = "#84fab0" if item['prediction'] == 0 else "#f5576c"
                st.markdown(f"""
                <div style="background: {color}20; padding: 0.75rem; border-radius: 0.5rem; margin-bottom: 0.5rem;">
                    <small>{item['timestamp']}</small><br>
                    <strong>{icon} {item['text'][:50]}...</strong><br>
                    <small>Confidence: {item['confidence']:.1%}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No analyses yet. Try analyzing some text!")
        
        # Word Cloud for common terms
        st.subheader("Common Terms")
        if st.session_state.history:
            all_text = " ".join([h['text'] for h in st.session_state.history])
            wordcloud = WordCloud(width=400, height=200, background_color='white').generate(all_text)
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.caption("Analyze some text to generate word cloud")

# ============================================
# TAB 2: Batch Processing
# ============================================
with tab2:
    st.subheader("Batch Processing")
    st.markdown("Upload a CSV file or paste multiple comments for bulk analysis")
    
    upload_col, manual_col = st.columns(2)
    
    with upload_col:
        st.markdown("### Upload CSV File")
        st.caption("CSV should have a column named 'text'")
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            if 'text' in df_upload.columns:
                st.success(f"Loaded {len(df_upload)} comments")
                
                if st.button("Analyze All", type="primary"):
                    with st.spinner(f"Analyzing {len(df_upload)} comments..."):
                        results_batch = []
                        for idx, row in df_upload.iterrows():
                            cleaned = clean_text(str(row['text']))
                            input_vec = vectorizer.transform([cleaned])
                            pred = model.predict(input_vec)[0]
                            proba = model.predict_proba(input_vec)[0]
                            confidence = proba[1] if pred == 1 else proba[0]
                            results_batch.append({
                                'text': row['text'][:100],
                                'prediction': 'Hate Speech' if pred == 1 else 'Normal',
                                'confidence': confidence
                            })
                        
                        df_results = pd.DataFrame(results_batch)
                        st.markdown("### Batch Results")
                        
                        # Summary stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Comments", len(df_results))
                        with col2:
                            hate_count = (df_results['prediction'] == 'Hate Speech').sum()
                            st.metric("Hate Speech Detected", hate_count, 
                                     delta=f"{(hate_count/len(df_results))*100:.1f}%")
                        with col3:
                            avg_conf = df_results['confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_conf:.1%}")
                        
                        # Results table
                        st.dataframe(df_results, width="stretch")
                        
                        # Download results
                        csv = df_results.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="batch_analysis_results.csv",
                            mime="text/csv"
                        )
            else:
                st.error("CSV must contain a 'text' column")
    
    with manual_col:
        st.markdown("### Manual Batch Entry")
        st.caption("Enter one comment per line")
        batch_text = st.text_area(
            "Paste multiple comments:",
            placeholder="Comment 1\nComment 2\nComment 3",
            height=200
        )
        
        if st.button("📊 Analyze Batch"):
            if batch_text.strip():
                comments = [c.strip() for c in batch_text.split('\n') if c.strip()]
                with st.spinner(f"Analyzing {len(comments)} comments..."):
                    batch_results = []
                    for comment in comments:
                        cleaned = clean_text(comment)
                        input_vec = vectorizer.transform([cleaned])
                        pred = model.predict(input_vec)[0]
                        batch_results.append({
                            'text': comment[:50],
                            'result': "🚫 Hate" if pred == 1 else "✅ Safe"
                        })
                    
                    df_batch = pd.DataFrame(batch_results)
                    st.dataframe(df_batch, width="stretch")
                    
                    # Visual summary
                    hate_count = (df_batch['result'] == "🚫 Hate").sum()
                    fig = go.Figure(data=[go.Pie(
                        labels=['Safe', 'Hate Speech'],
                        values=[len(comments)-hate_count, hate_count],
                        marker_colors=['#84fab0', '#f5576c'],
                        hole=0.4
                    )])
                    fig.update_layout(title="Batch Analysis Summary", height=400)
                    st.plotly_chart(fig, width="stretch")
            else:
                st.warning("Please enter some comments")

# ============================================
# TAB 3: Chat Simulator
# ============================================
with tab3:
    st.subheader("Chat Message Simulator")
    st.markdown("Simulate a conversation and see real-time hate speech detection")
    
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    # Chat display area
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_messages:
            if msg['role'] == 'user':
                st.markdown(f"""
                <div class="chat-bubble-user">
                    <strong>You:</strong> {msg['text']}<br>
                    <small style="opacity:0.7;">{'🚫 Hate detected' if msg.get('is_hate') else '✅ Safe'}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-bubble-bot">
                    <strong>Bot:</strong> {msg['text']}
                </div>
                """, unsafe_allow_html=True)
    
    # Message input form - CORRECTED WITH st.form()
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            chat_input = st.text_input(
                "Type your message:", 
                key="chat_input", 
                placeholder="Type in Roman Urdu and press Enter..."
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Align button with input
            submit_button = st.form_submit_button("Send", type="primary", width="stretch")
        
        if submit_button and chat_input.strip():
            # Analyze message
            cleaned = clean_text(chat_input)
            input_vec = vectorizer.transform([cleaned])
            prediction = model.predict(input_vec)[0]
            
            # Add user message
            st.session_state.chat_messages.append({
                'role': 'user',
                'text': chat_input,
                'is_hate': prediction == 1
            })
            
            # Generate bot response
            if prediction == 1:
                bot_response = "⚠️ I detected hate speech in your message."
            else:
                bot_response = "👍 Your message looks good!"
            
            st.session_state.chat_messages.append({
                'role': 'bot',
                'text': bot_response
            })
            
            st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_messages = []
        st.rerun()

# ============================================
# TAB 4: Analytics Dashboard
# ============================================
with tab4:
    st.subheader("Analytics Dashboard")
    
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history)
        
        # Row 1: Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Analyses", len(df_history))
        with col2:
            hate_rate = (df_history['prediction'].sum() / len(df_history)) * 100
            st.metric("Hate Speech Rate", f"{hate_rate:.1f}%")
        with col3:
            avg_conf = df_history['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        with col4:
            st.metric("Unique Users", "Session Only")
        
        # Row 2: Trend chart
        st.markdown("### Analysis Trend")
        df_history['timestamp_parsed'] = pd.to_datetime(df_history['timestamp'], format='%H:%M:%S')
        fig = px.line(df_history, x='timestamp_parsed', y='confidence', 
                      color='prediction', title="Confidence Over Time",
                      labels={'confidence': 'Confidence', 'timestamp_parsed': 'Time'})
        st.plotly_chart(fig, width="stretch")
        
        # Row 3: Distribution
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(df_history, names='prediction', title="Prediction Distribution",
                        color='prediction', color_discrete_map={0: '#84fab0', 1: '#f5576c'})
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            fig = px.histogram(df_history, x='confidence', nbins=20, 
                              title="Confidence Distribution",
                              color_discrete_sequence=['#667eea'])
            fig.update_layout(bargap=0.05)
            st.plotly_chart(fig, width="stretch")
        
        # Feedback summary
        if st.session_state.feedback_data:
            st.markdown("### 📝 User Feedback Summary")
            df_feedback = pd.DataFrame(st.session_state.feedback_data)
            feedback_counts = df_feedback['feedback'].value_counts()
            fig = px.bar(feedback_counts, x=feedback_counts.index, y='count',
                        title="Feedback Distribution", color=feedback_counts.index,
                        color_discrete_map={'correct': '#84fab0', 'incorrect': '#f5576c'})
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("No data yet. Start analyzing comments to see analytics!")

# ============================================
# TAB 5: About & Guide
# ============================================
with tab5:
    st.subheader("About & User Guide")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Purpose
        This tool helps identify hate speech in **Roman Urdu** (Urdu written in English script) 
        to promote safer online discussions.
        
        ### How It Works
        1. **Text Preprocessing**: Cleans and normalizes Roman Urdu text
        2. **Feature Extraction**: Converts text to TF-IDF features with n-grams
        3. **Classification**: ML model predicts if content contains hate speech
        
        ### Understanding Results
        
        | Result | Meaning | Action |
        |--------|---------|--------|
        | Safe | Normal content | No action needed |
        | Hate Speech | Potentially offensive | Review before posting |
        
        ### Features
        - **Real-time Analysis**: Instant detection as you type
        - **Batch Processing**: Analyze multiple comments at once
        - **Chat Simulator**: Test in conversation context
        - **Analytics Dashboard**: Track your analysis history
        
        ### Roman Urdu Examples
        
        **Normal Text:**
        - "Aaj ka din bohot acha hai"
        - "Mujhe ye movie pasand aayi"
        
        **Hate Speech Examples:**
        - "Tum bohot bure ho"
        - "Tumhari soch kharab hai"
        """)
    
    with col2:
        st.markdown("""
        ### Tips for Best Results
        
        1. **Write naturally** - The model works best with natural Roman Urdu
        2. **Avoid excessive slang** - Uncommon words may be misclassified
        3. **Check confidence scores** - Low confidence means ambiguous text
        4. **Provide feedback** - Help improve the model
        
        ### Need Help?
        
        - Check the preprocessing steps in analysis view
        - Review model performance metrics in sidebar
        - Use batch processing for bulk analysis
        
        ### Model Information
        
        - **Training Data**: Labeled Roman Urdu comments
        - **Selected Model**: **{}** 
        - **Selected Model F1**: {:.4f}
        - **Best Available F1**: {:.4f} ({})
        - **Selection Criteria**: {}
        """.format(
            selection_metadata['selected_model'],
            selection_metadata['selected_f1'],
            results_df['F1-Score'].max(),
            results_df.loc[results_df['F1-Score'].idxmax(), 'Model'],
            selection_metadata.get('selection_criteria', 'F1 within 3% threshold, then fastest')
        ))

# ============================
# Footer
# ============================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.caption("🔒 Your data stays in your browser")
with footer_col2:
    st.caption("🤖 Powered by Machine Learning")
with footer_col3:
    st.caption("v2.0 - Enhanced Edition")

# Export feedback data option
if st.session_state.feedback_data and st.button("📤 Export Feedback Data"):
    df_feedback_export = pd.DataFrame(st.session_state.feedback_data)
    csv_feedback = df_feedback_export.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Feedback CSV",
        data=csv_feedback,
        file_name="model_feedback.csv",
        mime="text/csv"
    )