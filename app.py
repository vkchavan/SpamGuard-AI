# app.py — SpamGuard AI (Animated Landing + About + Tech Stack)
import streamlit as st
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, accuracy_score
from train_model import load_data, clean_text, train_and_save, DATA_PATH, MODEL_PATH

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="SpamGuard AI", page_icon="📨", layout="wide")

# ---------- CACHED MODEL ----------
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

# ---------- SESSION STATE ----------
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to_app():
    st.session_state.page = "app"
    st.rerun()

def go_to_home():
    st.session_state.page = "home"
    st.rerun()

# ---------- CSS STYLING ----------
st.markdown("""
<style>
@keyframes gradientShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
.stApp {
  background: linear-gradient(-45deg, #2563eb, #7c3aed, #06b6d4, #3b82f6);
  background-size: 400% 400%;
  animation: gradientShift 12s ease infinite;
  color: white;
}
.hero-container {
  text-align: center;
  padding: 140px 20px 80px;
}
.hero-title {
  font-size: 4rem;
  font-weight: 800;
  color: white;
  text-shadow: 2px 2px 10px rgba(0,0,0,0.3);
  animation: fadeDown 1.5s ease-out;
}
.hero-subtitle {
  font-size: 1.3rem;
  color: #e2e8f0;
  margin-top: 0.5rem;
  animation: fadeUp 2s ease-out;
}
@keyframes fadeDown {
  from { opacity: 0; transform: translateY(-40px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(40px); }
  to { opacity: 1; transform: translateY(0); }
}
.start-btn {
  margin-top: 2.5rem;
  background-color: white;
  color: #1e3a8a;
  font-weight: 700;
  font-size: 1.2rem;
  border: none;
  border-radius: 12px;
  padding: 0.8em 2.5em;
  box-shadow: 0px 4px 20px rgba(255,255,255,0.2);
  transition: all 0.3s ease;
  cursor: pointer;
}
.start-btn:hover {
  background-color: #f1f5f9;
  box-shadow: 0px 8px 25px rgba(255,255,255,0.3);
  transform: scale(1.05);
}
.footer {
  text-align: center;
  margin-top: 100px;
  color: #e2e8f0;
  font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- LANDING PAGE ----------
if st.session_state.page == "home":
    st.markdown("""
    <div class="hero-container">
      <h1 class="hero-title">📨 SpamGuard AI</h1>
      <p class="hero-subtitle">
        Protect your inbox from spam — detect suspicious SMS & emails instantly.<br>
        Powered by Support Vector Machines (SVM) + TF-IDF, crafted with Streamlit.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Detecting Spam", use_container_width=True):
            go_to_app()

    st.markdown("""
    <div class="footer">
      © 2025 MVJ Solutions — Built with ❤️ using Streamlit
    </div>
    """, unsafe_allow_html=True)

# ---------- APP PAGE ----------
elif st.session_state.page == "app":
    st.sidebar.title("⚙️ Controls")
    if st.sidebar.button("🏠 Back to Home"):
        go_to_home()

    if st.sidebar.button("🔁 Retrain Model"):
        with st.spinner("Training model..."):
            train_and_save()
        st.sidebar.success("✅ Model retrained successfully!")

    model = load_model()
    if os.path.exists(MODEL_PATH):
        st.sidebar.success("Model loaded ✅")
    else:
        st.sidebar.warning("Train the model first.")

    st.title("📨 SpamGuard AI — Dashboard")
    st.caption("Smart SMS & Email Spam Detection using SVM + TF-IDF")

    tabs = st.tabs(["🔍 Predict", "📁 Bulk Upload", "📊 Insights", "ℹ️ About & Developers"])

    # ---- Predict Tab ----
    with tabs[0]:
        st.header("🔍 Single Message Prediction")
        text = st.text_area("Enter or paste a message:", height=150)
        if st.button("Predict"):
            if not model:
                st.error("Model not loaded. Please train first.")
            elif not text.strip():
                st.warning("Please enter a message.")
            else:
                pred = model.predict([clean_text(text)])[0]
                if pred == "spam":
                    st.error("🚫 SPAM Detected")
                else:
                    st.success("✅ HAM (Not Spam)")

    # ---- Bulk Tab ----
    with tabs[1]:
        st.header("📁 Bulk CSV Prediction")
        uploaded = st.file_uploader("Upload a CSV with a column named `message`", type=['csv'])
        if uploaded:
            df = pd.read_csv(uploaded, encoding='latin1', header=0, on_bad_lines='warn')

            if 'message' not in df.columns:
                st.error("CSV must include a column `message`.")
            else:
                with st.spinner("Predicting..."):
                    df['message_clean'] = df['message'].astype(str).apply(clean_text)
                    df['prediction'] = model.predict(df['message_clean'])
                st.success("✅ Predictions complete.")
                st.dataframe(df.head(10))
                st.download_button("⬇️ Download Results", 
                                   df.to_csv(index=False).encode('utf-8'),
                                   "spam_predictions.csv", "text/csv")

    # ---- Insights Tab ----
        # ---- Insights Tab ----
        # ---- Insights Tab ----
      # ---- Insights Tab ----
    with tabs[2]:
        st.header("📊 Model Insights & Analytics")

        if model and os.path.exists(DATA_PATH):
            df = load_data()
            df['message_clean'] = df['message'].apply(clean_text)
            y_true = df['label']
            y_pred = model.predict(df['message_clean'])
            acc = accuracy_score(y_true, y_pred)

            # 🧠 Data Summary Metrics
            total_msgs = len(df)
            spam_count = len(df[df['label'] == 'spam'])
            ham_count = len(df[df['label'] == 'ham'])
            spam_ratio = (spam_count / total_msgs) * 100
            avg_length = df['message'].apply(lambda x: len(str(x))).mean()

            colA, colB, colC, colD = st.columns(4)
            colA.metric("📦 Total Messages", f"{total_msgs}")
            colB.metric("🚫 Spam Messages", f"{spam_count}")
            colC.metric("✅ Ham Messages", f"{ham_count}")
            colD.metric("📊 Spam Ratio", f"{spam_ratio:.1f}%")

            st.markdown("---")
            st.metric("🎯 Model Accuracy", f"{acc*100:.2f}%")

            # 1️⃣ Confusion Matrix Heatmap
            cm = confusion_matrix(y_true, y_pred, labels=['ham', 'spam'])
            fig, ax = plt.subplots()
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=ax
            )
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

            st.markdown("---")

            # 2️⃣ Word Clouds (Spam vs Ham)
            st.subheader("☁️ Word Clouds — Spam vs Ham Messages")
            col1, col2 = st.columns(2)
            spam_text = " ".join(df[df['label'] == 'spam']['message_clean'])
            ham_text = " ".join(df[df['label'] == 'ham']['message_clean'])

            with col1:
                st.markdown("**🚫 Spam Messages**")
                wc_spam = WordCloud(
                    width=600, height=300, background_color='white', colormap='Reds'
                ).generate(spam_text)
                fig1, ax1 = plt.subplots()
                ax1.imshow(wc_spam, interpolation='bilinear')
                ax1.axis('off')
                st.pyplot(fig1)

            with col2:
                st.markdown("**✅ Ham Messages**")
                wc_ham = WordCloud(
                    width=600, height=300, background_color='white', colormap='Greens'
                ).generate(ham_text)
                fig2, ax2 = plt.subplots()
                ax2.imshow(wc_ham, interpolation='bilinear')
                ax2.axis('off')
                st.pyplot(fig2)

            st.markdown("---")

            # 3️⃣ Bar Chart - Top 10 Frequent Spam Words
            st.subheader("🔤 Top 10 Most Common Words in Spam Messages")
            from collections import Counter
            import re

            words = re.findall(r'\b\w+\b', spam_text)
            word_freq = Counter(words)
            top_words = pd.DataFrame(word_freq.most_common(10), columns=['Word', 'Frequency'])

            fig_bar, ax_bar = plt.subplots()
            sns.barplot(data=top_words, x='Frequency', y='Word', palette='Reds_r', ax=ax_bar)
            ax_bar.set_title("Most Frequent Words in Spam Messages")
            st.pyplot(fig_bar)

        else:
            st.info("⚠️ Train the model to view insights.")




    # ---- About & Developers Tab ----
    with tabs[3]:
        st.header("ℹ️ About SpamGuard AI")
        st.markdown("""
        **SpamGuard AI** is a Machine Learning-powered web application that classifies SMS or Email messages as **Spam** or **Ham** (Not Spam).  
        It uses Natural Language Processing (NLP) techniques combined with **Support Vector Machine (SVM)** classification for high accuracy.  

        ### 🧠 How It Works:
        1. The app cleans and tokenizes text messages using NLTK.
        2. It converts text into numerical features using **TF-IDF Vectorization**.
        3. The **SVM model** learns to classify spam vs. ham messages.
        4. The trained pipeline is saved using Joblib for instant predictions.

        ---
        ### ⚙️ Tech Stack:
        | Component | Technology |
        |------------|-------------|
        | Frontend UI | Streamlit |
        | Machine Learning | scikit-learn (LinearSVC) |
        | Text Processing | NLTK |
        | Data Visualization | Matplotlib, Seaborn, WordCloud |
        | Dataset | SMS Spam Collection (UCI ML Repository) |
        | Programming Language | Python 3 |

        ---
        ### 👩‍💻 Developers — MVJ Solutions
        | Name | Email | GitHub |
        |------|--------|--------|
        | **Junaid Shaikh** | [junaidshaikh1311@gmail.com](mailto:junaidshaikh1311@gmail.com)| [github.com/junaid3234](https://github.com/junaid3234) |
        | **Vaishnavi Chavan** | [vaishnavichavan1712@gmail.com](mailto:vaishnavichavan1712@gmail.com)) | [github.com/vkchavan](https://github.com/vkchavan) |
        """)
