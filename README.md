<h1 align="center">📨 SpamGuard AI</h1>
<p align="center">
  <b>Smart SMS & Email Spam Detection using SVM + TF-IDF</b><br>
  Built with ❤️ using Streamlit, Scikit-learn & NLP
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" /></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-App-red.svg" /></a>
  <a href="https://scikit-learn.org/stable/"><img src="https://img.shields.io/badge/Scikit--learn-ML-orange.svg" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" /></a>
</p>

---

## 🚀 Overview

**SpamGuard AI** is an intelligent machine-learning web app that detects **spam SMS and emails** in real time.  
It leverages **Support Vector Machine (SVM)** and **TF-IDF Vectorization** for high-accuracy classification of messages as `spam` or `ham`.

---

## ✨ Features

✅ Real-time spam detection for single messages  
✅ Bulk message prediction via CSV upload  
✅ Interactive analytics (confusion matrix, word clouds, top-word plots)  
✅ NLTK-based text preprocessing (stopword removal, tokenization, etc.)  
✅ Persistent model storage with Joblib  
✅ Modern UI powered by Streamlit  

---

## 🧠 Tech Stack

| Component | Technology |
|------------|-------------|
| Frontend / UI | Streamlit |
| ML Model | Scikit-learn (SVM Classifier) |
| NLP Processing | NLTK, TF-IDF |
| Visualization | Matplotlib / Seaborn / WordCloud |
| Model Storage | Joblib |
| Language | Python 3.9 + |

---

## 📂 Project Structure


SpamGuard-AI/

├── app.py # Streamlit web app
├── train_model.py # Model training script
├── data/ # Dataset folder (SMS spam collection)
├── model/ # Saved pipeline model.joblib
├── requirements.txt # Dependencies
└── README.md # Documentation


---

## 🧾 Dataset

Uses the **UCI SMS Spam Collection Dataset**  
Total messages = **5,572**  
- `ham` → legitimate messages  
- `spam` → unsolicited messages  

---

## ⚙️ Installation & Usage

```bash
# 1️⃣ Clone the repo
git clone https://github.com/vkchavan/SpamGuard-AI.git
cd SpamGuard-AI

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ (Optional) Train model
python train_model.py

# 4️⃣ Run the Streamlit app
streamlit run app.py


Open the local URL shown in the terminal (e.g. http://localhost:8501) and explore the app.


🪪 License

Distributed under the MIT License.
See the LICENSE
file for details.