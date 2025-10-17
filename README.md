# <h1 align="center">📨 SpamGuard AI</h1>

# <p align="center">

# &nbsp; <b>Smart SMS \& Email Spam Detection using SVM + TF-IDF</b><br>

# &nbsp; Built with ❤️ using Streamlit, Scikit-learn \& NLP

# </p>

# 

# ---

# 

# \## 🚀 Overview

# 

# \*\*SpamGuard AI\*\* is a Machine Learning web app that detects spam SMS or emails using \*\*Support Vector Machine (SVM)\*\* and \*\*TF-IDF vectorization\*\*.  

# It features a modern Streamlit interface with beautiful analytics visualizations.

# 

# ---

# 

# \## ✨ Features

# \- 🔍 Real-time spam detection for SMS \& emails  

# \- 📁 Bulk prediction via CSV upload  

# \- 📊 Interactive insights: Confusion Matrix, Word Clouds, Top Word Charts  

# \- 🧠 Machine Learning model: SVM + TF-IDF  

# \- 🧹 Text preprocessing with NLTK  

# \- 🖥️ Elegant Streamlit-based UI  

# 

# ---

# 

# \## ⚙️ Tech Stack

# 

# | Component | Technology |

# |------------|-------------|

# | 💻 Frontend UI | Streamlit |

# | 🧠 Machine Learning | Scikit-learn (LinearSVC) |

# | 🧹 Text Cleaning | NLTK |

# | 📊 Visualization | Matplotlib, Seaborn, WordCloud |

# | 💾 Model Storage | Joblib |

# | 🧱 Language | Python 3 |

# 

# ---

# 

# \## 🧩 Project Structure

# 📦 svm/

# ┣ 📜 app.py # Streamlit web app

# ┣ 📜 train\_model.py # Model training pipeline

# ┣ 📂 data/ # Dataset folder

# ┣ 📂 model/ # Trained model (pipeline.joblib)

# ┣ 📜 requirements.txt # Dependencies

# ┗ 📜 README.md # Documentation

# 

# yaml

# Copy code

# 

# ---

# 

# \## 🧠 Dataset

# Dataset: \[UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)  

# Contains 5572 labeled SMS messages:  

# \- \*\*ham (normal)\*\* messages  

# \- \*\*spam\*\* messages  

# 

# ---

# 

# \## ▶️ Run Locally

# 

# ```bash

# \# Step 1: Clone this repo

# git clone https://github.com/vkchavan/SpamGuard-AI.git

# 

# \# Step 2: Go inside folder

# cd SpamGuard-AI

# 

# \# Step 3: Install dependencies

# pip install -r requirements.txt

# 

# \# Step 4: Train model

# python train\_model.py

# 

# \# Step 5: Run app

# streamlit run app.py

# 📊 Insights \& Visualizations

# 🎯 Model Accuracy Metric

# 

# 🧮 Confusion Matrix Heatmap

# 

# ☁️ Word Clouds for Spam \& Ham

# 

# 🔤 Top Frequent Spam Words Chart

# 

# 🧾 Dataset Summary Dashboard

# 

# 👩‍💻 Developers — MVJ Solutions

# Name	Email	GitHub

# Vaishnavi Chavan	vaishnavichavan1712@gmail.com	github.com/vkchavan

# Junaid Shaikh	junaidshaikh1311@gmail.com	github.com/junaid3234

# 

# 🌟 Screenshots

# Home Page	Prediction Page

# 

# Insights Dashboard	Bulk Upload

# 

# 🏷️ License

# This project is open-source under the MIT License.

# 

# <p align="center">💬 Built with ❤️ by <b>MVJ Solutions</b></p> ```

