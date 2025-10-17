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

# It provides a beautiful and interactive UI built with \*\*Streamlit\*\*, including visual insights and analytics for model performance.

# 

# ---

# 

# \## ✨ Features

# \- 🔍 Real-time spam detection for SMS \& emails  

# \- 📁 Bulk prediction from CSV files  

# \- 📊 Interactive insights (Confusion Matrix, Word Clouds, Top Words Chart)  

# \- 🧠 Machine Learning model: SVM + TF-IDF  

# \- 🧹 Text cleaning using NLTK preprocessing  

# \- 🖥️ Simple, modern UI built in Streamlit  

# 

# ---

# 

# \## ⚙️ Tech Stack

# 

# | Component | Technology |

# |------------|-------------|

# | 💻 Frontend UI | Streamlit |

# | 🧠 Machine Learning | scikit-learn (LinearSVC) |

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

# \## ▶️ How to Run Locally

# 

# ```bash

# \# Step 1: Clone this repo

# git clone https://github.com/vkchavan/SpamGuard-AI.git

# 

# \# Step 2: Navigate into project folder

# cd SpamGuard-AI

# 

# \# Step 3: Install dependencies

# pip install -r requirements.txt

# 

# \# Step 4: Train the model

# python train\_model.py

# 

# \# Step 5: Run the web app

# streamlit run app.py

# 📊 Data Visualization Highlights

# 🎯 Confusion Matrix (Model Accuracy)

# 

# ☁️ Spam vs Ham Word Clouds

# 

# 🔤 Top Frequent Spam Words Bar Chart

# 

# 📈 Dataset Summary Dashboard

# 

# 🧑‍💻 Developers — MVJ Solutions

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

# This project is open source under the MIT License.

# 

# <p align="center">💬 Built with ❤️ by MVJ Solutions</p> ```

