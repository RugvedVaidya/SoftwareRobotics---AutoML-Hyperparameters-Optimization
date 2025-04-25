# SoftwareRobotics---AutoML-Hyperparameters-Optimization

# 🧠 AutoML System for Hyperparameter Optimization

An AI-powered AutoML system that simplifies the machine learning pipeline by automating data preprocessing, model selection, and hyperparameter optimization. Just upload your dataset, choose your target column, and let the system do the rest — from cleaning your data to evaluating multiple models and showing the best results.

## 🚀 Features

- 📁 **Dataset Upload**: Accepts any CSV dataset
- 🎯 **Target Column Selection**: User selects the column to predict
- 🧹 **Smart Data Cleaning**: Automatically handles missing values, encodes categorical features, and scales data when needed
- 🤖 **Model Training**: Supports multiple ML models:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - XGBoost
  - LightGBM
- 🛠️ **Hyperparameter Optimization**: Performs Grid Search or Random Search for best hyperparameters
- 📊 **Model Comparison**: Compares all models and displays accuracy (rounded to 2 decimal places)
- 🌐 **Web Interface**: Easy-to-use Flask-based UI for interaction

---

## 📦 Installation

1. **Clone the repository**

git clone https://github.com/yourusername/automl-hyperparameter-optimization.git
cd automl-hyperparameter-optimization

2.**Create and activate a virtual environment**
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3.**Install dependencies**

4.**Start the flask app:**
python app.py

5.**Open your broswer and visit:**
http://127.0.0.1:5000

6.**Steps to follow on the web interface:**
Upload your dataset (CSV file)
Select the target column
Click on “Run AutoML”
View best performing model and accuracy

## 📂 Project Structure
