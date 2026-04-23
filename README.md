
# Employee Attrition Prediction System

## Overview

This project focuses on analyzing employee data and predicting attrition using machine learning techniques. The goal is to help organizations identify employees who are at risk of leaving and take proactive steps to improve retention.

The project covers the complete workflow — from data analysis and preprocessing to model building and deployment using a Streamlit web application.

---

## Key Features

* Exploratory Data Analysis (EDA) to understand employee behavior
* Data preprocessing and feature engineering using pipelines
* Handling class imbalance using SMOTE
* Training and comparing multiple machine learning models
* Model evaluation using standard performance metrics
* Deployment using Streamlit for real-time predictions

---

## Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* XGBoost
* Imbalanced-learn (SMOTE)
* Streamlit

---

## Project Structure

```
Employee-Attrition-ML/
│
├── employee_attrition_spyder.py      # Model training and analysis
├── streamlit_attrition_app.py        # Streamlit web application
├── best_attrition_model.pkl          # Trained model
├── preprocessor.pkl                  # Data preprocessing pipeline
├── HR_comma_sep.csv                  # Dataset
├── requirements.txt                  # Dependencies
└── README.md                         # Project documentation
```

---

## How to Run the Project

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Train the model

```
python employee_attrition_spyder.py
```

### 3. Run the Streamlit app

```
streamlit run streamlit_attrition_app.py
```

---

## Model Performance

Multiple models were trained and evaluated, including Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Gradient Boosting, and XGBoost.

The best-performing model was selected based on ROC-AUC score and overall performance on the test dataset.

---

## Application

The Streamlit application allows users to:

* Input employee details
* Predict the likelihood of attrition
* View probability scores
* Get basic HR recommendations based on prediction

---

## Insights

Some key observations from the analysis:

* Employees with low satisfaction levels are more likely to leave
* High working hours are associated with increased attrition
* Lack of promotion impacts employee retention
* Salary level plays a significant role in employee decisions

---

## Future Improvements

* Hyperparameter tuning for improved model performance
* Integration with real-time HR systems
* Deployment on cloud platforms
* Enhanced UI with interactive visualizations

---

## Author

Priya Kumari
B.Tech CSE (Data Science)
Lovely Professional University

