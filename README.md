# Heart Attack Risk Prediction Using Machine Learning

## Project Overview

This project aims to predict the **risk of heart attacks** using machine learning techniques based on both clinical and non-clinical patient data. 
The dataset used contains over 9,000 patient records (reduced to ~5,000 after cleaning), with 25 input features and a binary target variable indicating heart attack risk ('0 = No Risk', '1 = Risk').

Four supervised learning models—**Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Decision Tree, and Random Forest**—were trained and evaluated. The Random Forest model showed the best overall performance, achieving an accuracy of **66%**.

---

## Dataset

- **Source:** GitHub
- **Initial Size:** ~9,000 records
- **Final Size:** ~5,000 records (after cleaning and removing invalid values)
- **Target Variable:** 'Heart Attack Risk' (0 or 1)
- **Key Features Used:**
  - Age
  - Blood Pressure (Systolic/Diastolic)
  - Cholesterol
  - Smoking
  - Obesity
  - Physical Activity
  - Stress Level
  - Previous Heart Problems
  - Diet, Gender (encoded)

---

## Project Pipeline

### 1. Data Preprocessing
- Converted complex fields like `Blood Pressure` (e.g., 120/80) into `BP_systolic` and `BP_diastolic`.
- Used **LabelEncoder** to convert categorical fields (e.g., Gender, Diet) into numerical form.
- Removed entries with missing or biologically unrealistic values.

### 2. Exploratory Data Analysis (EDA)
- Generated histograms and calculated descriptive statistics.
- Identified key features through **correlation heatmap** analysis.

### 3. Handling Class Imbalance
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the target variable and improve generalization.

### 4. Modeling
- Trained and tested four models: SVM, KNN, Decision Tree, and Random Forest.
- Used **GridSearchCV** for hyperparameter tuning and **10-fold cross-validation**.

### 5. Evaluation
- Evaluated models using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - AUC-ROC Curve
- Visualized performance with confusion matrix and performance table.

---

## Results & Discussion

Among all the models tested, the **Random Forest** algorithm achieved the best performance, with:
- **Accuracy:** 66%
- **Balanced performance** across precision, recall, and F1 score
- Strong identification of important features contributing to risk

Although 66% may seem modest, the result is promising given the limited and cleaned dataset. The project demonstrates that with more data and further tuning, these models could be used for real-world risk prediction.

---

## How to Run This Project

### 1. Clone the Repository
'''bash
git clone https://github.com/LaxmiSowjanya/heart-attack-prediction.git
cd heart-attack-prediction

### 2. Setup the environment with required python libraries (refer 'Dependencies' in the below)

### 3. Run it in your environment 
'''Jupyter Notebook
jupyter notebook heart_attack_prediction.ipynb

---


### Dependencies

Python 3.9+

pandas

numpy

matplotlib

seaborn

scikit-learn

imbalanced-learn

jupyter (optional)

---
