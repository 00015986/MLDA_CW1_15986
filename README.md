## Student Dropout / Academic Success Prediction
```
Python: 3.11.10
Conda: 24.11.3
```
</b>

### Project Description
```
This project predicts a student's academic outcome using machine learning.
The user can enter specific feature values, and the model will predict whether the student is likely to:

Drop out
Stay Enrolled
Graduate Successfully

Two ML models were created:

Random Forest requires 18 input features
K-Nearest Neighbors KNN requires 36 input features 

The final accuracy ranges between 70% and 85% depending on the model.
The app also shows the probability distribution for each class to help understand model confidence.
```
</b>

### Streamlit App link
```
The Streamlit UI is hosted on Streamlit Community Cloud:
```
link: <code>https://mlda-cw1-15986-predict-student-dropout-academic-success.streamlit.app</code>
</b>

### Prerequisites
To run this project locally, install:

conda <code>24.11.3</code>
Python <code>3.11.10</code>
jupyter <code>1.1.1</code>

### How to download and run the code locally
```
If you want to clone and run the project locally, follow these steps:
```
<code>git clone https://github.com/00015986/MLDA_CW1_15986.git</code>
<code>cd MLDA_CW1_15986</code>

#### To install packages
<code>pip install -r requirements.txt</code>

#### After installing depandensis you can run application code
<code>streamlit run ui_part/Predict_Dropout_Success_ui_app.py</code>
</b>

### Folder Structure
```
MLDA_CW1_15986/
│
├── project paper/
│     └── MLDA_CW1_15986.docx               # written project paper
│
├── src/
│     ├── dataset/                           # dataset used for training
│     ├── plots/                             # diagrams, visualizations
│     ├── saved_features/                    # selected features for RF/KNN
│     └── saved_models/                      # trained ML models
│
├── ui_part/
│     └── Predict_Dropout_Success_ui_app.py  # Streamlit UI app
│
├── README.md
└── requirements.txt
```
</b>

## Used Machine Learning Algorithms
<ul>
  <li>Decision Tree</li>
  <li>Random Forest</li>
  <li>KNN</li>
</ul>
</b>

## Model Evaluation Metrics
<ul>
  <li>Accuracy</li>
  <li>Precision (macro)</li>
  <li>Recall (macro)</li>
  <li>F1-score (macro)</li>
</ul>
</b>

## Hyperparameter Tuning
To improve model performance, <code>GridSearchCV</code> was used for Random Forest and KNN models.
