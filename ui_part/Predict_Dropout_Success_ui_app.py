
import streamlit as st
import pandas as pd
import joblib
import os

# getting the repo root by going 2 level up
current_file = os.path.abspath(__file__)
repo_root = os.path.dirname(os.path.dirname(current_file))

# Building path to the models and selected features' files
model_path_rf = os.path.join(repo_root, "src", "saved_models", "random_forest.pkl")
model_path_knn = os.path.join(repo_root, "src", "saved_models", "knn_model.pkl")
model_path_scaler = os.path.join(repo_root, "src", "saved_models", "knn_scaler.pkl")

sf_path_rf = os.path.join(repo_root, "src", "saved_features", "rf_selected_features.pkl")
sf_path_knn = os.path.join(repo_root, "src", "saved_features", "knn_selected_features.pkl")

# Page titles and layouts
st.set_page_config(page_title="Student Outcome Predictor", layout="centered")
st.title("Student Dropout / Success Predictor (RF | KNN)")

# Class labeling
label_map = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}

# Loading the trained models and scaler
# Random Forest model
rf_model = joblib.load(model_path_rf)

# KNN model
knn_model = joblib.load(model_path_knn)

# Scaler needed for KNN because KNN uses scaled data
scaler = joblib.load(model_path_scaler)

# Selected features used by Random Forest and KNN
sf_rf = joblib.load(sf_path_rf)
sf_knn = joblib.load(sf_path_knn)

# Dictionaries that convert human text to numeric ML values
# These make the UI friendly so users do not enter numbers
GENDER = {"Male": 1, "Female": 0}

MARITAL = {
    "Single": 1, "Married": 2, "Widowed": 3,
    "Divorced": 4, "Union": 5, "Legally Separated": 6
}

NATIONALITY = {
    "Portuguese": 1, "German": 2, "Spanish": 6, "Italian": 11,
    "Dutch": 13, "English": 14, "Lithuanian": 17, "Angolan": 21,
    "Cape Verdean": 22, "Guinean": 24, "Mozambican": 25,
    "Santomean": 26, "Turkish": 32, "Brazilian": 41,
    "Romanian": 62, "Moldovan": 100, "Mexican": 101,
    "Ukrainian": 103, "Russian": 105, "Cuban": 108, "Colombian": 109
}

COURSES = {
    "Biofuel Production Technologies": 33,
    "Animation & Multimedia Design": 171,
    "Social Service (Evening)": 8014,
    "Agronomy": 9003,
    "Communication Design": 9070,
    "Veterinary Nursing": 9085,
    "Informatics Engineering": 9119,
    "Equinculture": 9130,
    "Management": 9147,
    "Social Service": 9238,
    "Tourism": 9254,
    "Nursing": 9500,
    "Oral Hygiene": 9556,
    "Marketing Management": 9670,
    "Journalism & Communication": 9773,
    "Basic Education": 9853,
    "Management (Evening)": 9991
}

APPLICATION_MODE = {
    "1st Phase": 1,
    "Other Higher Courses": 7,
    "International Student": 15,
    "1st Phase (Madeira)": 16,
    "2nd Phase": 17,
    "3rd Phase": 18,
    "Change of Course": 43
}

APPLICATION_ORDER = {f"Choice {i}": i for i in [0,1,2,3,4,5,6,9]}

DAYTIME = {"Daytime": 1, "Evening": 0}

PREV_QUAL = {
    "Secondary Education": 1,
    "Bachelor Degree": 2,
    "Master Degree": 4,
    "Basic Education (3rd Cycle)": 19,
    "12th Year Not Completed": 9
}

# Parent qualifications
MOTHER_QUAL = {"Secondary Education": 1, "Master Degree": 4, "Doctorate": 5}
FATHER_QUAL = {"Secondary Education": 1, "Master Degree": 4, "Doctorate": 5}

# Parent occupations
MOTHER_OCC = {"Teacher": 123, "Health Worker": 122, "Admin": 4, "Unskilled Worker": 9}
FATHER_OCC = {"Teacher": 123, "Engineer": 121, "Technician": 135, "Service Worker": 151}

GDP = {"GDP 0": 0, "GDP 1": 1, "GDP 2": 2, "GDP 3": 3, "GDP 4": 4, "GDP 5": 5}
UNEMP = {"0%": 0, "1%": 1.0, "2%": 2.0, "3%": 3.0, "4%": 4.0, "5%": 5.0, "6%": 6.0, "7%": 7.0, "8%": 8.0, "9%": 9.0, "10%": 10.0}
INFLATION = {"0.3%": 0.3, "0.5%": 0.5, "2.8%": 2.8}

# Some long complicated feature names have friendlier names here
DISPLAY_NAME = {
    "Curricular units 1st sem (approved)": "1st Sem – Units Approved",
    "Curricular units 1st sem (grade)": "1st Sem – Avg Grade",
    "Curricular units 1st sem (evaluations)": "1st Sem – Evaluations",
    "Curricular units 1st sem (enrolled)": "1st Sem – Enrolled Units",
    "Curricular units 1st sem (credited)": "1st Sem – Credited Units",
    "Curricular units 1st sem (without evaluations)": "1st Sem – No Eval Units",

    "Curricular units 2nd sem (approved)": "2nd Sem – Units Approved",
    "Curricular units 2nd sem (grade)": "2nd Sem – Avg Grade",
    "Curricular units 2nd sem (evaluations)": "2nd Sem – Evaluations",
    "Curricular units 2nd sem (enrolled)": "2nd Sem – Enrolled Units",
    "Curricular units 2nd sem (credited)": "2nd Sem – Credited Units",
    "Curricular units 2nd sem (without evaluations)": "2nd Sem – No Eval Units",
}

# Function that builds the correct input widget
def make_widget(feature):

    label = DISPLAY_NAME.get(feature, feature)

    # Categorical features drop-down selection
    if feature == "Gender":
        return GENDER[st.selectbox(label, list(GENDER))]

    if feature == "Marital status":
        return MARITAL[st.selectbox(label, list(MARITAL))]

    if feature == "Nacionality":
        return NATIONALITY[st.selectbox(label, list(NATIONALITY))]

    if feature == "Course":
        return COURSES[st.selectbox(label, list(COURSES))]

    if feature == "Application mode":
        return APPLICATION_MODE[st.selectbox(label, list(APPLICATION_MODE))]

    if feature == "Application order":
        return APPLICATION_ORDER[st.selectbox(label, list(APPLICATION_ORDER))]

    if "Daytime" in feature:
        return DAYTIME[st.selectbox(label, list(DAYTIME))]

    if feature == "Previous qualification":
        return PREV_QUAL[st.selectbox(label, list(PREV_QUAL))]

    if feature == "Mother's qualification":
        return MOTHER_QUAL[st.selectbox(label, list(MOTHER_QUAL))]

    if feature == "Father's qualification":
        return FATHER_QUAL[st.selectbox(label, list(FATHER_QUAL))]

    if feature == "Mother's occupation":
        return MOTHER_OCC[st.selectbox(label, list(MOTHER_OCC))]

    if feature == "Father's occupation":
        return FATHER_OCC[st.selectbox(label, list(FATHER_OCC))]

    if feature == "GDP":
        return GDP[st.selectbox(label, list(GDP))]

    if feature == "Unemployment rate":
        return UNEMP[st.selectbox(label, list(UNEMP))]

    if feature == "Inflation rate":
        return INFLATION[st.selectbox(label, list(INFLATION))]

    # Yes/No binary questions
    if feature in ["Debtor", "Scholarship holder", "International",
                   "Displaced", "Tuition fees up to date"]:
        return 1 if st.radio(label, ["No", "Yes"]) == "Yes" else 0

    # Numeric fields
    if feature == "Admission grade":
        return st.number_input(label, 0.0, 200.0, 100.0)

    if feature == "Previous qualification (grade)":
        return st.number_input(label, 0.0, 200.0, 90.0)

    if feature == "Age at enrollment":
        return st.number_input(label, 15, 70, 18)

    if "grade" in feature:
        return st.number_input(label, 0.0, 20.0, 10.0)

    if "(" in feature:
        return st.number_input(label, 0, 40, 0)

    # Default fallback numeric input
    return st.number_input(label, 0.0, 9999.0, 0.0)

# User picks the model RF or KNN 
model_choice = st.selectbox("Choose Model:", ["Random Forest (RF)", "KNN (scaled)"])

required_features = sf_rf if model_choice == "Random Forest (RF)" else sf_knn

st.markdown("## Enter Required Details Below:")

# Collect all user inputs
inputs = {}
with st.form("form_inputs"):
    for feature in required_features:
        inputs[feature] = make_widget(feature)
    submit = st.form_submit_button("Predict")

# Make the prediction when user clicks the button
if submit:

    # Build dataframe with correct columns
    df = pd.DataFrame([inputs]).reindex(columns=required_features, fill_value=0)

    # KNN requires scaled data
    if model_choice == "KNN (scaled)":
        scaled_data = scaler.transform(df[sf_knn])
        prediction = knn_model.predict(scaled_data)[0]
        probabilities = knn_model.predict_proba(scaled_data)[0]
    else:
        # Random Forest does not need scaling
        prediction = rf_model.predict(df)[0]
        probabilities = rf_model.predict_proba(df)[0]

    # Show result
    st.success(f"### Predicted Outcome: **{label_map[prediction]}**")

    # Show probabilities table
    probs_df = pd.DataFrame({
        "Class": list(label_map.values()),
        "Probability": probabilities
    })

    st.markdown("### Prediction Probabilities")
    st.dataframe(probs_df)
