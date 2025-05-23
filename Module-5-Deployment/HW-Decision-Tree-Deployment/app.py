import re
import streamlit as st
import joblib
import pandas as pd

# Load model and preprocessors

# debug model path
# model_data = joblib.load(
# "Module-5-Deployment/HW-Decision-Tree-Deployment/models/weather_model_dt.joblib"
# )

# streamlit model path
model_data = joblib.load("models/weather_model_dt.joblib")

model = model_data['model']
imputer = model_data['imputer']
scaler = model_data['scaler']
encoder = model_data['encoder']
numeric_columns = model_data['numeric_cols']
categorical_columns = model_data['categorical_cols']
encoded_cols = model_data['encoded_cols']
input_cols = model_data['input_cols']

# Title
st.title("Rain Prediction Demo")

# Create input form
st.header('Input Weather Features')

user_input = {}


def format_label(col_name):
    if col_name == "RISK_MM":
        return "The amount of rain (in millimeters) recorded for the next day"
    # Split letters and numbers
    col_name = re.sub(r'(?<=[a-zA-Z])(?=\d)', ' ', col_name)
    col_name = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ',
                      col_name)  # Split camel case
    col_name = col_name.replace("Min", "Minimum")
    col_name = col_name.replace("Max", "Maximum")
    col_name = col_name.replace("Dir", "Direction")
    col_name = col_name.replace("Speed", "Speed (in kilometers per hour)")
    col_name = col_name.replace("Temp", "Temperature (in degrees Celsius)")
    col_name = col_name.replace("Pressure", "Pressure (in hPa)")
    col_name = col_name.replace("Evaporation", "Evaporation (in millimeters)")
    col_name = col_name.replace("Rainfall", "Rainfall (in millimeters)")
    col_name = col_name.replace(
        "RainTomorrow", " Indicator of whether it rained the next day (Yes or No)")
    col_name = col_name.replace("Cloud", "The cloud cover (measured in oktas)")
    return col_name


numeric_cols_left, numeric_cols_right = st.columns(2)
for i, col in enumerate(numeric_columns):
    col_widget = numeric_cols_left if i % 2 == 0 else numeric_cols_right
    df = model_data.get('train_inputs', pd.DataFrame())
    max_val = float(df[col].max() * 100) if col in df else 100.0

    user_input[col] = col_widget.slider(
        label=format_label(col),
        min_value=0.0,
        max_value=max_val,
        value=0.0
    )

train_raw_df = model_data.get('train_raw_df', pd.DataFrame())

cat_cols_left, cat_cols_right = st.columns(2)
for i, col in enumerate(categorical_columns):
    col_widget = cat_cols_left if i % 2 == 0 else cat_cols_right
    label = format_label(col)
    if col == "RainToday":
        user_input[col] = col_widget.selectbox(label, options=["No", "Yes"])
    else:
        options = (
            sorted(train_raw_df[col].dropna().unique())
            if col in train_raw_df
            else ["Unknown"]
        )
        user_input[col] = col_widget.selectbox(label, options=options)

if st.button("Predict"):

    # Step 1: Create single-row DataFrame
    input_df = pd.DataFrame([user_input])

    # Step 2: Impute & Scale Numeric
    input_df[numeric_columns] = imputer.transform(input_df[numeric_columns])
    input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])

    # Step 3: One-hot Encode Categoricals
    encoded = encoder.transform(input_df[categorical_columns])
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols)

    # Step 4: Drop original categorical and add encoded
    input_df.drop(columns=categorical_columns, inplace=True)
    input_df = pd.concat([input_df, encoded_df], axis=1)

    # Step 5: Align columns
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0  # Fill missing dummy variables

    input_df = input_df[model.feature_names_in_]  # Columns order

    # Step 6: Predict
    prediction = model.predict(input_df)[0]
    st.header("Prediction")
    st.write(
        f"Will it rain tomorrow? **{'Yes' if prediction == 'Yes' else 'No'}**")
