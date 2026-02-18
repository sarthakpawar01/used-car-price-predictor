import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# ---------------- LOAD FILES (CACHED) ----------------
@st.cache_resource
def load_model():
    model = joblib.load("car_price_model.pkl")
    encoder = joblib.load("encoder.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, encoder, model_columns

@st.cache_data
def load_data():
    return pd.read_csv("car_dataset_clean.csv")

model, encoder, model_columns = load_model()
df = load_data()

# ---------------- GET DROPDOWN LISTS ----------------
brand_list = sorted(df["brand"].unique())
fuel_list = sorted(df["fuel_type"].unique())
seller_list = sorted(df["seller_type"].unique())

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}
.stApp {
    background-color: #fff9c4;
}
.title-box {
    background-color: #0a66c2;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 2px solid #4caf50;
    margin-bottom: 35px;
}
.title-text {
    font-size: 42px;
    font-weight: bold;
    color: white;
}
.subtitle-text {
    font-size: 18px;
    color: #fff3e0;
    margin-top: 8px;
}
.stButton>button {
    background-color: #25d366;
    color: white;
    font-size: 24px;
    border-radius: 12px;
    height: 55px;
    width: 60%;
    margin: auto;
    display: block;
}
.result-box {
    background-color: #5f259f;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: #ffffff;
    margin-top: 25px;
    border: 3px solid #e65100;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("""
<div class="title-box">
    <div class="title-text"> Used Car Price Predictor </div>
    <div class="subtitle-text">Predict resale value instantly using Machine Learning</div>
</div>
""", unsafe_allow_html=True)

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", brand_list)

    filtered_models = df[df["brand"] == brand]["model"].unique()
    model_name = st.selectbox("Model", sorted(filtered_models))

    filtered_cars = df[
        (df["brand"] == brand) &
        (df["model"] == model_name)
    ]["car_name"].unique()

    car_name = st.selectbox("Car Name", sorted(filtered_cars))

    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    vehicle_age = st.slider("Vehicle Age", 0, 20, 5)
    km_driven = st.number_input("KM Driven", 0, 300000, 50000)

with col2:
    fuel_type = st.selectbox("Fuel Type", fuel_list)
    seller_type = st.selectbox("Seller Type", seller_list)

    mileage = st.number_input("Mileage", 0.0, 50.0, 18.0)
    engine = st.number_input("Engine (CC)", 500, 5000, 1200)
    max_power = st.number_input("Max Power", 20.0, 300.0, 80.0)
    seats = st.number_input("Seats", 2, 10, 5)

# ---------------- PREDICT ----------------
if st.button("🚀 Predict Price"):

    input_df = pd.DataFrame({
        "car_name": [car_name],
        "brand": [brand],
        "model": [model_name],
        "vehicle_age": [vehicle_age],
        "km_driven": [km_driven],
        "seller_type": [seller_type],
        "fuel_type": [fuel_type],
        "transmission_type": [transmission],
        "mileage": [mileage],
        "engine": [engine],
        "max_power": [max_power],
        "seats": [seats]
    })

    categorical_cols = input_df.select_dtypes(include=["object"]).columns
    numeric_cols = input_df.select_dtypes(exclude=["object"]).columns

    input_cat = encoder.transform(input_df[categorical_cols])
    input_cat = pd.DataFrame(input_cat, columns=encoder.get_feature_names_out())

    input_num = input_df[numeric_cols].reset_index(drop=True)

    input_final = pd.concat([input_num, input_cat], axis=1)
    input_final = input_final.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_final)

    st.markdown(
        f'<div class="result-box">💰 Estimated Price: ₹ {int(prediction[0]):,}</div>',
        unsafe_allow_html=True
    )

# ---------------- FOOTER ----------------
st.markdown("""
    <div style="
        position: fixed;
        bottom: 10px;
        right: 20px;
        font-size: 14px;
        color: black;
        font-weight: 500;">
        Developed by Sarthak Pawar
    </div>
""", unsafe_allow_html=True)
