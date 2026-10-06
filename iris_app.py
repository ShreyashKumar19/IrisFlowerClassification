import streamlit as st
import joblib


# Load the saved model and scaler
model = joblib.load("iris_model.pkl")
scaler = joblib.load("iris_scaler.pkl")


st.title("Iris Flower Classifier")

st.write("Enter the measurements of the flower below.")

sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    value=0.2
)


if st.button("Predict Species"):

    flower = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    flower_scaled = scaler.transform(flower)

    prediction = model.predict(flower_scaled)

    st.success("Predicted Species: " + prediction[0])
