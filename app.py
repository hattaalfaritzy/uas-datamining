import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load pre-trained model
loaded_model = pickle.load(open('model_uas.pkl', 'rb'))

# Streamlit app
def main():
    # Set page title and favicon
    st.set_page_config(
        page_title="Insurance Charges Prediction App",
        page_icon="📊"
    )

    # Set app title and introduction with custom color
    st.title("Insurance Prediction App")
    st.subheader("Nama : Muhammad Hatta Alfaritzy")
    st.subheader("NIM : 2020230059")

    # Sidebar styling
    st.sidebar.header("User Input")
    st.sidebar.markdown("Adjust the values to make predictions.")

    # Input fields for user to enter data
    age = st.sidebar.slider("Age", min_value=18, max_value=100, value=25)
    sex = st.sidebar.radio("Sex", options=['Female', 'Male'])
    bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    children = st.sidebar.slider("Number of Children", min_value=0, max_value=5, value=0)
    smoker = st.sidebar.radio("Smoker", options=['No', 'Yes'])

    # Convert categorical inputs to numerical values
    sex = 0 if sex == 'Female' else 1
    smoker = 1 if smoker == 'Yes' else 0

    # Display user input in a styled card with custom color
    st.sidebar.subheader("User Input:")
    user_input_df = pd.DataFrame(np.array([age, sex, bmi, children, smoker]).reshape(1, -1),
                                  columns=['Age', 'Sex', 'BMI', 'Children', 'Smoker'])
    st.sidebar.dataframe(user_input_df.style.set_properties(**{'background-color': '#ffe699', 'color': 'black'}))

    # Make prediction using the loaded model
    charge_pred = loaded_model.predict(user_input_df)[0]

    # Display prediction in a styled card with custom color
    st.subheader("Predicted Insurance Charges:")
    st.write(f"${charge_pred:.2f}", key='prediction', format="%.2f", help="Predicted value",
             unsafe_allow_html=True,
             )
    st.markdown(
        """<style>
                div[data-baseweb="card"] {
                    background-color: #99ff99;
                    color: black;
                }
            </style>""",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
