import streamlit as st
import requests

# Define the backend server url
backend_server_url = "http://127.0.0.1:8000"

# Page configurations
st.set_page_config(
    page_title = ('Prediction'),
    layout = 'wide',
    page_icon = '🧠'
)

cx, cy, cz = st.columns(3)

with cy:
    st.title("SEPSIS PREDICTION APP")

st.divider()

st.markdown('Select a model to predict with')

columna, columnb, columnc = st.columns(3)

with columna:
    if st.button('Adaboost', use_container_width=True):
        st.session_state['backend_server_url'] = "http://127.0.0.1:8000/ad_predict"

with columnb:
    if st.button('Logistic Regression', use_container_width=True):
        st.session_state['backend_server_url'] = "http://127.0.0.1:8000/lr_predict"

with columnc:
    if st.button('Random Forest', use_container_width=True):
        st.session_state['backend_server_url'] = "http://127.0.0.1:8000/rf_predict"



def predict_form():
    

    with st.form('input_feature'):
        # text_fields for sepsis features
        st.header("Input Sepsis Attributes")

        # Define the number of cols to use
        col1, col2, col3 = st.columns(3)

        with col1:
            plasma = st.number_input('Plasma Glucose', min_value=0.0, step=0.1, key='plasma')
            bt1 = st.number_input("Blood Work Result-1 (mu U/ml)", min_value=0.0, step=0.1, key='btn1')
            pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, step=0.1, key='pressure')
        with col2:
            bt2 = st.number_input("Blood Work Result-2 (mm)", min_value=0.0, step=0.1, key='bt2')
            bt3 = st.number_input("Blood Work Result-3 (mu U/ml)", min_value=0.0, step=0.1, key='bt3')
            bmi = st.number_input("Body Mass Index (BMI) ", min_value=0.0, step=0.1, key='bmi')
        with col3:    
             bt4 = st.number_input("Blood Work Result-4 (mu U/ml)", min_value=0.0, step=0.1, key='bt4')
             age = st.number_input("Patient Age", min_value=0, step=1, max_value=100, key='age')
             insurance = st.selectbox("Insurance", ("Positive", "Negative"), placeholder="Are you insured?",  key='insurance')


        # Prediction button
        if st.form_submit_button("Predict Sepsis"):
            # input data dictionary
            input_data = {
                "plasma": plasma,
                "bt1": bt1,
                "pressure": pressure,
                "bt2": bt2,
                "bt3": bt3,
                "age": age,
                "insurance": insurance,
                "bmi": bmi,
                "bt4": bt4
            }
            
            
            # Send request to FastAPI server
            response = requests.post(f"{st.session_state['backend_server_url']}", json=input_data)

            
            # Display the prediction
            if response.status_code == 200:
                prediction = response.json().get('final_prediction')
                st.success(f"The patient is sepsis {prediction['prediction']} with probability of {prediction['probability']}%")
            else:
                try:
                    error_detail = response.json().get('detail', 'No details provided')
                except ValueError:
                    error_detail = 'Non-JSON response received'
                    st.error(f"Error: {error_detail}")
            
            if response == KeyError:
                st.warning('No model selected, please select a model to continue')

if __name__ == '__main__':
    predict_form()