import streamlit as st
import pickle
import numpy as np

# Load the saved model
model=pickle.load(open(r'D:\Machine_Learning_Project\SLR_model\linear_regression_model.pkl','rb'))

# Set the title of the Streamlit app
st.title('Salary Prediction App')

# Add a brief description
st.write('This app predicts the salary based on year of experience using a simple linear regression model. ')

# Add input widget for user to enter year of experience
years_experience=st.number_input('Enter years of experience:',min_value=0.0,max_value=50.0,value=1.0,step=0.5)

# When the button is clicked, make predictions
if st.button('prediction salary'):
    # make a predict using the trained model
    experience_input=np.array([[years_experience]])
    prediction=model.predict(experience_input)

    # Display the result
    st.success(f'The predicted salary for {years_experience} years of experience is: ${prediction[0]:,.2f}')

# Display information about the model
st.write('The model was trained using a dataset of salaries and years of experience.built model by Swati Jadhav')
