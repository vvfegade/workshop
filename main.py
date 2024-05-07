import streamlit as st
import pandas as pd
import pycaret
from pycaret.classification import load_model

#st.write("Streamlit version:", st.__version__)
#st.write("Pandas version:", pd.__version__)
#st.write("PyCaret version:", pycaret.__version__)

# Load the trained model
saved_final_model = load_model('classification_titanic')

# Function to preprocess user input
def preprocess_input(pclass, sex, age, sibsp, parch, fare, cabin, embarked):
    # Create a DataFrame with user input
    input_df = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Cabin': [cabin],
        'Embarked': [embarked]
    })
    return input_df

# Function to predict survival
def predict_survival(input_df):
    prediction = saved_final_model.predict(input_df)
    probability = saved_final_model.predict_proba(input_df)[0][1]
    return prediction, probability

# Streamlit UI
def main():
    st.title('Titanic Survival Prediction')
    st.markdown('''
    This app predicts whether a passenger survived the Titanic disaster.
    Please enter the required information:
    ''')

    # User input fields
    pclass = st.selectbox('Passenger Class', [1, 2, 3])
    sex = st.selectbox('Sex', ['male', 'female'])
    age = st.number_input('Age', min_value=0, max_value=100, value=30)
    sibsp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)
    parch = st.number_input('Number of Parents/Children Aboard', min_value=0, max_value=10, value=0)
    fare = st.number_input('Fare', min_value=0, max_value=1000, value=50)
    cabin = st.text_input('Cabin', '')
    embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])

    # Predict button
    if st.button('Predict'):
        input_df = preprocess_input(pclass, sex, age, sibsp, parch, fare, cabin, embarked)
        prediction, probability = predict_survival(input_df)
        if prediction[0] == 1:
            st.success(f'The passenger is predicted to have survived with a probability of {probability:.2f}')
        else:
            st.error(f'The passenger is predicted to have not survived with a probability of {1-probability:.2f}')

if __name__ == '__main__':
    main()
