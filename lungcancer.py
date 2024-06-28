import numpy as np
import pandas as pd 
#import matplotlib.pyplot as plt
#import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import streamlit as st 




def inputprocessor():
    #Reading the file
    Data = pd.read_csv("surveylungcancer.csv")
    Data = Data.drop("GENDER", axis = 1)

    Data = Data.replace(1, 0)
    Data = Data.replace(2, 1)
    #Test Data collection
    st.sidebar.header("Lung Cancer Model Input Parameters")
    Age = st.sidebar.text_input("Age", 26)
    Smoking = st.sidebar.selectbox(label="smoking", options= ["Yes", "No"])
    Yellow_Fingers = st.sidebar.selectbox(label="Yellow Fingers", options= ["Yes", "No"])
    Anxiety = st.sidebar.selectbox(label="Anxiety", options= ["Yes", "No"])
    PeerPressure = st.sidebar.selectbox(label="Peer Pressure", options= ["Yes", "No"])
    Chronic_Disease = st.sidebar.selectbox(label="Chronic Disease", options= ["Yes", "No"])
    Fatigue = st.sidebar.selectbox(label="Fatigue", options= ["Yes", "No"])
    Allergy = st.sidebar.selectbox(label="Allergy", options= ["Yes", "No"])
    Wheezing = st.sidebar.selectbox(label="Wheezing", options= ["Yes", "No"])
    Alcohol = st.sidebar.selectbox(label="Alcohol", options= ["Yes", "No"])
    Coughing = st.sidebar.selectbox(label="Coughing", options= ["Yes", "No"])
    Shortness_of_breath = st.sidebar.selectbox(label="Shortness of Breath", options= ["Yes", "No"])
    Swallowing_Difficulty = st.sidebar.selectbox(label="Swallowing_Difficulty", options= ["Yes", "No"])
    Chest_Pain = st.sidebar.selectbox(label="Chest pain", options= ["Yes", "No"])
    button = st.sidebar.button("Process")
    test_data = pd.DataFrame({
        'AGE': [Age],
        'SMOKING': [Smoking],
        'YELLOW_FINGERS': [Yellow_Fingers],
        'ANXIETY': [Anxiety],
        'PEER_PRESSURE': [PeerPressure],
        'CHRONIC DISEASE': [Chronic_Disease],
        'FATIGUE': [Fatigue],
        'ALLERGY': [Allergy],
        'WHEEZING': [Wheezing],
        'ALCOHOL CONSUMING': [Alcohol],
        'COUGHING': [Coughing],
        'SHORTNESS OF BREATH': [Shortness_of_breath],
        'SWALLOWING DIFFICULTY': [Swallowing_Difficulty],
        'CHEST PAIN': [Chest_Pain]
    })
    st.header("LUNG CANCER MODEL")
    st.write(test_data)
    st.subheader("Result will appear in this section")

    test_data = test_data.replace('Yes', 1)
    test_data = test_data.replace('No', 0)
    Data = Data.replace('YES', 1)
    Data = Data.replace('NO', 0)
    

    if button:      
        X_train = Data.drop('LUNG_CANCER', axis = 1)
        Y_train = pd.DataFrame({
            'LUNG_CANCER': Data['LUNG_CANCER']
        })


        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        test_data = np.array(test_data)

        #print(Y_train)
        #print(Data)

        #sns.heatmap(Data.corr())
        #plt.show()
        #accuracy = 0.739
        
        classifier = XGBRegressor(learning_rate = 0.05, subsample = 1.0, colsample_bynode = 0.4)
        classifier.fit(X_train, Y_train)
        predicted_outcome = classifier.predict(test_data)

        
        if predicted_outcome >= 0.7:
            predicted_outcome = 1
        elif predicted_outcome < 0.7:
            predicted_outcome = 0

        if predicted_outcome == 0:
            st.write(f'The predicted result is {predicted_outcome}. Cancer not detected')
        elif predicted_outcome == 1:
            st.write(f'The predicted result is {predicted_outcome}. Cancer detected. You are adviced to see a doctor')
inputprocessor()
