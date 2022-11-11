# Important libaries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from warnings import WarningMessage

df = pd.read_csv("diabetes.csv")

# Title
st.title("Diabetes Prediction App")
st.sidebar.header("Patient Data")
st.subheader("Description Stats of the Data")
st.write(df.describe())

# Data Splitting
X =df.drop(['Outcome'],axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Function to get user input
def user_report():
    pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 2)
    glucose = st.sidebar.slider("Glucose", 0, 300, 110)
    bp = st.sidebar.slider("Blood Pressure", 0, 122, 80)
    sk = st.sidebar.slider("Skin Thickness", 0, 99, 12)
    insulin = st.sidebar.slider("Insulin", 0, 846, 80)
    bmi = st.sidebar.slider("BMI", 0, 67, 5)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.07, 2.42, 0.37)
    age = st.sidebar.slider("Age", 21, 81, 33)

    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'Blood Pressure': bp,
        'Skin Thickness': sk,
        'Insulin': insulin,
        'BMI': bmi,
        'Diabetes Pedigree Function': dpf,
        'Age': age
    }

    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data
#Patient Data
user_data = user_report()
st.subheader("Patient Information")
st.write(user_data)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)
user_result = model.predict(user_data)

# visualization
st.title("Visualized Patient Data")

# color
if user_result[0]==0:
    color = "blue"
else:
    color = "red"

# Age vs pregnancies
st.header("Pregnancy Count Graph (Other vs Yours)")
fig_preg = plt.figure()
ax1 = sns.scatterplot(x='Age', y='Pregnancies', data=df, hue='Outcome', palette='Greens')
ax2 = sns.scatterplot(x=user_data['Age'], y=user_data['Pregnancies'], s=150, color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 20, 2))
plt.title("0 = No Diabetes, 1 = Diabetes")
st.pyplot(fig_preg)

#outcome
st.header("Your Report: ")
output = ""
if user_result[0]==0:
    output = "No Diabetes"
    st.balloons()
else:
    output = "Metha kam khaien"
    st.warring("Sugar Level High")
st.title(output)
# st.subheader("Accuracy: ")
# st.write(accuracy_score(y_test, model.predict(X_test))*100 + "%")