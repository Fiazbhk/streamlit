import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

st.header("BMI Calculator")

weight = st.number_input("Weight (kg)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
height = st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=150.0, step=0.1)
bmi = weight / ((height/100)**2)
st.write("Your BMI = ", bmi)









