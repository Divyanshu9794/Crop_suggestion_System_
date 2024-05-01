import numpy as np
import pandas as pd
import streamlit as st
import pickle


# import sklearn
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.figure_factory as ff
try:
    model = pickle.load(open('model.pkl', 'rb'))
    df = pd.read_csv("Crop_recommendation.csv")
except Exception as e:
    print("An error occurred:", e)

converts_dict = {
    'Nitrogen': 'N',
    'Phosphorus': 'P',
    'Potassium': 'K',
    'Temperature': 'temperature',
    'Humidity': 'humidity',
    'Rainfall': 'rainfall',
    'ph': 'ph'
}

def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    input = np.array([[n, p, k, temperature, humidity, ph, rainfall]]).astype(np.float64)
    prediction = model.predict(input)
    return prediction[0]

# def scatterPlotDrawer(x,y):
#     fig = plt.figure(figsize=(20,15))
#     sns.set_style("whitegrid")
#     sns.scatterplot(data=df, x=x, y=y, hue="label", size="label", palette="deep", sizes=(20, 200), legend="full")
#     plt.xlabel(x, fontsize=22)
#     plt.ylabel(y, fontsize=22)
#     plt.xticks(rotation=90, fontsize=18)
#     plt.legend(prop={'size': 18})
#     plt.yticks(fontsize=16)
#     st.write(fig)

# def barPlotDrawer(x,y):
#     fig = plt.figure(figsize=(20,15))
#     sns.set_style("whitegrid")
#     sns.barplot(data=df, x=x, y=y)
#     plt.xlabel("Crops", fontsize=22)
#     plt.ylabel(y, fontsize=22)
#     plt.xticks(rotation=90, fontsize=18)
#     plt.legend(prop={'size': 18})
#     plt.yticks(fontsize=16)
#     st.write(fig)

# def boxPlotDrawer(x,y):
#     fig = plt.figure(figsize=(20,15))
#     sns.set_style("whitegrid")
#     sns.boxplot(x=x, y=y, data=df)
#     sns.despine(offset=10, trim=True)
#     plt.xlabel("Crops", fontsize=22)
#     plt.ylabel(y, fontsize=22)
#     plt.xticks(rotation=90, fontsize=18)
#     plt.legend(prop={'size': 18})
#     plt.yticks(fontsize=16)
#     st.write(fig)

def main():
    html_temp_pred = """
    <div style="background-color:#025246 ;padding:10px;margin-bottom:30px">
    <h2 style="color:white;text-align:center;"> Which Crop To Cultivate? </h2>
    </div>
    """

    st.markdown(html_temp_pred, unsafe_allow_html=True)
    st.header("To predict your crop give values")
    
    n = st.text_input('Nitrogen', '0')
    p = st.text_input('Phosphorus', '0')
    k = st.text_input('Potassium', '0')
    temperature = st.text_input('Temperature', '0.0')
    humidity = st.text_input('Humidity', '0.0')
    ph = st.text_input('pH', '0.0')
    rainfall = st.text_input('Rainfall', '0.0')

    n = int(n)
    p = int(p)
    k = int(k)
    temperature = float(temperature)
    humidity = float(humidity)
    ph = float(ph)
    rainfall = float(rainfall)

    try:
        output_number = predict_crop(n, p, k, temperature, humidity, ph, rainfall)
        st.success('The most suitable crop for your field is {}'.format(output_number))
    except Exception as e:
        st.error(f"An error occurred while predicting the crop: {e}")
