import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

model = open('pipe.pickle', 'rb')
clf = pickle.load(model)
model.close()

st.title('Laptop Predictor')
st.markdown('Estimate the price of the laptop..!')

# brand
company = st.selectbox('Brand', ['Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'Chuwi', 'MSI',
       'Microsoft', 'Toshiba', 'Huawei', 'Xiaomi', 'Vero', 'Razer',
       'Mediacom', 'Samsung', 'Google', 'Fujitsu', 'LG'])

# type of laptop
type_ = st.selectbox('Type', ['Ultrabook', 'Notebook', 'Netbook', 'Gaming', '2 in 1 Convertible',
       'Workstation'])

# Ram
ram = st.selectbox('RAM(in GB)', [2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No','Yes'])

# IPS
ips = st.selectbox('IPS', ['No','Yes'])

# screen size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

# cpu
cpu = st.selectbox('CPU', ['Intel Core i5', 'Intel Core i7', 'AMD Processor', 'Intel Core i3',
       'Other Intel Processor'])

# hdd
hdd = st.selectbox('HDD(in GB)', [0,128,256,512,1024,2048])

# ssd
ssd = st.selectbox('SSD(in GB)', [0,8,128,256,512,1024])

# gpu
gpu = st.selectbox('GPU', ['Intel', 'AMD', 'Nvidia'])

# os
os = st.selectbox('OS', ['Mac', 'Others/No OS/Linux', 'Windows'])

df = {'Company':company,
      'TypeName':type_,
      'Ram':ram,
      'Weight':weight,
      'Touchscreen':touchscreen,
      'IPS':ips,
      'PPI':screen_size,
      'Screen Resolution':resolution,
      'Cpu_Brand':cpu,
      'HDD':hdd,
      'SSD':ssd,
      'Gpu_Brand':gpu,
      'OS':os}

df = pd.DataFrame([df])

if st.button('Predict Price'):
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,type_,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))
