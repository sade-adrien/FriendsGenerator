import streamlit as st
from predict_page import display_predict_page
from data_page import display_data_page
import parameters
import torch
from model import *
import os

#Loading model
base_directory = os.getcwd()
weights_file = os.path.join(base_directory, 'model_1.05.pth')

model = torch.load(weights_file, map_location=parameters.device)
model.eval()


#Web App with streamlit
st.markdown(
    f'''
        <style>
            .sidebar .sidebar-content {{
                width: 37px;
            }}
        </style>
    ''',
    unsafe_allow_html=True
)

page = st.sidebar.selectbox('Generator or Data', ('Friends Generator', 'Training Data'))

if page == 'Friends Generator':
    display_predict_page(model)

elif page == 'Training Data':
    display_data_page()
