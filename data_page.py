import streamlit as st
import os

base_directory = os.getcwd()
file_path = os.path.join(base_directory, 'friends_script.txt')

with open(file_path, 'r') as file:
    data = file.read()

def display_data_page():
    st.title('Friends Scripts Data (Season 1 to 8)')
    st.info('Some episodes are missings but the script of a given episode should be complete.')
    st.text(data)
