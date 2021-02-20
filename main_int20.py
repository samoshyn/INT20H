import streamlit as st
import numpy as np
import pandas as pd
from predict_with_test import *
from predict_by_hand import *

#from utilits_nbu.utils import *

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}    
    """
st.markdown(hide_footer_style, unsafe_allow_html=True)

def main():
    st.sidebar.title("Як бажаєте сформувати запит до моделі:")
    app_mode = st.sidebar.radio("Обрати формат", ["Екземпляр тестового набору", "Задати власноруч"])
    
    if app_mode == "Екземпляр тестового набору":
        pred_id = select_predict_sample()
        pred_value = make_predict(pred_id)            
        return pred_value
    
    if app_mode == "Задати власноруч":
        pred_id = make_data()
        pred_value = make_predict(pred_id)            
        return pred_value
pred_value = main()