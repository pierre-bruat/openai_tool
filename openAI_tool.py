

import concurrent.futures
#from client import RestClient
import datetime
from datetime import datetime, timedelta, time, date
import json
from datetime import *
import io
import numpy as np
import pandas as pd
import os
import openai
import requests
import streamlit as st
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid
import streamlit.components.v1 as html
from serpapi import GoogleSearch
from st_aggrid import AgGrid
import time as t
from tqdm import tqdm
import urllib
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib_venn import venn2 


from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

###


st.set_page_config(
    page_title='OpenAI tool', 
    page_icon=':brain',
    #layout = "wide",
    initial_sidebar_state='expanded',
    )
#st.sidebar.success("Menu")



hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
st.markdown(hide_table_row_index, unsafe_allow_html=True)

#@st.cache(ttl=3600)
#@st.cache(suppress_st_warning=True)



with st.sidebar:
    choose = option_menu("SEO toolbox", ["OpenAI tool","CHATGPT"],
                     icons=['robot'],
                     menu_icon="app-indicator", 
                     default_index=0, 
                     orientation="vertical",
                     styles={
    "container": {"padding": "5!important", "background-color": "#fafafa"},
    "icon": {"color": "#1e3c87", "font-size": "25px"}, 
    "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
    "nav-link-selected": {"background-color": "#1e3c87"},
    "alert":"display:none"}
)



if choose =="OpenAI tool":
    form = st.form(key='my-form-20')
    API_key = form.text_input("Insert API key")
    query = form.text_input("Ask anything you want")
    submit = form.form_submit_button('Submit')
    if submit:
        gif_runner = st.image("bsbot.gif")
        openai.api_key = API_key
        response = openai.Completion.create(
            #model="gpt-3.5-turbo",
            model="text-davinci-003",
            prompt=query,
            temperature=0,
            max_tokens=2000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)

        data = response.choices[0].text
        gif_runner.empty()
        st.write(data)


#https://platform.openai.com/docs/guides/chat/chat-vs-completions
if choose =="CHATGPT":
    form = st.form(key='my-form-21')
    API_key = form.text_input("Insert API key")
    role = form.text_input("Who am I ?")
    promt = form.text_input("Prompt")
    submit = form.form_submit_button('Submit')
    if submit:
        openai.api_key = API_key
        gif_runner = st.image("bsbot.gif")
        response = openai.ChatCompletion.create(
            model ="gpt-3.5-turbo",
            messages = [
            {"role":"system" , "content": role},
            {"role":"user" , "content": promt}],
            max_tokens = 2000)
        result = ''
        for choice in response.choices:
            result += choice.message.content
        gif_runner.empty()
        st.write(result)
