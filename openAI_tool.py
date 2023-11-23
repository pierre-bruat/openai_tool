

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
from streamlit_tags import st_tags 
from streamlit_option_menu import option_menu
from openai import OpenAI
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

role = "Tu es un ingénieur linguistique"

def seo_insights(df):
        answers_list = []
        for row in tqdm(df.itertuples()):
                keyword = " ".join(row.keyword.split(" "))
                response = openai.ChatCompletion.create(
                        model ="gpt-4",
                        messages = [
                        {"role":"system" , "content": role},
                        {"role":"user" , "content":f"Dans le cadre de la rédaction éditoriale d'un contenu autour du sujet suivant : {keyword}. Retourne les termes issus du champ lexical / sémantique autour de ce mot clé : {keyword} sous forme de liste. Les termes doivent être séparés par des virgules. "}],
                        max_tokens = 1000)
                result = ''
                for choice in response.choices:
                        result += choice.message.content
                answers_list.append(result)
        df["terms"] = answers_list
        return df



def calculate_score(row):
    content = row['Content']
    terms = [term.strip() for term in row['terms'].split(',')]
    terms_count = len(terms)
    terms_found = 0
    missing_terms = []

    for term in terms:
        if term.lower() in content:
            terms_found += 1
        else:
            missing_terms.append(term)

    score = (terms_found / terms_count) * 100
    missing_terms_str = ', '.join(missing_terms)  # Conversion de la liste en chaîne de caractères
    df['score'], df['missing_terms'] = zip(*df.apply(calculate_score, axis=1))
    return df





with st.sidebar:
    choose = option_menu("SEO toolbox", ["OpenAI tool","CHATGPT","ContentScoring"],
                     icons=['robot','robot','robot'],
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


#https://platform.openai.com/docs/guides/chat/chat-vs-completions
if choose =="CHATGPT":
    form = st.form(key='my-form-21')
    API_key = form.text_input("Insert API key")
    GTP_version = form.selectbox('Select GTP version', ('gpt-3.5-turbo', 'gpt-4-1106-preview'))
    Max_token = form.number_input("Max token value")
    role = form.text_area("User : Describe who you want me to be")
    promt = form.text_area("Prompt")
    submit = form.form_submit_button('Submit')
    if submit:
        client = OpenAI(api_key=API_key,)
        gif_runner = st.image("bsbot.gif")
        response = client.chat.completions.create(
            model = GTP_version,
            messages = [
            {"role":"system" , "content": role},
            {"role":"user" , "content": promt}],
            max_tokens = Max_token)
        result = ''
        for choice in response.choices:
            result += choice.message.content
        gif_runner.empty()
        st.write(result)




def seo_insights(df):
        answers_list = []
        for row in tqdm(df.itertuples()):
                keyword = " ".join(row.keyword.split(" "))
                response = openai.ChatCompletion.create(
                        model ="gpt-4",
                        messages = [
                        {"role":"system" , "content": role},
                        {"role":"user" , "content":f"Dans le cadre de la rédaction éditoriale d'un contenu autour du sujet suivant : {keyword}. Retourne les termes issus du champ lexical / sémantique autour de ce mot clé : {keyword} sous forme de liste. Les termes doivent être séparés par des virgules. "}],
                        max_tokens = 1000)
                result = ''
                for choice in response.choices:
                        result += choice.message.content
                answers_list.append(result)
        df["terms"] = answers_list
        return df




def calculate_score(row):
    content = row['Content'].lower()
    terms = [term.strip() for term in row['terms'].split(',')]
    terms_count = len(terms)
    terms_found = 0
    missing_terms = []

    for term in terms:
        if term.lower() in content:
            terms_found += 1
        else:
            missing_terms.append(term)

    score = (terms_found / terms_count) * 100
    missing_terms_str = ', '.join(missing_terms)  # Conversion de la liste en chaîne de caractères
    return score, missing_terms_str


def st_tags(value: list,
            suggestions: list,
            label: str,
            text: str,
            maxtags: int,
            key=None) -> list:
    '''
    :param maxtags: Maximum number of tags allowed maxtags = -1 for unlimited entries
    :param suggestions: (List) List of possible suggestions (optional)
    :param label: (Str) Label of the Function
    :param text: (Str) Instructions for entry
    :param value: (List) Initial Value (optional)
    :param key: (Str)
        An optional string to use as the unique key for the widget.
        Assign a key so the component is not remount every time the script is rerun.
    :return: (List) Tags

    Note: usage also supports keywords = st_tags()
    '''


# if choose =="ContentScoring":
#     form = st.form(key='my-form-22')
#     API_key = form.text_input("Insert API key")
#     keyword = form.text_input("Insert your keyword")
#     content = form.text_area('Text to analyze')
#     submit = form.form_submit_button('Submit')
#     if submit:
#         data = {'keyword': [keyword],'Content':[content]} 
#         df = pd.DataFrame(data)  
#         openai.api_key = API_key
#         gif_runner = st.image("bsbot.gif")
#         result = seo_insights(df)
#         gif_runner.empty()
#         df['score'], df['missing_terms'] = zip(*df.apply(calculate_score, axis=1))
#         st.metric("Optimization score",df["score"])
#         st.table(df)
#         missing_kw_list = df['missing_terms'].str.split(', ').tolist()
#         missing_kw_list = [mot_cle for sous_liste in missing_kw_list for mot_cle in sous_liste]
#         st.write(missing_kw_list)
#         st.write(type(missing_kw_list))
#         st_tags(value = missing_kw_list, suggestions = ["add new terms"], label=  "Enter keywords", text= "Press enter to add more", maxtags= 20, key=1)
#         keywords = st_tags(label='# Enter Keywords:', text='Press enter to add more', value=['Zero', 'One', 'Two'],suggestions=['five', 'six', 'seven', 'eight', 'nine', 'three', 'eleven', 'ten', 'four'],maxtags = 4, key='1')
#         form = st.form(key='my-form-23')
#         st_tags(value = missing_kw_list, suggestions = ["add new terms"], label=  "Enter keywords", text= "Press enter to add more", maxtags= 20, key=1)
#         keywords = st_tags(value = missing_kw_list, suggestions = ["add new terms"], label=  "Enter keywords", text= "Press enter to add more", maxtags= 20, key=1)
#         st.write(keywords)
