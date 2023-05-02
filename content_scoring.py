

import concurrent.futures
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
