import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re

def lower_text(text: str):
    return text.str.lower()

def remove_links(text: str):
    return re.sub(r'http[s]?://(?:[a-zA-Z0-9.-]+)', '', text)

def remove_special_characters(text: str):
    return re.sub(r'[!@#$%^&*()+{}\[\];:\'".?/<>|\\~`]', '', text)

def remove_newline_characters(text: str) -> str:
    return text.replace("\n", " ")

def remove_whitespaces_from_beg_and_end(text: str) -> str:
    return text.strip()