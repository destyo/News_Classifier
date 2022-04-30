import pandas as pd
import numpy as np 
import string
import re
import nltk
from nltk.util import ngrams


# Function to remove html tags
def clean_html(text):
    text = re.sub(r'<.*?>', '', text)
    return text

#Function to remove urls
def remove_url(text):
    text = re.sub(r'http\S+', 'url', text)
    return text

# Function to remove /n 
def remove_newline(text):
    text = text.replace('\n', ' ')
    return text

# Function to remove all punctuation
def remove_punctuation(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Fuctions to remove literal quotes and other special characters
remove_quotes = lambda x: x.replace('’', '').replace('“', '').replace('”','').replace('‘','').replace(' — ',' ')

#Funtion to clean numbers in text
def clean_numbers(text):
    text = re.sub(r'\d+', '', text)
    return text

# Lemmatization
def lemmatizer(text):
    lemmatizer = nltk.WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word,pos='v') for word in text.split()]
    return text

# Stop Words
def stopwords_removal(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    return text

# Preprocessing
def preprocessing(text, remove_number=False):
    #Normalize
    text = text.strip().lower()
    # Clean text
    text = clean_html(text)
    text = remove_url(text)
    text = remove_newline(text)
    text = remove_punctuation(text)
    text = remove_quotes(text)
    if remove_number:
        text = clean_numbers(text)
    # Lemmatization
    text = lemmatizer(text)
    #Removing Stop Words
    text = stopwords_removal(text)
    return text