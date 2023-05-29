import streamlit as st
import numpy as np
import pandas as pd
import joblib

import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
stop = set(stopwords.words("english"))
import spacy
nlp = spacy.load('en_core_web_sm')

def expand_hashtags(s: str):
    """ Convert string with hashtags """
    s_out = s
    for tag in re.findall(r'#\w+', s):
        s_out = s_out.replace(tag, expand_hashtag(tag))
    return s_out

def remove_last_hashtags(s: str):
    """ Remove all hashtags at the end of the text except #url """
    tokens = TweetTokenizer().tokenize(s)
    # If the URL was added, keep it
    url = "#url" if "#url" in tokens else None
    # Remove hashtags
    while len(tokens) > 0 and tokens[-1].startswith("#"):
        tokens = tokens[:-1]
    # Restore 'url' if it was added
    if url is not None:
        tokens.append(url)
    return ' '.join(tokens)

def expand_hashtag(tag: str):
    """ Convert #HashTag to separated words """
    res = re.findall('[A-Z]+[^A-Z]*', tag)
    return ' '.join(res) if len(res) > 0 else tag[1:]

def remove_stopwords(text) -> str:
    """ Remove stopwords from text """
    filtered_words = [word for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)

def lemmatize(sentence: str) -> str:
    """ Convert all words in sentence to lemmatized form """
    return " ".join([token.lemma_ for token in nlp(sentence)])

def text_clean(text: str) -> str:
    try:
        output = re.sub(r"https?://\S+", '', text)
        output = re.sub(r"@\w+", '', output)
        output = remove_last_hashtags(output)
        output = expand_hashtag(output)
        output = re.sub("[^a-zA-Z]+", ' ', output)
        output = re.sub(r"\s+", " ", output)
        output = lemmatize(output)
        output = remove_stopwords(output)
        return output.lower().strip()
    except:
        return ''

tfidf = joblib.load('tfidf.joblib')
model = joblib.load('disaster_model.joblib')

st.title('Disaster Tweets')
tweet = st.text_area('Enter your tweet',
                    placeholder='What\'s on your mind?')

def predict():

    with st.spinner('Please wait...'):
        text_cleaned = text_clean(tweet)
        vectorized_doc = np.array(tfidf.transform([text_cleaned]).todense())
        result = model.predict_proba(vectorized_doc)
        
        st.caption('Your tweet is \"'+tweet+'\"')
        if result[0][0] < result[0][1]:
            st.error("There is disaster (ᗒᗣᗕ)՞!")
        else :
            st.success("There isn't disaster （。ˇ ⊖ˇ）♡")
        
        chart_data = pd.DataFrame(
            result,
            columns=["No disaster", "Disaster"])
        st.bar_chart(chart_data)

clicked = st.button('Predict!', on_click=predict)