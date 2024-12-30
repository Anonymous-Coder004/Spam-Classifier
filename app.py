import streamlit as st
import pickle as pkl
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
nltk. download('punkt')
nltk.download('punkt_tab')
stem=PorterStemmer()
def tranform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y=[]
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y=[]
    for i in text:
        y.append(stem.stem(i))

    return " ".join(y)
tfidf=pkl.load(open('vectorizer.pkl','rb'))
model=pkl.load(open('model.pkl','rb'))
st.title('SMS Spam Classifier')
input_text=st.text_area('Enter the message')
if st.button('Predict'):
    transform_sms=tranform_text(input_text)
    vector_input=tfidf.transform([transform_sms])
    result=model.predict(vector_input)[0]
    if result==1:
        st.header('Spam')
    else:
        st.header('Not Spam')