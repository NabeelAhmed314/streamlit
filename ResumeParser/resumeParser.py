import streamlit as st
import pandas as pd
from pyresparser import ResumeParser
import numpy as np

st.title('Resume Parser')

sparko = pd.DataFrame([(1, 'SPARKO'),(2, 'SPARKO'),(3, 'SPARKO'),(4, 'SPARKO'),(5, 'SPARKO')], columns =['ID', 'Category'])

with st.spinner('Loading dataset...'):
    df = pd.read_csv('data/resume.csv', sep=",")
    df = df[['ID', 'Category']]
    df = pd.concat([df,sparko])
    df = df[df['Category'] == "SPARKO"]
st.text('Dataset loaded')    

st.write(df)

# st.text("\data\data\HR")
# st.text('data/data/' + df['Category'][0] + '/' + str(df['ID'][0]) + '.pdf')

# st.write(df)

for x in range(0, 5):
    with st.spinner('Parsing Resume # ' + str(df['ID'][x]) + '...'):
        data = ResumeParser('data/data/' + df['Category'][x] + '/' + str(df['ID'][x]) + '.pdf').get_extracted_data()
    st.subheader('Resume # ' + str(df['ID'][x]))
    st.write(data)   