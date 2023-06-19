import streamlit as st
import whisper
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import hamming
import jiwer
import Levenshtein

st.title('Audio Transcription')
model = 'medium'
language = 'english'

st.subheader('Model: ' + model)
st.subheader('Language: ' + language)

with st.spinner('Loading dataset...'):
    df = pd.read_csv('data/' + language + '-audio.tsv', sep='\t')
    df = df[['path', 'sentence']]
st.text('Dataset loaded')    

with st.spinner('Loading transcription model...'):
    model = whisper.load_model(model)
st.text('Transcription Model Loaded')

def predictSentence(filename):
    audio = whisper.load_audio(filename)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    lang = max(probs, key=probs.get)
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    return lang,result.text

def similarity(actual,prediction):
    distance = Levenshtein.distance(actual, prediction)
    # wer = jiwer.wer(actual, prediction)
    return distance
    # corpus = [actual,prediction]
    # vectorizer = TfidfVectorizer()
    # transform = vectorizer.fit_transform(corpus)
    # score = cosine_similarity(transform[0], transform[1])
    # return score[0][0]

with st.container():
    col1, col2,col3, col4, col5 = st.columns(5)
    with col1:
        st.write('Audio')
    with col2:
        st.write('Actual')
    with col3:
        st.write('Prediction')
    with col4:
        st.write('Lang')
    with col5:
        st.write('Levenshtein')        

similarities = []
actuals = []
predictions = []

# for x in range(0, len(df)):
for x in range(0, 5):
    with st.spinner('Predicting Audio # ' + str(x + 1) + '...'):
        audio_file = open('data/' + language + '-audio/' +  df['path'][x],'rb')
        audio_bytes = audio_file.read()
        lang, prediction = predictSentence(audio_file.name)
    with st.container():
        col1, col2,col3, col4, col5 = st.columns(5)
        with col1:
            st.audio(audio_bytes, format='audio/ogg')
        with col2:
            st.write(df['sentence'][x])
        with col3:
            st.write(prediction)
        with col4:
            st.write(lang)
        with col5:
            actuals.append(df['sentence'][x])
            predictions.append(prediction)
            score = similarity(df['sentence'][x],prediction)
            similarities.append(score)
            st.text(round(score,4))


st.text('Total Average Levenshtein: ' + str(round(np.sum(similarities) / len(similarities), 4)))


# audio_file = st.file_uploader('Upload Audio', type=["wav","mp3","m4a"])
# st.session_state.model = whisper.load_model("small")
# st.text('Transcription Model Loaded')


# if (st.sidebar.button('Transcribe Audio')):
#     if ("model" in st.session_state):
#         if (audio_file is not None):    
#             st.sidebar.success("Transcribing Audio")
#             audio = whisper.load_audio(audio_file.name)
#             audio = whisper.pad_or_trim(audio)
#             mel = whisper.log_mel_spectrogram(audio).to(st.session_state.model.device)
#             _, probs = st.session_state.model.detect_language(mel)
#             st.text(f"Detected language: {max(probs, key=probs.get)}")
#             options = whisper.DecodingOptions(fp16=False)
#             result = whisper.decode(st.session_state.model, mel, options)
#             # transcription = st.session_state.model.transcribe(audio_file.name)
#             st.sidebar.success("Transcription Complete")
#             # st.markdown(transcription['text'])
#             st.markdown(result.text)
#         else:
#             st.sidebar.error('Please upload an audio file')        
#     else:
#         st.sidebar.error('Please load transcription model')        

# if (audio_file is not None):
#     audio_bytes = audio_file.read()
#     st.sidebar.audio(audio_bytes, format='audio/ogg')