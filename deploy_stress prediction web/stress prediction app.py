# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 18:22:08 2021

@author: natvi
"""

import pickle
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

with open('D:/TK/SEM 5/psd/final project/deploy_stress prediction web/deploy_data.pkl', 'rb') as f:
    loaded_dataset = pickle.load(f)

# Model Building
model = Pipeline([('vectorizer', TfidfVectorizer(
    min_df=2, 
    ngram_range=(1, 2),
    lowercase=True, 
    max_features=4500, 
    stop_words='english',
    use_idf=True,
    smooth_idf=True
    )), ('svc', SVC(C=10, gamma=0.01, probability=True))])
x = loaded_dataset["text"].values.tolist()
y = loaded_dataset["label"]
model = model.fit(x , y)




def main():
    menu = ["Home", "About"]
    
    # giving a title
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Stress Prediction</h2>
    </div>
    <br>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.subheader("Home")
        with st.form(key='mlform'):
            col1, col2 = st.columns([2, 1])
            with col1:
                message = st.text_area("Commentar")
                submit_message = st.form_submit_button(label="Predict")
            with col2:
                st.write("Online Incremental")
                st.write("Predict Commentar as Stress or Not Stress")
        
        if submit_message:
            prediction = model.predict([message])
            prediction_proba = model.predict_proba([message])
            st.success("Data submitted")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.info("Original Input Text")
                st.write(message)
                
                st.success("Prediction Result")
                if (prediction[0] == 0):
                    st.write('The person is not stress')
                else:
                    st.write('The person is stress')
            
            with res_col2:
                st.info("Probability Matrix")
                st.write(prediction_proba)
                
    else:
        st.subheader("About")
        st.write("This Stress Prediction Web Application made by Kelompok 7 PSD A which consists of ")
        st.write("- Nathanael Victor Darenoh (195150207111038)")
        st.write("- Farid Syauqi Nirwan (195150207111052)")
        st.write("- Farhan Rizqi Pradiptya (195150207111009)")
        
if __name__ =='__main__':
    main()
        