import streamlit as st
import google.generativeai as genai
api_key = "AIzaSyBhND33Sv5KLwqlGYlfGJUxVWbpco1JNOU"
genai.configure(api_key = api_key)

st.header("Sentiment Analysis")
text = st.text_input("Enter the movie review")

if st.button("get response"):
  model = genai.GenerativeModel("gemini-1.0-pro")
  response = model.generate_content(text + "this text negative or positive ? give answer only one word.")
  st.header(response.text)
