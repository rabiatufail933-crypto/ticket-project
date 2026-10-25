import joblib
import streamlit as st
from text_utils import clean_text

st.title("Intelligent Customer Support System")
st.write("Ticket likho, model category batayega.")

model = joblib.load("models/model.pkl")
ticket = st.text_area("Ticket text", "I was charged twice for my invoice")

if st.button("Classify"):
    clean = clean_text(ticket)
    st.success("Category: " + str(model.predict([clean])[0]))
    if hasattr(model, "decision_function"):
        scores = model.decision_function([clean])[0]
        st.bar_chart(dict(zip(model.classes_, scores)))
