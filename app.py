import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Model load karna (Jo tumne Kaggle se download kiya tha)
try:
    model = joblib.load('iitm_model.pkl')
except:
    st.error("Model file 'iitm_model.pkl' nahi mili! Please use upload karein.")

st.set_page_config(page_title="IITM Success Predictor AI", page_icon="🎓")

st.title("🎓 IITM BS Data Science: Success Predictor")
st.markdown("Yeh app **Random Forest Classifier (95.5% Accuracy)** use kar raha hai.")

# User Inputs
st.sidebar.header("Apne Details Fill Karein")
math = st.sidebar.slider("Mathematics Score", 0, 100, 70)
english = st.sidebar.slider("English Score", 0, 100, 70)
logic = st.sidebar.slider("Logic & Aptitude", 0, 100, 70)
hours = st.sidebar.slider("Hafte mein kitne ghante padhte ho?", 5, 50, 20)
coding = st.sidebar.selectbox("Kya pehle coding aati hai?", ["No", "Yes"])

coding_val = 1 if coding == "Yes" else 0

# DataFrame for Prediction
user_data = pd.DataFrame({
    'math_score': [math],
    'english_score': [english],
    'logic_score': [logic],
    'study_hours': [hours],
    'coding_exp': [coding_val]
})

if st.button("Predict My Success "):
    prediction = model.predict(user_data)[0]
    # AI kitna sure hai (Probability)
    probability = model.predict_proba(user_data)[0][1] * 100

    if prediction == 1:
        st.success(f"### Chance of Success: {probability:.1f}%")
        st.balloons()
        st.write("AI ke mutabiq aapka profile IITM Qualifier ke liye bahut strong hai!")
    else:
        st.warning(f"### Chance of Success: {probability:.1f}%")
        st.write("Aapko Logic aur Math par thoda aur kaam karne ki zaroorat hai.")

    # Visualization
    categories = ['Math', 'English', 'Logic', 'Study Hours', 'Experience']
    values = [math, english, logic, (hours/50)*100, coding_val*100]
    fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself'))
    st.plotly_chart(fig)
