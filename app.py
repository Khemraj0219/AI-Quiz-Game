import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
MODEL_PATH = 'Best_Voting_Model.joblib'
model = joblib.load(MODEL_PATH)

def make_prediction(subtopic, question_type, difficulty_level):
    """
    Make predictions using the trained model.
    """
    try:
        le_subtopic = LabelEncoder()
        le_qtype = LabelEncoder()
        le_difficulty = LabelEncoder()
        
        # Sample encoding to simulate the trained model's encoding
        subtopic_encoded = le_subtopic.fit_transform([subtopic])[0]
        qtype_encoded = le_qtype.fit_transform([question_type])[0]
        difficulty_encoded = le_difficulty.fit_transform([difficulty_level])[0]
        
        # Prepare the input for prediction
        input_data = [[subtopic_encoded, qtype_encoded, difficulty_encoded]]
        
        # Make prediction
        prediction = model.predict(input_data)
        return prediction[0]
    except Exception as e:
        return f"Error during prediction: {e}"

# Streamlit App Title
st.set_page_config(page_title="AI-Based Interactive Quiz Game", page_icon="🧠")
st.title("🧠 AI-Based Interactive Quiz Game")
st.markdown("Welcome to the interactive quiz game powered by AI. Ask your questions and get answers instantly!")

# User Input Section
st.sidebar.title("Enter Question Details:")
subtopic = st.sidebar.selectbox("Select Subtopic:", ["Shapes", "Geometry", "Arithmetic", "Algebra"])
question_type = st.sidebar.selectbox("Select Question Type:", ["MCQ", "True/False", "Fill in the Blank", "Short Answer"])
difficulty_level = st.sidebar.selectbox("Select Difficulty Level:", ["Easy", "Medium", "Hard"])

question = st.text_input("🔍 Enter your question here:")

if st.button("Submit Question"):
    if question:
        st.write("🔄 Processing your question...")

        # Make a prediction using the model
        prediction = make_prediction(subtopic, question_type, difficulty_level)
        
        st.success(f"🤖 Predicted Answer: {prediction}")
    else:
        st.warning("❗ Please enter a question.")

# Footer
st.markdown("<br><hr><center>Made with ❤️ by Your Name</center>", unsafe_allow_html=True)
