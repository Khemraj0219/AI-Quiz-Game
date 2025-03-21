import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from gtts import gTTS
import os

# Load the trained model
MODEL_PATH = 'Best_Voting_Model.joblib'
model = joblib.load(MODEL_PATH)

# Load the dataset (uploaded to your GitHub repository)
DATASET_PATH = 'Unit1_Shapes_5000_Subtopics.csv'
data = pd.read_csv(DATASET_PATH)

# Streamlit App Title
st.set_page_config(page_title="AI-Based Interactive Quiz Game", page_icon="🧠")
st.title("🧠 AI-Based Interactive Quiz Game")
st.markdown("Welcome to the interactive quiz game powered by AI. Let's test your knowledge!")

# Label Encoders for Categorical Data
le_subtopic = LabelEncoder()
le_qtype = LabelEncoder()
le_difficulty = LabelEncoder()

# Encode the dataset
data['Subtopic Encoded'] = le_subtopic.fit_transform(data['Subtopic'])
data['Question Type Encoded'] = le_qtype.fit_transform(data['Question Type'])
data['Difficulty Level Encoded'] = le_difficulty.fit_transform(data['Difficulty Level'])

# Initialize Session State for Score Tracking
if 'score' not in st.session_state:
    st.session_state.score = 0
    st.session_state.question_index = 0
    st.session_state.questions = data.sample(10).reset_index(drop=True)
    st.session_state.feedback = []

# Check if all questions have been answered
if st.session_state.question_index < 10:
    # Get current question data
    question_data = st.session_state.questions.iloc[st.session_state.question_index]
    question = question_data['Question']
    correct_answer = question_data['Correct Answer']
    subtopic = question_data['Subtopic']
    qtype = question_data['Question Type']
    difficulty = question_data['Difficulty Level']

    # Display the question
    st.write(f"### Question {st.session_state.question_index + 1}: {question}")
    user_answer = st.text_input("Your Answer:")

    # Play question as audio
    if st.button("Play Question"):
        tts = gTTS(text=question, lang='en')
        tts.save("question.mp3")
        audio_file = open("question.mp3", "rb")
        st.audio(audio_file.read(), format="audio/mp3")
        audio_file.close()
        os.remove("question.mp3")
    
    if st.button("Submit Answer"):
        if user_answer.strip() != "":
            # Prepare input for the model
            subtopic_encoded = le_subtopic.transform([subtopic])[0]
            qtype_encoded = le_qtype.transform([qtype])[0]
            difficulty_encoded = le_difficulty.transform([difficulty])[0]

            # Prepare the input data for the model
            input_data = [[subtopic_encoded, qtype_encoded, difficulty_encoded]]
            
            # Make prediction with the AI model
            prediction = model.predict(input_data)[0]

            # Check if the user's answer matches the correct answer
            if user_answer.lower().strip() == str(correct_answer).lower().strip():
                st.session_state.score += 1
                st.success(f"✅ Correct! Well done! The correct answer is: {correct_answer}")
                st.session_state.feedback.append(f"Q{st.session_state.question_index + 1}: Correct!")
            else:
                st.error(f"❌ Incorrect! The correct answer is: {correct_answer}")
                st.session_state.feedback.append(f"Q{st.session_state.question_index + 1}: Incorrect!")

            st.write(f"🤖 AI Model Prediction: {prediction}")
            
            # Move to the next question
            st.session_state.question_index += 1

else:
    # Display final score and feedback
    st.write(f"### Quiz Completed! 🎉 Your Score: {st.session_state.score}/10")
    st.write("### Feedback on your answers:")
    for feedback in st.session_state.feedback:
        st.write(feedback)
    st.balloons()

    # Reset quiz for replay
    if st.button("Play Again"):
        st.session_state.score = 0
        st.session_state.question_index = 0
        st.session_state.feedback = []
        st.session_state.questions = data.sample(10).reset_index(drop=True)
