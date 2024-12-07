from flask import Flask, render_template, request, jsonify
from nltk.chat.util import Chat, reflections
from joblib import load
import pandas as pd
import sklearn.preprocessing
import json
import random

app = Flask(__name__)

# Load the training data
training = pd.read_csv('disease prediction/Data/Training.csv')

# Load the saved model
loaded_model = load('disease_prediction_model.joblib')

# cols based on the training data
cols = training.columns[:-1]

# Load the LabelEncoder used during training
le = sklearn.preprocessing.LabelEncoder()
le.fit(training['prognosis'])

# Load chatbot pairs from JSON file
with open('chatbot_pairs.json', 'r') as file:
    chatbot_pairs_data = json.load(file)

# Extract chatbot pairs
chatbot_pairs = chatbot_pairs_data.get('chatbot_pairs', [])

# Load disease pairs from JSON file
with open('disease_pairs.json', 'r') as file:
    disease_pairs_data = json.load(file)

# Extract disease pairs
disease_pairs = disease_pairs_data.get('disease_pairs', [])

# Combine chatbot pairs and disease pairs
pairs = chatbot_pairs + disease_pairs

# Convert pairs to the format needed by Chat
patterns_responses = [(pattern, response) for pattern, responses in pairs for response in responses]

# dictionary of reflections for NLTK chatbot
reflections = {
    "I am": "you are", "I was": "you were", "I": "you", "I'm": "you are", "I'd": "you would",
    "I've": "you have", "I'll": "you will", "my": "your", "you are": "I am", "you were": "I was",
    "you've": "I have", "you'll": "I will", "your": "my", "yours": "mine", "you": "me", "me": "you"
}

# Function to predict disease based on user symptoms
def predict_disease(user_input_text):
    user_input_symptoms = set(user_input_text.lower().split(','))
    user_symptoms = [1 if symptom.lower() in user_input_symptoms else 0 for symptom in cols]
    user_input_df = pd.DataFrame([user_symptoms], columns=cols)
    user_prediction = loaded_model.predict(user_input_df)
    predicted_disease = le.inverse_transform(user_prediction)
    return predicted_disease[0]

# Function to handle chatbot responses
def get_chatbot_response(user_input_text):
    for pair in pairs:
        patterns = pair.get("patterns", [])
        responses = pair.get("responses", [])
        for pattern in patterns:
            if pattern.lower() in user_input_text.lower():
                return random.choice(responses)
    return None

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input_text = request.form['user_input']
    symptoms_input = request.form['symptoms']
    
    if user_input_text.lower() == 'quit':
        return jsonify({'response': "Goodbye! Have a great day."})

    # Check if the user has entered symptoms
    elif "symptoms" in request.form and symptoms_input.strip() != "" or "not feeling" in request.form:
        # If symptoms are provided, predict the disease
        if symptoms_input:
            predicted_disease = predict_disease(symptoms_input)
            response = {
                'response': f"ChatBot: I predict that the patient may have {predicted_disease}.",
                'symptoms': symptoms_input  # Include symptoms in the response
            }
        else:
            response = {'response': "ChatBot: I'm here to help! If you have symptoms, feel free to let me know."}

    else:
        # For general chatbot responses
        chatbot_response = get_chatbot_response(user_input_text)
        if chatbot_response:
            response = {'response': f"ChatBot: {chatbot_response}"}
        else:
            response = {'response': "ChatBot: I'm here to help! If you have symptoms, feel free to let me know."}

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)