import nltk
from nltk.chat.util import Chat, reflections
from joblib import dump, load
import pandas as pd
import sklearn.preprocessing
import json
import random

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

# Function to interact with the chatbot
def chatbot():
    print("Hi there! I'm ChatBot. Let's have a chat.")
    print("Type 'quit' to exit.")

    while True:
        user_input_text = input("user (Type 'quit' to exit): ")

        if user_input_text.lower() == 'quit':
            print("Goodbye! Have a great day.")
            break

        elif "symptoms" in user_input_text.lower() or "not feeling" in user_input_text.lower():
            symptoms_input = input("Enter symptoms separated by commas: ")
            predicted_disease = predict_disease(symptoms_input)
            print(f"ChatBot: I predict that the patient may have {predicted_disease}.")

        else:
            chatbot_response = get_chatbot_response(user_input_text)
            if chatbot_response:
                print(f"ChatBot: {chatbot_response}")
            else:
                print("ChatBot: I'm here to help! If you have symptoms, feel free to let me know.")

if __name__ == "__main__":
    chatbot()