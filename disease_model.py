from joblib import dump, load
import pandas as pd
import sklearn.preprocessing

# Load the training data
training = pd.read_csv('disease prediction/Data/Training.csv')

# Load the saved model
loaded_model = load('disease_prediction_model.joblib')

# Define cols based on the training data
cols = training.columns[:-1]

# Load the LabelEncoder used during training
le = sklearn.preprocessing.LabelEncoder()
le.fit(training['prognosis'])

# Get user input for symptoms as text
user_input_text = input("Enter the patient's symptoms, separated by commas: ")
user_input_symptoms = set(user_input_text.lower().split(','))

# Create a new list to store user symptoms as 1 or 0
user_symptoms = [1 if symptom.lower() in user_input_symptoms else 0 for symptom in cols]

# Preprocess user input (make sure it matches the format used during training)
user_input = pd.DataFrame([user_symptoms], columns=cols)

# Make predictions using the loaded model
user_prediction = loaded_model.predict(user_input)

# Decode the predicted label back to the original class
predicted_disease = le.inverse_transform(user_prediction)

print("Predicted Disease:", predicted_disease[0])
