import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

training = pd.read_csv('disease prediction\Data\Training.csv')
testing= pd.read_csv('disease prediction\Data\Testing.csv')
cols= training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']
y1= y

reduced_data = training.groupby(training['prognosis']).max()

#mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']  
testy    = le.transform(testy)

clf_knn = KNeighborsClassifier()
clf_knn.fit(x_train, y_train)

# Make predictions on the test set
y_pred = clf_knn.predict(x_test)

classification_rep= classification_report(y_test, y_pred)
print("K-Nearest Neighbors:")

print(clf_knn.score(x_test, y_test))
print("\nClassification Report:\n", classification_rep)

# import joblib

# # Save the model to a file
# joblib.dump(clf_knn, 'disease_prediction_model.joblib')

# # Load the model from the file
# loaded_model = joblib.load('knn_model.joblib')


# # Display the features used during training
# features_used = training.columns[:-1]
# print("Features Used During Training:", features_used)

# Display the names of all features
# all_features = training.columns.tolist()
# print("All Features:", all_features)

# curl -X POST -H "Content-Type: application/json" -d '{"symptoms": [" stomach_pain", " acidity", " ulcers_on_tongue", " vomiting", " cough", " chest_pain"]}' http://localhost:5000/predict
# Invoke-WebRequest -Uri http://localhost:5000/predict -Method Post -Body '{"symptoms": [" stomach_pain", " acidity", " ulcers_on_tongue", " vomiting", " cough", " chest_pain"]}' -ContentType "application/json"
# curl -Uri http://localhost:5000/predict -Method Post -Body '{"symptoms": [" stomach_pain", " acidity", " ulcers_on_tongue", " vomiting", " cough", " chest_pain"]}' -ContentType "application/json"

# vomiting, sunken_eyes, dehydration, diarrhoea
# itching, skin_rash, stomach_pain, burning_micturition, spotting_ urination