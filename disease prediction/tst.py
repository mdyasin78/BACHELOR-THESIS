import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Read the training and testing data
training = pd.read_csv('disease prediction\Data\Training.csv')
testing = pd.read_csv('disease prediction\Data\Testing.csv')

cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

# Introduce random noise to features
noise_factor = 0.05
x_noisy = x + np.random.normal(0, noise_factor, x.shape)

# Mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_noisy, y, test_size=0.33, random_state=42)

# Read the testing data and map strings to numbers
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

# Initialize and train the K-Nearest Neighbors classifier
clf_knn = KNeighborsClassifier()
clf_knn.fit(x_train, y_train)

# Make predictions on the test set
y_pred = clf_knn.predict(x_test)

# Evaluate the model
classification_rep = classification_report(y_test, y_pred)
print("K-Nearest Neighbors:")
print(clf_knn.score(x_test, y_test))
print("\nClassification Report:\n", classification_rep)





# import pandas as pd

# # Read the CSV file into a DataFrame
# df = pd.read_csv(r'C:\Users\moham\Music\chatbot project\diseases\disease prediction\Data\dataset.csv')

# # # Get unique values in the first column
# # first_column_unique_values = df.iloc[:, 0].unique()

# # fco=first_column_unique_values.value_counts()

# # # Print or use the unique values as needed
# # print(first_column_unique_values)
# # print(fco)

# # # Get value counts for the first column
# # first_column_value_counts = df.iloc[:, 0].value_counts()

# # # Print or use the value counts as needed
# # print(first_column_value_counts)


# # Get the count of unique values in the first column
# unique_count = df.iloc[:, 0].nunique()

# # Print or use the unique count as needed
# print(unique_count)
