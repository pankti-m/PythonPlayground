import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv('text_classification_data.csv')
x = data['text']
y = data['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Vectorize the data
vectorizer = CountVectorizer()
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

# Train the data
model = MultinomialNB()
model.fit(x_train_vectorized, y_train)

# Predict using the above model
y_pred = model.predict(x_test_vectorized)

# Calculate the accuracy and confusion matrix of the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy *100}%')
class_labels = np.unique(y_test)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix HeatMap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

#Prediction on Unseen Data
user_input = 'I like programming'
user_input_vectorized = vectorizer.transform([user_input])
predicted_label = model.predict(user_input_vectorized)
print(f"Input Text = '{user_input}'")
print(f"Category Predicted = '{predicted_label[0]}'")

user_input = 'The next summer Olympics will be in 2028'
user_input_vectorized = vectorizer.transform([user_input])
predicted_label = model.predict(user_input_vectorized)
print(f"Input Text = '{user_input}'")
print(f"Category Predicted = '{predicted_label[0]}'")

user_input = 'The US President will hand over the Oscar Trophy this year.'
user_input_vectorized = vectorizer.transform([user_input])
predicted_label = model.predict(user_input_vectorized)
print(f"Input Text = '{user_input}'")
print(f"Category Predicted = '{predicted_label[0]}'")
