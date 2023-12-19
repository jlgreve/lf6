import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Define dataset
data = pd.read_csv('dataset.csv')
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['statement'], data['support_level'], test_size=0.2,
                                                    random_state=42)

# Create TF-IDF vectors for training data
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Preprocess the test data using the same vectorizer
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Make predictions on the test set
y_pred = classifier.predict(X_test_tfidf)

# Evaluate the models performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ----------------------- model in prod ----------------------- #
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_data_tfidf = tfidf_vectorizer.fit_transform(data['statement'])
classifier = MultinomialNB()
classifier.fit(X_data_tfidf, data['support_level'])

# Save the trained model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(classifier, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)