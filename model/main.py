import pandas as pd
import os
import pickle
from common.get_logger import get_logger
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import local_config as cf

log = get_logger("Classifier", level='debug')
# Define dataset
log.info(f"Loading dataset from {cf.DATASET_PATH}")
data = pd.read_csv(cf.DATASET_PATH)
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
log.info(f"Accuracy: {round(accuracy_score(y_test, y_pred) * 100, 2)}%")
# print("Classification Report:\n", classification_report(y_test, y_pred))

# ----------------------- model in prod ----------------------- #
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_data_tfidf = tfidf_vectorizer.fit_transform(data['statement'])
classifier = MultinomialNB()
classifier.fit(X_data_tfidf, data['support_level'])

# Save the trained model to a file
if os.path.exists(cf.CLASSIFIER_PATH):
    os.remove(cf.CLASSIFIER_PATH)
with open(cf.CLASSIFIER_PATH, 'wb') as f:
    pickle.dump(classifier, f)
log.info(f"Saved {str(classifier)} in {cf.CLASSIFIER_PATH}")

if os.path.exists(cf.VECTORIZER_PATH):
    os.remove(cf.VECTORIZER_PATH)
with open(cf.VECTORIZER_PATH, 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
log.info(f"Saved {str(tfidf_vectorizer)} in {cf.VECTORIZER_PATH}")
