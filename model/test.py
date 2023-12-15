import pickle

with open('../../chatbot/model.pkl', 'rb') as file:
    classifier = pickle.load(file)

with open('../../chatbot/vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

statement = input("Give an input")
new_statements_tfidf = tfidf_vectorizer.transform([statement])

# Make predictions
predictions = classifier.predict(new_statements_tfidf)

# Print the predictions
print(predictions)
