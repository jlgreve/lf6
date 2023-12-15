import logging
from flask import Flask, request, render_template
from gpt import get_gpt_response
import pickle

app = Flask(__name__)

# Set up logging to the console
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])


@app.route('/prompt/<message>')
def process_prompt(message):
    return get_gpt_response(message)


chat_history = []


@app.route('/')
def home():
    return render_template('index.html', chat_history=chat_history)


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']

    # Classify support level
    support_level = classify_level(user_input)

    if support_level == 0:
        try:
            bot_response = get_gpt_response(user_input)
        except Exception as e:
            # Log error
            logging.error(f"Error processing user input: {e}")
            # Set bot_response to a default error message
            bot_response = f"Sorry, an error occurred. Please try again later."
    else:
        bot_response = ("I apologize for any inconvenience you're experiencing. It seems that your issue requires the"
                        " attention of our first-level support team. Please provide us with your contact number,"
                        "and a support representative will get in touch with you shortly to assist you further. "
                        "Thank you for your understanding.")

    chat_history.append({'user': user_input, 'bot': bot_response})
    return render_template('index.html', chat_history=chat_history)


def classify_level(input):
    with open('../model/model.pkl', 'rb') as file:
        classifier = pickle.load(file)
    with open('../model/vectorizer.pkl', 'rb') as file:
        tfidf_vectorizer = pickle.load(file)

    new_statements_tfidf = tfidf_vectorizer.transform([input])
    return classifier.predict(new_statements_tfidf)[0]


if __name__ == '__main__':
    app.run()
