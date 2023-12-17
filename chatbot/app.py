import logging

from gpt import get_gpt_response
from util import yaml_from_file, pickle_from_file
from datetime import datetime
from flask import Flask, request, render_template, redirect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Global variables
glob_config: dict[str, dict] = {}
glob_chat_history: list[dict] = []
glob_classifier: MultinomialNB = None
glob_tfidf_vectorizer: TfidfVectorizer = None

app: Flask = Flask(__name__)


@app.route('/prompt/<message>')
def endpoint_prompt(message) -> str:
    return get_gpt_response(message)


@app.route('/')
def endpoint_home() -> str:
    return render_template('index.html', chat_history=glob_chat_history)


# For each message in chat_history:
# Always a pair of user input & bot response
@app.route('/chat', methods=['POST'])
def endpoint_chat():
    user_msg_time = datetime.now().strftime("%d/%m/%Y | %H:%M:%S")

    user_input: str = request.form['user_input']

    # Classify support level
    support_level = classify_level(user_input)

    if support_level == 0:
        try:
            bot_response = get_gpt_response(user_input)
        except Exception as e:
            # Log error
            logging.error(f"Error processing user input: {e}")
            # Set bot_response to a default error message
            bot_response = "Sorry, an error occurred. Please try again later."
    else:
        bot_response = ("I apologize for any inconvenience you're experiencing. It seems that your issue requires the"
                        " attention of our first-level support team. Please provide us with your contact number,"
                        "and a support representative will get in touch with you shortly to assist you further. "
                        "Thank you for your understanding.")

    bot_msg_time = datetime.now().strftime("%d/%m/%Y | %H:%M:%S")
    glob_chat_history.append(
        {'user': user_input, 'bot': bot_response, 'user_time': user_msg_time, 'bot_time': bot_msg_time}
    )

    return redirect('/')


def classify_level(enquiry: str):
    new_statements_tfidf = glob_tfidf_vectorizer.transform([enquiry])

    return glob_classifier.predict(new_statements_tfidf)[0]


if __name__ == '__main__':
    # Load config
    glob_config = yaml_from_file('config.yaml')

    # Set up logging to the console
    logging.basicConfig(
        level=glob_config['logging']['level'],
        format=glob_config['logging']['format'],
        handlers=[logging.StreamHandler()]
    )

    # Load support level classifier model
    try:
        glob_classifier = pickle_from_file(glob_config['models']['classifier'])
        glob_tfidf_vectorizer = pickle_from_file(glob_config['models']['vectorizer'])
    except FileNotFoundError:
        logging.error(
            'Fatal Error!\n'
            'The classification model has not been generated.\n'
            'To generate the model, navigate the terminal to "lf6/model/" and run: python main.py'
        )
        exit(1)
    except Exception as ex:
        logging.error(
            'Fatal Error!\n'
            'An unexpected exception has occurred while loading the classification model.\n'
            f'Exception: {ex}'
        )
        exit(1)

    # Start WebApp
    app.run()
