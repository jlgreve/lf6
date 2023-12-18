import logging
import uuid

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
phone_number = "+49 1910 1217"
ticket_number = 100

app: Flask = Flask(__name__)


# TODO imlement max size of message
@app.route('/prompt/<message>')
def endpoint_prompt(message) -> str:
    return get_gpt_response(message)


@app.route('/')
def endpoint_home() -> str:
    return render_template('index.html', chat_history=glob_chat_history)


@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback = request.form
    print(feedback)
    thank_you_message = "Thank you! Your feedback has been submitted."
    bot_msg_time = datetime.now().strftime("%d/%m/%Y | %H:%M:%S")
    return render_template('index.html', chat_history=glob_chat_history,
                           type="feedback_submitted", bot_msg_time=bot_msg_time, thank_you_message=thank_you_message)


# For each message in chat_history:
# Always a pair of user input & bot response
@app.route('/chat', methods=['POST'])
def endpoint_chat():
    global ticket_number
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
    if support_level == 2:
        bot_response = ("I am glad i was able to help you. Please feel free to tell us how you felt about my support "
                        "so we are able to improve our services!")
    else:
        bot_response = (f"I apologize for the inconvenience, but I am not able to understand your request. Please feel "
                        f"free to contact us under <b>{phone_number}</b> and one of our employees will support you "
                        f"with your problem. To speed things up, please note down your ticket number: #"
                        f"<b>{ticket_number:06d}</b> so we have an easier time to find your request. We look "
                        f" forward to be hearing from you!")
        ticket_number += 1
    bot_msg_time = datetime.now().strftime("%d/%m/%Y | %H:%M:%S")
    glob_chat_history.append(
        {'user': user_input, 'bot': bot_response, 'user_time': user_msg_time, 'bot_time': bot_msg_time}
    )
    return render_template('index.html', chat_history=glob_chat_history,
                           type="feedback") if support_level == 2 else redirect('/')


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
