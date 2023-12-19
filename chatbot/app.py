import logging
import time
import uuid

import flask
from sqlalchemy import select, Engine, update
from sqlalchemy.orm import Session

import chatbot.orm as orm

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
glob_db_engine: Engine = None

phone_number = "+49 1910 1217"
ticket_number = 100

app: Flask = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///project2.db"
orm.db.init_app(app)


def get_chat_history(chat_id: int) -> orm.ChatHistory:
    chat: orm.ChatHistory = orm.ChatHistory.query.filter_by(id=chat_id).one_or_none()

    history = chat.messages

    print("History:")
    for x in history:
        print(f"A: {x}")

    return chat


def create_chat_history() -> orm.ChatHistory:
    new_history = orm.ChatHistory()
    orm.db.session.add(new_history)
    orm.db.session.commit()

    new_status = orm.ChatStatus(
        chat_id=new_history.id,
        status=orm.ChatStatusEnum.started,
        time_reached=time.time_ns(),
        active=True
    )
    orm.db.session.add(new_status)
    orm.db.session.commit()

    return new_history


def add_chat_message(chat_history: orm.ChatHistory, from_support: bool, content: str) -> orm.ChatMessage:
    new_message = orm.ChatMessage(
        chat_id=chat_history.id,
        from_support=from_support,
        time_sent=time.time_ns(),
        content=content
    )
    orm.db.session.add(new_message)
    orm.db.session.commit()

    return new_message


def change_chat_status(chat_history: orm.ChatHistory, new_status: orm.ChatStatusEnum) -> orm.ChatStatus:
    orm.db.session.execute(
        update(orm.ChatStatus).
        where(orm.ChatStatus.chat_id == chat_history.id).
        where(orm.ChatStatus.active is True).
        values(active=False)
    )

    print("Old Statuses:")
    for old_status in chat_history.statuses:
        print(old_status)

    new_status = orm.ChatStatus(
        chat_id=chat_history.id,
        status=new_status,
        time_reached=time.time_ns(),
        active=True
    )
    orm.db.session.add(new_status)
    orm.db.session.commit()

    return new_status


# TODO imlement max size of message
@app.route('/prompt/<message>')
def endpoint_prompt(message) -> str:
    return get_gpt_response(message)


@app.route('/')
def endpoint_index() -> flask.Response:
    return redirect('/chat')


@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback = int(request.form["feedback"])
    thank_you_message = "Thank you! Your feedback has been submitted."
    bot_msg_time = datetime.now().strftime("%d/%m/%Y | %H:%M:%S")
    return render_template('index.html', chat_history=glob_chat_history,
                           support_level=3, bot_msg_time=bot_msg_time, thank_you_message=thank_you_message)


@app.route('/chat', methods=['GET', 'POST'])
def endpoint_chat_no_id():
    if request.method == 'GET':
        return render_template('index.html', chat_history=[])

    new_history = create_chat_history()
    add_chat_message(new_history, False, request.form['user_input'])

    return redirect(f'/chat/{new_history.id}')


# For each message in chat_history:
# Always a pair of user input & bot response
@app.route('/chat/<int:chat_id>', methods=['GET', 'POST'])
def endpoint_chat_with_id(chat_id: int):
    if request.method == 'GET':
        history = get_chat_history(chat_id)

        print(f"History: {history}")
        for message in history.messages:
            print(f"Message: {message}")

        return render_template('index.html', chat_history=history.messages)

    global ticket_number
    feedback_message = ""
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
    elif support_level == 2:
        bot_response = ("I am glad i was able to help you. Please feel free to tell us how you felt about my support "
                        "so we are able to improve our services!")
    else:
        bot_response = (f"I apologize for the inconvenience, but I am not able to understand your request. Please feel "
                        f"free to contact us under <b>{phone_number}</b> and one of our employees will support you "
                        f"with your problem. To speed things up, please note down your ticket number: #"
                        f"<b>{ticket_number:06d}</b> so we have an easier time to find your request. We look "
                        f" forward to be hearing from you!")
        feedback_message = (
            "I am sorry for not being able to help you. Please feel free to tell us how you felt about my support "
            "so we are able to improve our services!")
        ticket_number += 1
    bot_msg_time = datetime.now().strftime("%d/%m/%Y | %H:%M:%S")
    glob_chat_history.append(
        {'user': user_input, 'bot': bot_response, 'user_time': user_msg_time, 'bot_time': bot_msg_time}
    )
    return render_template('index.html', chat_history=glob_chat_history,
                           support_level=support_level, bot_msg_time=bot_msg_time, feedback_message=feedback_message)


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

    with app.app_context():
        orm.db.create_all()

    # Start WebApp
    app.run()
