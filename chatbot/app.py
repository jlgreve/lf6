import logging
import subprocess
import time
from datetime import datetime
from typing import List, Dict, Any

import flask
from flask import Flask, request, render_template, redirect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sqlalchemy import Engine

import orm
from common.get_logger import get_logger
from gpt import get_gpt_response
from orm import ChatStatus, ChatFeedback, ChatHistory, ChatMessage, ChatStatusEnum
from util import yaml_from_file, pickle_from_file

# Global variables
glob_config: dict[str, dict] = {}
glob_chat_history: list[dict] = []
glob_classifier: MultinomialNB = None
glob_tfidf_vectorizer: TfidfVectorizer = None
glob_db_engine: Engine = None

phone_number = "+49 1910 1217"
ticket_number = 100

app: Flask = Flask(__name__)


def get_chat_history(chat_id: int) -> List[Dict[str, Any]]:
    chat: ChatHistory = ChatHistory.query.filter_by(id=chat_id).one_or_none()

    messages: List[ChatMessage] = chat.messages

    return [{
        'time': datetime.fromtimestamp(message.time_sent / 1e9).strftime("%d/%m/%Y | %H:%M:%S"),
        'from_support': message.from_support,
        'content': message.content
    } for message in messages]


def create_chat_history() -> ChatHistory:
    new_history = ChatHistory()
    orm.db.session.add(new_history)
    orm.db.session.commit()

    new_status = ChatStatus(
        chat_id=new_history.id,
        status=ChatStatusEnum.started,
        time_reached=time.time_ns(),
        active=True
    )
    orm.db.session.add(new_status)
    orm.db.session.commit()

    return new_history


def create_chat_feedback(chat_id: int, feedback: int):
    new_feedback = ChatFeedback(
        id=chat_id,
        stars=feedback
    )
    orm.db.session.add(new_feedback)
    orm.db.session.commit()


def add_chat_message(chat_id: int, from_support: bool, content: str) -> ChatMessage:
    new_message = orm.ChatMessage(
        chat_id=chat_id,
        from_support=from_support,
        time_sent=time.time_ns(),
        content=content
    )
    orm.db.session.add(new_message)
    orm.db.session.commit()

    return new_message


def get_chat_status(chat_id: int) -> ChatStatusEnum:
    chat_status: ChatStatus = ChatStatus.query.filter_by(id=chat_id, active=True).one_or_none()
    return chat_status.status if chat_status is not None else 0


def change_chat_status(chat_id: int, new_status: ChatStatusEnum) -> ChatStatus:
    ChatStatus.query.filter_by(chat_id=chat_id, active=True).update({'active': False})

    new_status = ChatStatus(
        chat_id=chat_id,
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


@app.route('/submit_feedback/<int:chat_id>', methods=['POST'])
def submit_feedback(chat_id: int):
    feedback = int(request.form["feedback"])
    create_chat_feedback(chat_id, feedback)

    change_chat_status(chat_id, ChatStatusEnum.ended)
    add_chat_message(chat_id, True, "Thank you for submitting your feedback!")

    return redirect(f'/chat/{chat_id}')


def handle_user_input(chat_id: int, user_input: str):
    if chat_id is None or user_input is None or len(user_input) == 0:
        return

    add_chat_message(chat_id, False, user_input)

    # Classify support level
    support_level = classify_level(user_input)
    if support_level == 0:
        try:
            add_chat_message(chat_id, True, get_gpt_response(user_input))
        except Exception as e:
            # Log error
            log.error(f"Error processing user input: {e}")
            # Set bot_response to a default error message
            add_chat_message(chat_id, True, "Sorry, an error occurred. Please try again later.")
    elif support_level == 2:
        add_chat_message(chat_id, True,
                         ("I am glad i was able to help you. Please feel free to tell us how you felt about my support "
                          "so we are able to improve our services!"))

        change_chat_status(chat_id, ChatStatusEnum.pending_feedback)
    else:
        add_chat_message(chat_id, True, (
            f"I apologize for the inconvenience, but I am not able to understand your request. Please feel "
            f"free to contact us under <b>{phone_number}</b> and one of our employees will support you "
            f"with your problem. To speed things up, please note down your ticket number: #"
            f"<b>{chat_id:06d}</b> so we have an easier time to find your request. We look "
            f" forward to be hearing from you!"))

        add_chat_message(chat_id, True,
                         (
                             "I am sorry for not being able to help you. Please feel free to tell us how you felt about my support "
                             "so we are able to improve our services!"))

        change_chat_status(chat_id, ChatStatusEnum.support_escalated)
        change_chat_status(chat_id, ChatStatusEnum.pending_feedback)

@app.route('/chat', methods=['GET', 'POST'])
def endpoint_chat_no_id():
    if request.method == 'GET' or request.form['user_input'] is None:
        return render_template('index.html', chat_history=[])

    new_history = create_chat_history()
    handle_user_input(new_history.id, request.form['user_input'])

    return redirect(f'/chat/{new_history.id}')


# For each message in chat_history:
# Always a pair of user input & bot response
@app.route('/chat/<int:chat_id>', methods=['GET', 'POST'])
def endpoint_chat_with_id(chat_id: int):
    log.info(f'Got a {request.method} request for the chat history with the id {chat_id}')

    if request.method == 'GET':
        return render_template('index.html',
                               chat_history=get_chat_history(chat_id),
                               chat_status=get_chat_status(chat_id), id=chat_id)

    user_input: str = request.form['user_input']

    if user_input is None or len(user_input) == 0:
        log.info('Empty user input.')
        return render_template('index.html',
                               chat_history=get_chat_history(chat_id),
                               chat_status=get_chat_status(chat_id), id=chat_id)

    handle_user_input(chat_id, user_input)
    return render_template('index.html', chat_history=get_chat_history(chat_id),
                           chat_status=get_chat_status(chat_id), id=chat_id)


def classify_level(enquiry: str):
    new_statements_tfidf = glob_tfidf_vectorizer.transform([enquiry])
    return glob_classifier.predict(new_statements_tfidf)[0]


def execute_file(file_path, log):
    completed_process = subprocess.run(['python', file_path], capture_output=True, text=True)
    if completed_process.returncode == 0:
        log.info(f"Executed {file_path} successfully.")
    else:
        log.debug(f"Error: Failed to execute '{file_path}'. "
                  f"Error output: {completed_process.stderr}")


if __name__ == '__main__':
    log = get_logger("Chatbot", level='debug')

    # Execute model training
    #execute_file('../model/main.py', log)
    # Load config
    glob_config = yaml_from_file('config.yaml')

    # Init DB
    app.config["SQLALCHEMY_DATABASE_URI"] = glob_config['database']['url']
    orm.db.init_app(app)

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
        log.error(
            'Fatal Error!\n'
            'The classification model has not been generated.\n'
            'To generate the model, navigate the terminal to "lf6/model/" and run: python main.py'
        )
        exit(1)
    except Exception as ex:
        log.error(
            'Fatal Error!\n'
            'An unexpected exception has occurred while loading the classification model.\n'
            f'Exception: {ex}'
        )
        exit(1)

    with app.app_context():
        orm.db.create_all()

    # Start WebApp
    app.run()
