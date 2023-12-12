import logging

from flask import Flask, request, render_template
from gpt import get_gpt_response

app = Flask(__name__)


@app.route('/prompt/<message>')
def process_prompt(message):
    return get_gpt_response(message)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    try:
        bot_response = get_gpt_response(user_input)
    except Exception as e:
        # Log error
        print(f"Error processing user input: {e}")
        # Set bot_response to a default error message
        bot_response = f"Sorry, an error occurred. Please try again later."
    return render_template('index.html', user_input=user_input, bot_response=bot_response)


if __name__ == '__main__':
    app.run()
