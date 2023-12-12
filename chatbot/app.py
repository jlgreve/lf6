import logging

from flask import Flask, request, render_template
from gpt import get_gpt_response

app = Flask(__name__)

# Set up logging to the console
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])


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
    try:
        bot_response = get_gpt_response(user_input)
    except Exception as e:
        # Log error
        logging.error(f"Error processing user input: {e}")
        # Set bot_response to a default error message
        bot_response = f"Sorry, an error occurred. Please try again later."

    chat_history.append({'user': user_input, 'bot': bot_response})
    return render_template('index.html', chat_history=chat_history)


if __name__ == '__main__':
    app.run()
