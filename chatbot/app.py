from flask import Flask
from gpt import get_gpt_response
app = Flask(__name__)

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route('/prompt/<message>')
def process_prompt(message):
    return get_gpt_response(message)

if __name__ == '__main__':
    app.run()
