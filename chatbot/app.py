from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route('/prompt')
def process_prompt():
    return

if __name__ == '__main__':
    app.run()
