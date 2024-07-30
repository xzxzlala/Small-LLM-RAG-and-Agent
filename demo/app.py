from flask import Flask, request, jsonify, render_template
from chatbot import ChatBot  # Import your ChatBot class

app = Flask(__name__)
chatbot = ChatBot()  # Initialize your chatbot

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    step = request.form.get('step')
    additional_input = request.form.get('additional_input')
    if step is not None and step != "":
        additional_input = user_input
    print("user add input")
    print(user_input)
    print(additional_input)
    response, next_step = chatbot.chat(user_input, step, additional_input)

    return jsonify({'response': response, 'next_step': next_step})

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port = 5002)
