import pickle

from flask import Flask, render_template

from functions.sound_recording import start_rec
from functions.result import new_result

app = Flask(__name__)

sr, seconds = 44100, 5


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', mood='????')


@app.route('/start_rec', methods=['GET', 'POST'])
def start_rec_browser():
    start_rec(sr=sr, seconds=seconds)
    return render_template('index.html', mood=str(new_result(pickled_model=pickled_model)[0]))


if __name__ == "__main__":
    pickled_model = pickle.load(open('trained_model\model.pkl', 'rb'))
    app.run(host="0.0.0.0", port=80, debug=True)
