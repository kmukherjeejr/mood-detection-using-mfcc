from flask import Flask, render_template, request, redirect, url_for

import glob
import os

import librosa
import librosa.display
import numpy as np
import soundfile
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from functions.sound_recording import start_rec

app = Flask(__name__)

sr, seconds = 44100, 5


def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result


emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

observed_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']


def load_data(test_size=0.2):
    x, y = [], []
    for file in glob.glob("D:\\Downloads\\Audio_Speech_Actors_01-24\\Actor_*\\macro-output\\*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


"""## 4. Loading test and train data"""

x_train, x_test, y_train, y_test = load_data(test_size=0.25)

"""## 5. MLP Classifier"""

model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive',
                      max_iter=500)

"""### 5.1 Training the model"""

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy * 100))


def new_result(file='E:/Documents/PyCharmProjects/mood_detection_using_mfcc/static/output/test_input.wav'):
    test = np.array(
        extract_feature(file,
                        mfcc=True, chroma=True, mel=True)).reshape(1, -1)
    y_pred = model.predict(test)
    print(y_pred[0])
    return y_pred


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', mood='????')


@app.route('/start_rec', methods=['GET', 'POST'])
def start_rec_browser():
    start_rec(sr=sr, seconds=seconds)
    return render_template('index.html', mood=str(new_result()[0]))


if __name__ == "__main__":
    app.run(debug=True)
