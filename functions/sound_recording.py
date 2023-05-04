import sounddevice
from scipy.io.wavfile import write

sr = 44100  # Sample rate
seconds = 5  # Audio recorded for 5 seconds


def start_rec(sr=sr, seconds=seconds):
    print("Recording...")
    record_voice = sounddevice.rec(int(seconds * sr), samplerate=sr, channels=1)
    sounddevice.wait()
    write("static/output/test_input.wav", sr, record_voice)