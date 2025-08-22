from flask import Flask, render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start')
def start_detection():
    subprocess.Popen(["python", "driver_drowsiness.py"])  # your script name
    return "Drowsiness Detection Started!"

if __name__ == "__main__":
    app.run(debug=True)
