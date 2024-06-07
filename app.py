from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import time
import numpy as np
from model import detect_and_track_cars, model, zone, active_cars, timers, alarms

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    video_file = request.files.get('videoFile')
    if video_file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        video_file.save(file_path)
        return jsonify({'video_source': file_path})
    else:
        video_source = request.form.get('videoSource')
        return jsonify({'video_source': video_source})

def gen_frames(source):
    cap = cv2.VideoCapture(source)
    window_width, window_height = 800, 600

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (window_width, window_height))
        frame = detect_and_track_cars(frame, model, zone, active_cars, timers, alarms)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    video_source = request.args.get('video_source', default='', type=str)
    return Response(gen_frames(video_source), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
