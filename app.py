from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('models/asl-yolov8n-cls.pt')
camera = cv2.VideoCapture(0)
latest_prediction = {"letter": "nothing", "probability": 0.0}

def generate_frames():
    global latest_prediction
    while True:
        success, frame = camera.read()
        if not success:
            break
        print(frame.shape)
        frame = cv2.flip(frame, 1)
        
        new_results = model.predict(frame, verbose=False)
        class_key = new_results[0].probs.top1
        prob = float("{:.2f}".format(new_results[0].probs.top1conf.item()))
        prob = 0.0 if prob < 0.7 else prob
        letter = new_results[0].names[class_key] if prob > 0.7 else "nothing"
        latest_prediction = {"letter": letter, "probability": prob}
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_prediction')
def current_prediction():
    return jsonify(latest_prediction)

if __name__ == '__main__':
    app.run(port=8080, debug=True)