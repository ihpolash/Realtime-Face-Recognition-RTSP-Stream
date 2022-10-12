from flask import Flask, render_template, Response, request
from live_stream import FaceRecognition
from face_detect import face_detect
# from predict_expression import predict_expression
from PIL import Image, ImageOps
import cv2

app = Flask(__name__)

# front_cam = "rtsp://admin:ismail.2022@71.187.210.42:5541/cam/realmonitor?channel=1&subtype=0"
# back_cam = "rtsp://admin:ismail.2022@71.187.210.42:5542/cam/realmonitor?channel=1&subtype=0"

front_cam = 0
back_cam = 1

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('front_cam.html')

@app.route('/back', methods=['GET', 'POST'])
def index_back():
    return render_template('back_cam.html')

def gen(FaceRecognition):
    while True:
        frame = FaceRecognition.face_recognition()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/front_cam')
def video_feed():
    return Response(gen(FaceRecognition(front_cam)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/back_cam')
def video_feed_back():
    return Response(gen(FaceRecognition(back_cam)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/face_detect", methods=['GET', 'POST'])
def result():
	return render_template("home.html")

@app.route("/submit", methods = ['GET', 'POST'])
def face_detect_api():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename	
        img.save(img_path)

        name, fr, last_entry = face_detect(img_path)
        cv2.imwrite(img_path, fr)
        
    return render_template("home.html", prediction = name, img_path = img_path, last_entry= last_entry)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
