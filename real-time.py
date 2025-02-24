'''
# start streaming server

cd ~/mediamtx
./mediamtx


# start stream at- rtsp://localhost:8554/mystream

cd ~/EE297A/videos/
ffmpeg -re -stream_loop -1 -i 13151722_3840_2160_30fps.mp4 -c:v libx264 -preset fast -b:v 3000k -maxrate 5000k -bufsize 10000k -vf "scale=1280:720" -r 15 -an -f rtsp rtsp://localhost:8554/mystream

'''


import argparse
import cv2
import numpy as np
from flask import Flask, render_template, request, Response, flash
import os
from ultralytics import YOLO
from torch.cuda import is_available
from time import sleep
import psycopg2 as pg
from psycopg2.extras import DictCursor
import io
import signal

MODEL = YOLO("yolov8s.pt")
IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"}  # image suffixes
VID_FORMATS = {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"}  # video suffixes
COLORS = np.random.uniform(0, 255, size=(1000, 3)) # random colors for different classes
imgpath = None
device =  "cuda:0" if is_available() else "cpu"
video_path = None
source_link = None
its_image = False
PROJECT_PATH = "runs/detect"
os.makedirs(f"{PROJECT_PATH}", exist_ok=True)
DOWNLOADS_FOLDER = "uploads"
os.makedirs(f"{DOWNLOADS_FOLDER}", exist_ok=True)
app = Flask(__name__)
app.secret_key = 'ee_267_sjsu$2024'

# connect to the database
from yaml import safe_load
def load_db_config(filename="db_config.yaml"):
    with open(filename, "r") as file:
        config = safe_load(file)
    return config["database"]

conn = None
cur = None
# def connect_db():
# Load credentials
db_params = load_db_config()
try:
    conn = pg.connect(**db_params)
    cur = conn.cursor(cursor_factory=DictCursor)
    print(conn, cur)
    print("Connected to the database successfully!")
    # Create table if it does not exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id SERIAL PRIMARY KEY,
            image_name VARCHAR(255),
            boxes BYTEA,
            labels BYTEA,
            conf BYTEA,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    conn.commit()  # Commit table creation
except Exception as e:
    print("Error:", e)
    
def store_detection_data(image_name, boxes, labels, conf):
    global conn, cur  # Use global variables

    # Convert NumPy arrays to byte format
    def array_to_bytes(array):
        buf = io.BytesIO()
        np.save(buf, array)
        buf.seek(0)
        return buf.read()

    # Convert all three arrays to bytes
    boxes_bytes = array_to_bytes(boxes)
    labels_bytes = array_to_bytes(labels)
    conf_bytes = array_to_bytes(conf)

    # Insert the data into the database
    cur.execute("""
        INSERT INTO detections (image_name, boxes, labels, conf)
        VALUES (%s, %s, %s, %s);
    """, (image_name, boxes_bytes, labels_bytes, conf_bytes))

    conn.commit()
    # Print the number of rows inserted
    print(f"Inserted {cur.rowcount} row(s) into the detections")
    return

# @app.teardown_appcontext
# def close_db(error=None):
#     global conn, cur
#     if conn is not None:
#         cur.close()
#         conn.close()
#         print("Database connection closed.")


# Register signal handlers for graceful shutdown
def shutdown_signal_handler(signal, frame):
    print("Shutting down gracefully...")
    if conn is not None:
        cur.close()
        conn.close()
    exit(0)

# Register the signal handler for SIGINT (Ctrl+C) and SIGTERM (termination signal)
signal.signal(signal.SIGINT, shutdown_signal_handler)
signal.signal(signal.SIGTERM, shutdown_signal_handler)


@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/", methods=["GET", "POST"])
def predict_img():
    global imgpath, its_image, video_path, device, source_link, conn, cor
    video_path = None
    imgpath = None
    source_link = None
    its_image = False
    if request.method == "POST":
        if request.files:
            if len(request.files["file"].filename)==0:
                if 'text' in request.form:
                    source_link = request.form['text']
                    video_feed()
                    return render_template('index.html')
                else:
                    flash("Please upload a file first to proceed!")
                    return render_template('index.html')
            else:
                f = request.files['file']
                basepath = os.path.dirname(__file__)
                filepath = os.path.join(basepath,DOWNLOADS_FOLDER, f.filename)
                f.save(filepath)
                predict_img.imgpath = f.filename
                                                
                imgpath, file_extension = (f.filename.rsplit('.', 1)[0].lower(), f.filename.rsplit('.', 1)[1].lower())

                if os.path.exists(f"{PROJECT_PATH}"):
                    os.makedirs(f"{PROJECT_PATH}/{imgpath}", exist_ok=True)
                else:
                    os.makedirs(f"runs", exist_ok=True)
                    os.makedirs(f"{PROJECT_PATH}")
                    os.makedirs(f"{PROJECT_PATH}/{imgpath}")

                if file_extension in IMG_FORMATS:
                    if conn == None or cur == None:
                        flash("Error connecting to the database!")
                        return render_template('index.html')
                    its_image = True
                    frame = cv2.imread(filepath)
                    if frame is None:
                        flash("Could not read Image, Please upload a valid image file!")
                        return render_template('index.html')
                    # Inference
                    results = MODEL.predict(
                        frame, show=False, verbose=False, save=False, device=device, conf=0.5
                    )

                    # Check if robot is detected
                    if results[0].boxes.cpu().numpy().xyxy.shape[0] != 0:
                        # Show results on image
                        boxes = results[0].boxes.cpu().numpy().xyxy.astype(int)
                        labels = results[0].boxes.cpu().numpy().cls
                        conf = results[0].boxes.cpu().numpy().conf
                        store_detection_data(f.filename, boxes, labels, conf)
                        for box, label, conf in zip(boxes, labels, conf):
                            x1, y1, x2, y2 = box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[int(label)], 4)
                            cv2.putText(
                                frame,
                                MODEL.names[int(label)] + ": " + str(round(conf, 2)),
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                4,
                                COLORS[int(label)],
                                4,
                            )
                    cv2.imwrite(f"{PROJECT_PATH}/{imgpath}/output.jpg", frame)
                    image_feed()
                    return render_template('index.html')  
                
                elif file_extension in VID_FORMATS:
                    its_image = False
                    video_path = filepath
                    video_feed()
                    return render_template('index.html')

        if request.form:
            if len(request.form["text"])==0:
                flash("Please provide a link first to proceed!")
                return render_template('index.html')
            else:
                if 'text' in request.form:
                    source_link = request.form['text']
                    video_feed()
                    return render_template('index.html')

    return render_template('index.html')


def get_video_frame():
    global imgpath, video_path, device, source_link

    if video_path!=None and source_link==None:
        pth = video_path
        video_path = None
        imgpath = None
        cap = cv2.VideoCapture(pth)
        if not cap.isOpened():
            flash("Error opening video file, please upload a valid video!")
            return render_template('index.html')
        # Initialize variables
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(
            f"{PROJECT_PATH}/{imgpath}/output.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Inference
            results = MODEL.predict(
                frame, show=False, verbose=False, save=False, device=device, conf=0.5
            )
            sleep(0.1)
            # Check if robot is detected
            if results[0].boxes.cpu().numpy().xyxy.shape[0] != 0:
                # Show results on image
                boxes = results[0].boxes.cpu().numpy().xyxy.astype(int)
                labels = results[0].boxes.cpu().numpy().cls
                conf = results[0].boxes.cpu().numpy().conf
                for box, label, conf in zip(boxes, labels, conf):
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[int(label)], 2)
                    cv2.putText(
                        frame,
                        MODEL.names[int(label)] + ": " + str(round(conf, 2)),
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        COLORS[int(label)],
                        2,
                    )
            out.write(frame)
            cv2.imwrite(f"{PROJECT_PATH}/{imgpath}/output.jpg", frame)
        
            _,jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        
        cap.release()
        out.release()
    
    elif source_link!=None:
        print("source_link: ", source_link)
        # Inference
        pth = source_link
        source_link = None
        results = MODEL.predict(
            pth, show=False, verbose=False, save=False, device=device, conf=0.5, stream=True
        )
        # sleep(0.1)
        while True:
            for result in results:
                frame = result.orig_img
                # Show results on image
                boxes = result.boxes.cpu().numpy().xyxy.astype(int)
                labels = result.boxes.cpu().numpy().cls
                conf = result.boxes.cpu().numpy().conf
                for box, label, conf in zip(boxes, labels, conf):
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[int(label)], 2)
                    cv2.putText(
                        frame,
                        MODEL.names[int(label)] + ": " + str(round(conf, 2)),
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        COLORS[int(label)],
                        2,
                    )
                h,w,_ = frame.shape
                if h>w:
                    frame = cv2.resize(frame, (240, 320))
                _,jpeg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
    else: # clear the image
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n\r\n')

def get_image_frame():
    global imgpath, its_image
    if its_image:
        img_files = f"{PROJECT_PATH}/{imgpath}/output.jpg"
        image = cv2.imread(img_files)
        _, jpeg = cv2.imencode('.jpg', image) 
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
    else: # clear the image
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n\r\n')
    imgpath = None
    its_image = False

@app.route("/image_feed")
def image_feed():
    return Response(get_image_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# function to display the detected objects video on html page
@app.route("/video_feed")
def video_feed():
    return Response(get_video_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov8 models")
    parser.add_argument("--port", default=8000, type=int, help="port number")
    args = parser.parse_args()
    # Start the Flask app in a separate thread
    app.run(host='0.0.0.0', port= args.port, debug=True, threaded=True)