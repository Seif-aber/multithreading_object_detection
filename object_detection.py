import cv2
import threading
import sqlite3
import os
import shutil
import queue
import json
import argparse
from ultralytics import YOLO
import pandas as pd


# Function to read frames from a video and put them into the frames queue (Thread 1)
def read_frames(video_path, save_images):
    # save_images : a boolean that indicates whether the frames will be stored in the disk
    cap = cv2.VideoCapture(video_path)
    id = 1 # We'll use an incremental ID for the images
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image_name = f"image_{id:03d}.jpg"
        frame_info = (id, image_name, frame) # we save the frame id, name and data
        frame_queue.put(frame_info)
        if save_images:
            cv2.imwrite(os.path.join(frames_path, image_name), frame)
        id+=1
    cap.release()

# Function for object detection 
def perform_object_detection(frame_info, save_images):
    if save_images:
        im_path = os.path.join(frames_path, frame_info[1])
        # Set verbose = False to hide prediction details
        detections =  model.predict(source=im_path, save=True, project="Results", name="Images", exist_ok=True, verbose = True)[0].boxes
    else:
        detections =  model.predict(source=frame_info[-1], save=False, verbose = True)[0].boxes
    # Since the model prediction gives lot of information, we choose : classes, confidences and bounding boxes details
    # We use tolist() method to convert Pytorch tensors to lists (this is useful to convert detections to json format)
    return {
        "classes" : detections.cls.tolist(),
        "confs" : detections.conf.tolist(),
        "bbox" : detections.xyxy.tolist() # the xyxy format : (x_min, y_min, x_max, y_max)
    }
    

# Function to perform object detection and put results into the detection results queue (Thread 2)
def detect_objects(save_images):   
    while True:
        frame_info = frame_queue.get()
        if frame_info is None:
            break
        result = perform_object_detection(frame_info, save_images)
        new_frame_info = (frame_info[0], frame_info[1], result) # we save id, image_name and detections
        detection_results_queue.put(new_frame_info)
    
# Function to insert results into the database (Thread 3)
def insert_into_database():
    conn = sqlite3.connect('model_detections.db')
    while True:
        result_info = detection_results_queue.get()
        if result_info is None:
            break
        # Convert detections to JSON format
        detections_json = json.dumps(result_info[-1])
        
        cursor = conn.cursor()
        # Get the lock before accessing the db
        with db_lock:
            cursor.execute("INSERT INTO model_detections (id, image_name, detections, is_read) VALUES (?, ?, ?, ?)",
                        (result_info[0], result_info[1], detections_json, False))
            conn.commit()
        cursor.close() 
        
        
        
    conn.close()
    
# Function to read from the db and update "is_read" value (Thread 4)
def reader():
    conn = sqlite3.connect('model_detections.db')
    
    while True:
        cursor = conn.cursor()
        # Read from the db where is_read = 0
        cursor.execute("SELECT * FROM model_detections where is_read = 0")
        results = cursor.fetchall()
        cursor.close()

        # Set is_read = True after reading
        cursor = conn.cursor()
        # Get the lock before accessing the db
        with db_lock:
            cursor.execute("UPDATE model_detections SET is_read = 1 WHERE is_read = 0")
            conn.commit()
        cursor.close()
        
        # The thread ends when all the frames of the video are read i.e : number of read records == number of video frames
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM model_detections WHERE is_read = 1")
        num_read_records = len(cursor.fetchall()) # get the number of read records
        cursor.close()
        if num_read_records == nb_frames: 
           break
    
    conn.close()

# Optional function to save the db in a csv file
def save_db_csv(db_path):
    conn = sqlite3.connect(db_path)  

    query = "SELECT * FROM model_detections"
    df = pd.read_sql_query(query, conn)

    conn.close()

    db_name = db_path.split("/")[-1].split(".")[0]
    df.to_csv(f"{db_name}.csv",index=False)




def get_id(filename):
    return int(filename.split('_')[1].split('.')[0])

# Additional function to create output video from the output images
def create_output_video(frames_folder, video_name):
    # Sort frames by id
    frame_files = sorted(os.listdir(frames_folder), key=get_id)

    # Get dimensions from the first image
    first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    frame_per_sec = fps 
    video = cv2.VideoWriter(video_name, fourcc, frame_per_sec, (width, height))

    # Write frames to the video
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frames_folder, frame_file))
        video.write(frame)

    video.release()


def main():
    
    # Global variables that will be used by the different threads
    global frame_queue, detection_results_queue, model, frames_path, output_path, nb_frames, fps, db_lock
    
    # Script arguments
    parser = argparse.ArgumentParser(description='Run an object detection model on a video')
    parser.add_argument('--v', help='Path of the video file', required=True)
    parser.add_argument('--m', help='Path of the Yolov8 model', default='Models/yolov8n.pt')
    parser.add_argument('--f', action='store_true', help='Save frames and predicted images', default=False)
    parser.add_argument('--s', action='store_true', help='Save the result video', default=False)
    parser.add_argument('--c', action='store_true', help='Save the db to a csv file', default=False)

    args = parser.parse_args()

    video_path = args.v
    model_path = args.m
    save_images = args.f
    save_video = args.s
    save_csv = args.c

    # Exit if the video's path is incorrect
    if not os.path.exists(video_path):
        print("Video not found!")
        return 

    
    # We set save_images = True if we want to create the video
    if save_video:
        save_images = True

    
    # Initialize queues
    frame_queue = queue.Queue()
    detection_results_queue = queue.Queue()

    # Get the number of frames and fps of the input video
    nb_frames = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
    
    # Define frames and predictions path
    frames_path = "Images/"
    output_path = "Results/Images"


    # Clean existing folders
    if os.path.exists(frames_path):
        shutil.rmtree(frames_path)
    if os.path.exists("Results"):
        shutil.rmtree("Results")
    if os.path.exists("model_detections.db"):    
        os.remove("model_detections.db")
    
    if save_images:
        # Create folders to save video frames and predictions
        if not os.path.exists(frames_path):
            os.mkdir(frames_path)

        if not os.path.exists(output_path):
            os.mkdir("Results")
            os.mkdir(output_path)
    
    # Load a YOLOv8 model
    # We'll use the models trained on COCO dataset : https://docs.ultralytics.com/models/yolov8
    model = YOLO(model_path)
    
    # Establish a connection to the database
    conn = sqlite3.connect('model_detections.db')
    cursor = conn.cursor()

    # Create the model_detections table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_detections (
            id INTEGER PRIMARY KEY,
            image_name TEXT,
            detections JSON,
            is_read BOOLEAN
        )
    ''')

    # Commit changes and close the connection
    conn.commit()
    conn.close()

    # Create a lock to synchronize db access between Thread 3 and Thread 4
    db_lock = threading.Lock()



    # Create and start threads
    thread_read_frames = threading.Thread(target=read_frames, args=(video_path,save_images))
    thread_read_frames.start()

    thread_detect_objects = threading.Thread(target=detect_objects, args=(save_images,))
    thread_detect_objects.start()
    
    thread_insert_into_db = threading.Thread(target=insert_into_database)
    thread_insert_into_db.start()
    
    thread_reader = threading.Thread(target=reader)
    thread_reader.start() 
      
    # Wait for threads to finish
    thread_read_frames.join()
    frame_queue.put(None)  # Signal to stop detection thread
    thread_detect_objects.join()
    detection_results_queue.put(None)  # Signal to stop database insertion thread
    thread_insert_into_db.join()
    thread_reader.join()
    
    # Save the model_detections.db to a csv file (Optional)
    if save_csv:
        save_db_csv("model_detections.db")
    
    # Create output video
    if save_video:
        input_video_name = video_path.split("/")[-1].split(".")[0] # extract the video name
        result_video_name = f"{input_video_name}_output.mp4"
        create_output_video(output_path, result_video_name)
        print(f"Video {result_video_name} saved successfully!")

if __name__=="__main__":
    main()  
