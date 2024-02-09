import numpy as np
import supervision as sv
from datetime import datetime
from ultralytics import YOLO
import pymysql

# label dictionary
label_dict = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

def perform_detection(source_video_path: str, target_video_path: str, selected_class_indices: list):
    """
    Perform object detection on the source video and store the results in a MySQL database. 
    Parameters:
        source_video_path: str - the path to the source video
        target_video_path: str - the path to store the annotated video
        selected_class_indices: list - a list of class indices to filter the detections
    Returns:
        None
    """
    # Connect to MySQL database using pymysql
    mydb = pymysql.connect(
        host="localhost",
        user="root",
        password="",
        database="detection"
    )
    mycursor = mydb.cursor()

    # Check if the table exists, if not, create it
    mycursor.execute("SHOW TABLES LIKE 'detection_data'")
    if not mycursor.fetchone():
        create_table_sql = """
        CREATE TABLE detection_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            tracker_id INT,
            class_id INT,
            class_name VARCHAR(255),
            detection_time DATETIME
        )
        """
        mycursor.execute(create_table_sql)
        mydb.commit()

    # Initialize YOLO model, tracker, annotators
    model = YOLO("yolov8m.pt")
    tracker = sv.ByteTrack()
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Set to store unique tracker IDs
    unique_tracker_ids = set()

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        # Record the current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Perform object detection
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Filter out detections with confidence below 0.5 and matching selected classes
        detections = detections[(detections.confidence > 0.5)]
        
        # Filter out by classes
        detections = detections[np.isin(detections.class_id, selected_class_indices)]

        # Update tracker with detections
        detections = tracker.update_with_detections(detections)

        # Extract class_id, tracker_id, and class_name
        class_tracker_pairs = [(int(class_id), int(tracker_id), results.names[int(class_id)]) for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)]

        # Store data in MySQL database for unique tracker IDs
        for class_id, tracker_id, class_name in class_tracker_pairs:
            if tracker_id not in unique_tracker_ids:
                sql = "INSERT INTO detection_data (tracker_id, class_id, class_name, detection_time) VALUES (%s, %s, %s, %s)"
                val = (tracker_id, class_id, class_name, current_time)
                mycursor.execute(sql, val)
                mydb.commit()
                unique_tracker_ids.add(tracker_id)

        # Annotate frame
        labels = [f"#{tracker_id} {class_name}" for class_id, tracker_id, class_name in class_tracker_pairs]
        annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)

        return annotated_frame

    # Process the video
    sv.process_video(
        source_path=source_video_path,
        target_path=target_video_path,
        callback=callback
    )
    
