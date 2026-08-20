from pathlib import Path
from detection import convert_to_frame , Detection
from ultralytics import YOLO
from database import conn , create_table , get_detection


#Path to video and model.
video_path = Path('/home/nowa/Desktop/projects/Qa_vehicle/person-bicycle-car-detection.mp4')
d_model = YOLO(Path('/home/nowa/Desktop/projects/Qa_vehicle/yolo11n.pt'))

create_table(conn)

frames = convert_to_frame(video_path).frame_extraction()
Detection(d_model).detector(frames)



