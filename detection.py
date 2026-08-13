import cv2
from database import conn, create_table, insert_detection


class convert_to_frame:
    def __init__(self, video):
        self.video = video

    def frame_extraction(self):
        cap = cv2.VideoCapture(str(self.video))
        if not cap.isOpened():
            exit()

        frame_counter = 0
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.step = int(self.fps // 4)
        frames = []

        while cap:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_counter % self.step == 0:
                timestamp = frame_counter / self.fps
                frames.append([timestamp, frame])
            frame_counter += 1

        cap.release()
        return frames

class Detection:

    def __init__(self, model):
        self.model = model
        self.conn = conn

    def detector(self, frames):

        for timestamp, frame in frames:
            labels = self.model(frame)

            for box in labels[0].boxes:
                cls_id = int(box.cls[0])
                cls_text = self.model.names[cls_id]
                insert_detection(self.conn, timestamp, cls_text)




