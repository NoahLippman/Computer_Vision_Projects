import os

from inference_sdk import InferenceHTTPClient
import numpy as np
import cv2
from baseballcv.functions import LoadTools
from baseballcv.model import Florence2, PaliGemma2, YOLOv9, DETR, RFDETR
from ultralytics import YOLO

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="dQKcEYMXDMssKrAQV4ck"
)
load_tools = LoadTools()
glove_track_model = YOLOv9(load_tools.load_model("glove_tracking"))

def estimateStance(coordinates : dict) -> str:
    try:
        preds = coordinates.get('predictions')
        keypoints = preds[0].get('keypoints')
        keypointList = [[i.get('x'), i.get('y')] for i in keypoints]

        LeftKnee = keypointList[0] if keypointList[0][0] < keypointList[2][0] else keypointList[2]
        RightKnee = keypointList[2] if keypointList[0][0] < keypointList[2][0] else keypointList[0]
        LeftAnkle = keypointList[1] if keypointList[0][0] < keypointList[2][0] else keypointList[3]
        RightAnkle = keypointList[3] if keypointList[0][0] < keypointList[2][0] else keypointList[1]

        if abs(LeftKnee[1] - RightKnee[1]) < 10:
            return "Two Knees Up"
        elif LeftKnee[1] > RightKnee[1]:
            if RightAnkle[0] > RightKnee[0]:
                return "Left Leg Kickstand"
            else:
                return "Right Knee Down"

        else:
            if LeftAnkle[0] < LeftKnee[0]:
                return "Right Leg Kickstand"
            else:
                return "Left Knee Down"
    except:
        pass

def findCoordinates(path):
    cap = cv2.VideoCapture(path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    start_time = duration / 2 - 1
    start_frame = int(start_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = 0
    saved_count = 0
    interval = 5  # Save every 5th frame
    stanceCounts = {"Two Knees Up":0, "Right Knee Down":0, "Left Leg Kickstand":0, "Left Knee Down":0, "Right Leg Kickstand":0}
    class_id = None
    frames_identified = 0
    while True:
        while (frames_identified < 3):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 5 == 0:
                results = glove_track_model.inference(
                    source = frame,
                    conf = .5,
                    show = True
                )
                # Access the detections
                for r in results:
                    for box in r.boxes:
                        # Extract coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
                        confidence = box.conf[0]  # Detection confidence
                        class_id = box.cls[0]  # Class ID of the detection
                        class_name = glove_track_model.names[int(class_id)]
                        if class_name == 'baseball':
                            frames_identified += 1
            frame_count += 1

        frame_count = 0
        while frame_count < 50:
            if frames_identified < 3:
                cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 5 == 0:
                result = CLIENT.infer(frame, model_id="catching-stance-estimator-uimxn/6")
                stance = estimateStance(result)
                print(stance)
                try:
                    stanceCounts[stance] = stanceCounts.get(stance) + 1
                except:
                    pass

            frame_count += 1

        break
    return stanceCounts
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    folder_path = "/Users/noahlippman/Documents/Catcher_Vids_Xavier/video"
    i = 1
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            print(findCoordinates(file_path))
            print(file_path)

            i += 1