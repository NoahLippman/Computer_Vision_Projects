import os
import time as tm
from inference_sdk import InferenceHTTPClient
from matplotlib.patches import Rectangle
from roboflow import Roboflow
from matplotlib import pyplot as plt, animation
from matplotlib.animation import FFMpegWriter
from matplotlib.animation import FuncAnimation
from ultralytics import YOLO
import cv2


rf = Roboflow(api_key="dQKcEYMXDMssKrAQV4ck")
project = rf.workspace().project("glove-tracker-8ik0s")
model = project.version("3").model

'''
job_id, signed_url, expire_time = model.predict_video(
    "/Users/noahlippman/Documents/Catcher Vids/UNLV_2_14/video/70.mp4",
    fps=5,
    prediction_type="batch-video",
)t
'''

ball_track_model = YOLO('https://data.balldatalab.com/index.php/s/YkGBwbFtsf34ky3/download/ball_tracking_v4-YOLOv11.pt')

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="dQKcEYMXDMssKrAQV4ck"
)

def trackBall(path: str):
    results = ball_track_model.predict(path, show=True, stream=True)
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        try:
            conf = boxes.conf[0]
            if conf > .35:
                ball_identified = True
                #break
        except:
            pass

def processVid(path : str):
    job_id, signed_url, expire_time = model.predict_video(
        path,
        fps=20,
        prediction_type="batch-video",
    )
    results = model.poll_for_video_results(job_id)
    while(len(results.keys()) == 0):
        tm.sleep(15)
        results = model.poll_for_video_results(job_id)
    #results = model.poll_until_video_results(job_id)
    time = [i for i in results['time_offset']]
    x = [i['predictions'][0]['x'] if (i.get('predictions') and len(i['predictions']) > 0 and 'x' in i['predictions'][0]) else None
         for i in results['glove-tracker-8ik0s']]
    y = [i['predictions'][0]['y'] if (i.get('predictions') and len(i['predictions']) > 0 and 'y' in i['predictions'][0]) else None
         for i in results['glove-tracker-8ik0s']]
    height = [i['predictions'][0]['height'] if (i.get('predictions') and len(i['predictions']) > 0 and 'height' in i['predictions'][0]) else None
         for i in results['glove-tracker-8ik0s']]
    width = [i['predictions'][0]['width'] if (i.get('predictions') and len(i['predictions']) > 0 and 'width' in i['predictions'][0]) else None
         for i in results['glove-tracker-8ik0s']]

    return (time, x, y, height, width)

def fixList(ls):
    previousTrack = 0
    newList = []
    for i in ls:
        if i != None:
            previousTrack = i
        newList.append(previousTrack)
    return newList

def plot_vid(data):
    time, x, y, height, width = data
    time, x, y, height, width = fixList(time), fixList(x), fixList(y), fixList(height), fixList(width)
    y_cropped = y[3:100]
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes()
    ax.set_xlim(400, 800)
    ax.set_ylim(500,100)
    ax.set_aspect('equal')
    max_y = y_cropped.index(max(y_cropped)) - 3
    time = time[max_y:]
    x = x[max_y:]
    y = y[max_y:]
    height = height[max_y:]
    width = width[max_y:]
    topLeft = [600, 215]
    topRight = [660, 215]
    bottomLeft = [600, 290]
    bottomRight = [660, 290]
    x_coords = [topLeft[0], topRight[0], bottomRight[0], bottomLeft[0], topLeft[0]]
    y_coords = [topLeft[1], topRight[1], bottomRight[1], bottomLeft[1], topLeft[1]]
    ax.plot(x_coords, y_coords, 'r-', linewidth = 2)
    point, = ax.plot([x[0]], [y[0]], 'bo', markersize = 10)

    def update(frame):
        point.set_data([x[frame]], [y[frame]])
        return point,
    anim = animation.FuncAnimation(fig, update, frames = len(time), interval = 50, blit = True)
    anim.save('rectangle_movement.mp4', writer = 'ffmpeg', fps = 20)

if __name__ == "__main__":
    path = "/Users/noahlippman/Documents/Catcher Vids/UNC_Game1/video/19.mp4"
    trackBall(path)
    #data = processVid(path)
    #plot_vid(data)

'''fig = plt.figure(figsize=(7,5))
axis = plt.axes(xlim = (0,.2),
                ylim = (0,1000))

plt.plot(x, y)
plt.show()

line,  = axis.plot([], [], lw = 2)

def init():
    line.set_data([], [])
    return line,

xdata, ydata = [], []

def animate(i):
    line.set_data(x[:i+1], y[:i+1])
    return line,

anim = animation.FuncAnimation(fig, animate, init_func = init, frames = len(x), interval = 20, blit = True)

anim.save('testAnimation.mp4', writer = 'ffmpeg', fps = 30)'''
