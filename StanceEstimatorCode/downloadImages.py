import cv2
import os

def download(path, vidNum, game):
    """
    Creates a png file for every 10th frame starting at 4 seconds for a given path
    Helps generate training data for roboflow keypoint detection

    Args:
        path (str): video path
        vidNum (int): Number of video in path

    Return:
        Nothing, just writes the file to disk
    """
    output_dir = '/Users/noahlippman/Documents/Catcher Photos/' + game
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(fps * 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = 0
    saved_count = 0
    interval = 10  # Save every 10th frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if saved_count < 15 and frame_count % interval == 0:
            cv2.imwrite(f'{output_dir}/{game}_vid{vidNum}frame_{saved_count:04d}.jpg', frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    print(f'Extracted {saved_count} frames from {frame_count} total frames')

if __name__ == "__main__":
    folder_path = "/Users/noahlippman/Documents/Catcher Vids/UNC_Game1/video"
    game = "UNC_Game1"
    i = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            download(file_path, i, game)
            i += 1