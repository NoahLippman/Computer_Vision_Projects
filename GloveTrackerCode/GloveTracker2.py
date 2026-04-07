import math
import os
import tempfile
import time as tm

from roboflow import Roboflow
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib import animation
from skimage.metrics import structural_similarity as ssim
import numpy as np
import pandas as pd

release_image = None
pitcher_coordinates = None
rf = None
glove_track_model = None
pitcher_position_model = None
CLIENT = None

def _init_models():
    global rf, glove_track_model, pitcher_position_model, release_image, pitcher_coordinates
    rf = Roboflow(api_key="dQKcEYMXDMssKrAQV4ck")
    glove_track_model = rf.workspace().project("glove-tracker-8ik0s").version("6").model
    pitcher_position_model = rf.workspace().project("pitcher-release-identifier").version("1").model
    release_image = cv2.imread('/Users/noahlippman/Documents/Computer_Vision_Projects/GloveTrackerCode/reference/reference_crop.png', cv2.IMREAD_GRAYSCALE)
    try:
        with open('/Users/noahlippman/Documents/Computer_Vision_Projects/GloveTrackerCode/reference/bbox_coords.json', 'r') as file:
            pitcher_coordinates = json.load(file)
    except FileNotFoundError:
        print(f"Error: The pitcher coordinate file was not found")
        return None
    except json.JSONDecodeError :
        print(f"Error: Could not decode JSON from the file")
        return None

def _inside_bbox(x, y, coords):
    """Return True if (x, y) is inside bbox coords, or if no coords are set."""
    if coords is None:
        return True
    return coords['x1'] <= x <= coords['x2'] and coords['y1'] <= y <= coords['y2']

def _pixel_to_trackman(gx, gy, cal):
    """Convert pixel coords to trackman (side, height) using two-point linear calibration.
    cal = {'px1','py1','side1','height1','px2','py2','side2','height2'}
    """
    pixels_per_side   = (cal['px2'] - cal['px1']) / (cal['side2']   - cal['side1'])
    pixels_per_height = (cal['py2'] - cal['py1']) / (cal['height2'] - cal['height1'])
    glove_side   = cal['side1']   + (gx - cal['px1']) / pixels_per_side
    glove_height = cal['height1'] + (gy - cal['py1']) / pixels_per_height
    return round(glove_side, 3), round(glove_height, 3)


def _show_frame(vid_path: str, frame_number: int, prediction: dict, label: str, suppress_display: bool = False):
    cap = cv2.VideoCapture(vid_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    x, y, w, h = prediction['x'], prediction['y'], prediction['width'], prediction['height']
    x1, y1 = int(x - w / 2), int(y - h / 2)
    x2, y2 = int(x + w / 2), int(y + h / 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if not suppress_display:
        cv2.imshow(label, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return frame

def _show_frame_with_dots(vid_path: str, frame_number: int, dots: list, label: str, suppress_display: bool = False):
    cap = cv2.VideoCapture(vid_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    for (x, y, color) in dots:
        cv2.circle(frame, (int(x), int(y)), 5, color, -1)
    # Burn label onto frame (split at ' - ' for two-line readability)
    parts = label.split(' - ', 1)
    cv2.putText(frame, parts[0], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if len(parts) > 1:
        cv2.putText(frame, parts[1], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if not suppress_display:
        cv2.imshow(label, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return frame

def ssim_similarity(img1, img2):
    # Compute SSIM (returns value in [0,1])
    ssim_value = ssim(img1, img2, data_range=img2.max() - img2.min())
    # Convert to percentage
    return ssim_value

def orb_similarity(img1, img2):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Match descriptors using BFMatcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    # Sort matches by distance (lower = better)
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate similarity: (good matches / total matches) * 100
    # Use top 10% matches for robustness
    num_matches = len(matches)
    if num_matches == 0:
        return 0.0  # No matches
    good_matches = matches[:int(num_matches * 0.1)]  # Top 10%
    similarity = (len(good_matches) / num_matches) * 100
    return similarity

def find_pitch_start(vid_path: str, debug_frames: list = None, suppress_display: bool = False):
    _init_models()

    # Trim first 2 seconds before submitting to inference
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    skip_frames = int(fps * 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)

    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    tmp_path = tmp.name
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
    cap.release()
    writer.release()

    try:
        job_id, signed_url, expire_time = pitcher_position_model.predict_video(
            tmp_path,
            fps=59,
            prediction_type="batch-video",
        )

        results = pitcher_position_model.poll_for_video_results(job_id)
        while len(results.keys()) == 0:
            tm.sleep(5)
            results = pitcher_position_model.poll_for_video_results(job_id)

        max_conf = 0
        start_frame_final = None
        start_track = None
        release_prediction = None
        frames_tracked = 0

        for i, result in enumerate(results['pitcher-release-identifier']):
            if frames_tracked > 15:
                break
            try:
                if result['predictions'][0]['class'] == 'Release':
                    frames_tracked += 1
                    confidence = result['predictions'][0]['confidence']
                    # Offset frame number by skip_frames so it maps back to the original video
                    orig_frame = results['frame_offset'][i] + skip_frames
                    annotated = _show_frame(vid_path, orig_frame, result['predictions'][0], str(confidence), suppress_display=suppress_display)
                    if debug_frames is not None and annotated is not None:
                        debug_frames.append(annotated)
                    if result['predictions'][0]['confidence'] >= max_conf:
                        start_frame_final = orig_frame
                        start_track = i
                        max_conf = result['predictions'][0]['confidence']
                        release_prediction = result['predictions'][0]
            except:
                pass

        if start_frame_final is not None:
            annotated = _show_frame(vid_path, start_frame_final, release_prediction, "Pitch Start - Confidence = " + str(max_conf), suppress_display=suppress_display)
            if debug_frames is not None and annotated is not None:
                debug_frames.append(annotated)

        return start_track, start_frame_final
    finally:
        os.remove(tmp_path)

def calculate_glove_latency(vid_path: str, start_frame: int, mph: float, fps, debug_frames: list = None, suppress_display: bool = False):
    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end_frame = min(start_frame + 60, total_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    tmp_path = tmp.name
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
    cap.release()
    writer.release()

    try:
        job_id, signed_url, expire_time = glove_track_model.predict_video(
            tmp_path,
            fps= 59,
            prediction_type="batch-video",
        )
        results = glove_track_model.poll_for_video_results(job_id)
        while(len(results.keys()) == 0):
            tm.sleep(5)
            results = glove_track_model.poll_for_video_results(job_id)

        tracked_after_delivery = results['glove-tracker-8ik0s']

        # Find first detection inside the bbox to use as resting baseline
        low_x = None
        low_y = None
        for track in tracked_after_delivery:
            try:
                x = track['predictions'][0]['x']
                y = track['predictions'][0]['y']
                if _inside_bbox(x, y, pitcher_coordinates):
                    low_x, low_y = x, y
                    break
            except:
                pass

        if low_x is None:
            return 0, results, None, None

        prev_x = low_x
        prev_y = low_y
        initial_movement_index = 0
        movement_start_x = 0
        movement_start_y = 0
        tracks_with_movement = 0
        for i, track in enumerate(tracked_after_delivery):
            try:
                cur_x = track['predictions'][0]['x']
                cur_y = track['predictions'][0]['y']

                # Skip detections outside bbox — forward-fill from previous
                if not _inside_bbox(cur_x, cur_y, pitcher_coordinates):
                    cur_x = prev_x
                    cur_y = prev_y

                difference = (cur_y-low_y) * -1
                prev_difference = (cur_y-prev_y) * -1
                dots = [
                    (low_x, low_y, (0,0,255)),
                    (cur_x, cur_y, (255,0,0)),
                ]
                annotated = _show_frame_with_dots(tmp_path, results['frame_offset'][i], dots, "Distance from Start: " + str(difference) + " - Distance from previous track: " + str(prev_difference), suppress_display=suppress_display)
                if debug_frames is not None and annotated is not None:
                    debug_frames.append(annotated)
                if difference > 50:
                    pass
                prev_x = cur_x
                prev_y = cur_y

                if(difference >= 4 or prev_difference > .75):
                    if tracks_with_movement == 0:
                        initial_movement_index = i
                        movement_start_x = cur_x
                        movement_start_y = cur_y
                    tracks_with_movement += 1
                else:
                    initial_movement_index = 0
                    tracks_with_movement = 0
                if tracks_with_movement >= 3:
                    initial_movement_index = initial_movement_index
                    actual_frame = results['frame_offset'][initial_movement_index]
                    dots = [
                        (low_x, low_y, (0, 0, 255)),        # red: resting position
                        (movement_start_x, movement_start_y, (255, 0, 0)),  # blue: movement start
                    ]
                    annotated = _show_frame_with_dots(tmp_path, actual_frame, dots, "Glove Movement Start", suppress_display=suppress_display)
                    if debug_frames is not None and annotated is not None:
                        debug_frames.append(annotated)
                    break
            except:
                pass

        return initial_movement_index, results, low_x, low_y
    finally:
        os.remove(tmp_path)


def write_debug_video(frames: list, out_path: str, fps: float, latency_frames: int = None):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for i, frame in enumerate(frames):
        if i == len(frames) - 1 and latency_frames is not None:
            frame = frame.copy()
            cv2.putText(frame, f"Glove Latency: {latency_frames} frames", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        writer.write(frame)
    writer.release()


def animate_glove_movement(
    results: dict,
    start_idx: int,
    output_path: str,
    mph: float = None,
    fps: float = 10.0,
):
    """
    Animate glove movement from Roboflow tracking results as a coordinate plot.

    Args:
        results: Roboflow results dict from glove_track_model (glove-tracker-8ik0s)
        start_idx: index into results where glove movement begins (initial_movement_index)
        output_path: path to save the output MP4
        mph: pitch velocity for velocity-based fallback endpoint calculation
        fps: video fps used for velocity fallback frame conversion
    """
    tracked = results['glove-tracker-8ik0s']
    time_offsets = results['time_offset']

    # Step 1: Extract and forward-fill (x, y) from start_idx onward
    raw_x, raw_y = [], []
    for track in tracked[start_idx:]:
        try:
            raw_x.append(track['predictions'][0]['x'])
            raw_y.append(track['predictions'][0]['y'])
        except (IndexError, KeyError):
            raw_x.append(None)
            raw_y.append(None)

    xs, ys = [], []
    prev_x, prev_y = 0, 0
    for x, y in zip(raw_x, raw_y):
        if x is not None:
            prev_x, prev_y = x, y
        xs.append(prev_x)
        ys.append(prev_y)

    n = len(xs)
    if n < 2:
        return

    # Step 2: Determine stable initial direction from first 3-5 non-trivial movement vectors
    vectors = [(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(n - 1)]
    initial_vecs = []
    for vx, vy in vectors[:min(10, len(vectors))]:
        mag = math.sqrt(vx ** 2 + vy ** 2)
        if mag > 0.5:
            initial_vecs.append((vx / mag, vy / mag))
        if len(initial_vecs) >= 5:
            break

    if not initial_vecs:
        init_dx, init_dy = 0.0, 1.0  # fallback: straight down
    else:
        init_dx = sum(v[0] for v in initial_vecs) / len(initial_vecs)
        init_dy = sum(v[1] for v in initial_vecs) / len(initial_vecs)
        mag = math.sqrt(init_dx ** 2 + init_dy ** 2)
        if mag > 0:
            init_dx /= mag
            init_dy /= mag

    # Step 3: Direction reversal detection — endpoint when angle > 20° for 3+ consecutive frames
    end_idx = n - 1
    ANGLE_THRESHOLD = 20.0
    MIN_CONSECUTIVE = 3
    consecutive_reversal = 0
    reversal_start = None

    for i, (vx, vy) in enumerate(vectors):
        mag = math.sqrt(vx ** 2 + vy ** 2)
        if mag < 0.5:
            consecutive_reversal = 0
            reversal_start = None
            continue
        dot = max(-1.0, min(1.0, (vx / mag) * init_dx + (vy / mag) * init_dy))
        angle = math.degrees(math.acos(dot))
        if angle > ANGLE_THRESHOLD:
            if consecutive_reversal == 0:
                reversal_start = i
            consecutive_reversal += 1
            if consecutive_reversal >= MIN_CONSECUTIVE:
                end_idx = reversal_start
                break
        else:
            consecutive_reversal = 0
            reversal_start = None

    # Step 4: Velocity fallback if no reversal detected
    if end_idx == n - 1 and mph is not None:
        time_s = 57.0 / (mph * 5280.0 / 3600.0)
        end_idx = min(int(time_s * fps), n - 1)

    # Step 5: Build matplotlib animation
    plot_xs = xs[:end_idx + 1]
    plot_ys = ys[:end_idx + 1]
    num_frames = len(plot_xs)

    if num_frames < 2:
        return

    # Calculate output FPS so every result frame in the window is shown once at natural timing
    abs_end = start_idx + end_idx
    t_start = time_offsets[start_idx] if start_idx < len(time_offsets) else 0
    t_end = time_offsets[abs_end] if abs_end < len(time_offsets) else t_start + num_frames / fps
    time_span = t_end - t_start
    fps_out = num_frames / time_span if time_span > 0 else 28.5

    x_range = max(plot_xs) - min(plot_xs)
    y_range = max(plot_ys) - min(plot_ys)
    x_pad = x_range * 0.3 if x_range > 0 else 30
    y_pad = y_range * 0.3 if y_range > 0 else 30
    x_min, x_max = min(plot_xs) - x_pad, max(plot_xs) + x_pad
    y_min, y_max = min(plot_ys) - y_pad, max(plot_ys) + y_pad

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor('#2d2d2d')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # inverted Y axis (video pixel coords)
    ax.set_aspect('equal')
    ax.set_title('Glove Movement Path 2')
    ax.tick_params(labelbottom=False, labelleft=False)

    # Static green dot at starting position
    ax.plot(plot_xs[0], plot_ys[0], 'go', markersize=10, label='Start', zorder=5)

    # Animated path line and red dot (current position)
    path_line, = ax.plot([], [], 'b-', linewidth=2)
    red_dot, = ax.plot([], [], 'ro', markersize=10, label='Current', zorder=6)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0)

    def _update(frame):
        path_line.set_data(plot_xs[:frame + 1], plot_ys[:frame + 1])
        red_dot.set_data([plot_xs[frame]], [plot_ys[frame]])
        return path_line, red_dot

    anim = animation.FuncAnimation(fig, _update, frames=num_frames, interval=1000 / fps_out, blit=True)
    writer = animation.FFMpegWriter(fps=fps_out)
    anim.save(output_path, writer=writer)
    plt.close(fig)

def work_on_dataframe(vid_data, pitch_data, video_folder, output_path, progress_callback=None, calibration=None):
    total = len(vid_data)
    valid_events = 0
    for i, (_, row) in enumerate(vid_data.iterrows()):
        if progress_callback is not None:
            progress_callback(i, total, f"Processing video {i + 1}/{total}: {row.get('#', '')}")
        if row['Pitch Result'] == "POA":
            pass
        elif row['Pitch Result'] == "Strike Taken" or row['Pitch Result'] == "Ball":
            path = os.path.join(video_folder, str(row['#']) + ".mp4")
            try:
                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                start_track, start_frame = find_pitch_start(path, suppress_display=True)
                if start_frame is None:
                    valid_events += 1
                    continue
                latency, glove_results, glove_x, glove_y = calculate_glove_latency(path, start_frame, 90, fps, suppress_display=True)
                pitch_data.loc[valid_events, 'Frame'] = latency
                if calibration is not None and glove_x is not None and glove_y is not None:
                    gs, gh = _pixel_to_trackman(glove_x, glove_y, calibration)
                    pitch_data.loc[valid_events, 'GloveLocSide']   = gs
                    pitch_data.loc[valid_events, 'GloveLocHeight'] = gh
                valid_events += 1
            except:
                valid_events += 1
                pass
        else:
            pitch_data.loc[valid_events, 'Frame'] = None
            valid_events += 1
    pitch_data.to_csv(output_path, index=False)
    if progress_callback is not None:
        progress_callback(total, total, "Done")
    return pitch_data


if __name__ == "__main__":
    '''
    folder_path = "/Users/noahlippman/Documents/Catcher Vids/UW_Game1/Test accuracy"
    i = 1
    for filename in os.listdir(folder_path):
        debug_out = "/Users/noahlippman/Documents/Catcher Vids/" + "/Annotated_Vid-" + filename
        filename = folder_path + "/" + filename
        cap = cv2.VideoCapture(filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        debug_frames = []
        start_track, start_frame = find_pitch_start(filename, debug_frames=debug_frames)
        if start_frame is not None:
            latency, glove_results = calculate_glove_latency(filename, start_frame, 90, fps, debug_frames=debug_frames)
            write_debug_video(debug_frames, debug_out, fps, latency_frames=latency)
            print("Glove latency (frames):", latency)
        i = i + 1
    '''
    path = "/Users/noahlippman/Documents/Framing Vids + Data/MIN_Game2/Vids/video/194.mp4"
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    debug_frames = []
    start_track, start_frame = find_pitch_start(path, debug_frames=debug_frames)
    latency, glove_results, _, _ = calculate_glove_latency(path, start_frame, 90, fps, debug_frames=debug_frames)
    #write_debug_video(debug_frames, debug_out, fps, latency_frames=latency)
    print("Glove latency (frames):", latency)
    animate_glove_movement(
        results=glove_results,
        start_idx=latency,
        output_path="glove_movement.mp4",
        mph=80,
        fps=fps,
    )


