import cv2
from flask import Flask, request, jsonify, send_file
from pyngrok import ngrok
from collections import defaultdict
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('YOLOv8X.pt')

# Start ngrok tunnel
public_url = ngrok.connect(5000)
public_url = str(public_url).split('"')[1]
print(f"NGROK URL: {public_url}")

# Dictionary to hold processed video info
processed_videos = {}

@app.route('/process_video', methods=['POST'])
def process_video():
    save_dir = "videos"
    processed_dir = "processed_videos"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    video = request.files['video']
    video_path = os.path.join(save_dir, video.filename)
    video.save(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Could not open video."}), 400

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    START = (0, frame_height - (frame_height // 5))
    END = (frame_width - 1, frame_height - (frame_height // 5))

    # Define the region for counting instead of a single line
    REGION_TOP = frame_height - (frame_height // 4)
    REGION_BOTTOM = frame_height - (frame_height // 5)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    processed_video_path = os.path.join(processed_dir, f"processed_{video.filename}")
    out = cv2.VideoWriter(processed_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

    track_history = defaultdict(list)
    crossed_objects = set()
    frequency_data = []
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * 5)  # 5-second interval
    count = 0
    last_interval = 0
    total_count = 0  # To store the total number of vehicles

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            frame_count += 1
            results = model.track(frame, classes=[2, 3, 5, 7], persist=True, tracker="bytetrack.yaml", conf=0.1, iou=0.3)

            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 100:
                        track.pop(0)

                    # Check if the vehicle is within the defined region
                    if START[0] < x < END[0] and REGION_TOP < y < REGION_BOTTOM:
                        if track_id not in crossed_objects:
                            crossed_objects.add(track_id)
                            count += 1
                            total_count += 1  # Increment the total count

            # Calculate the current interval
            current_interval = frame_count // interval_frames

            # Check if we are at a new 5-second interval
            if current_interval > last_interval:
                # Store the count for the last interval
                frequency_data.append(count)
                # Reset the count and crossed_objects set for the new interval
                count = 0
                crossed_objects.clear()
                last_interval = current_interval

            # Draw the region for counting
            cv2.rectangle(frame, (START[0], REGION_TOP), (END[0], REGION_BOTTOM), (0, 255, 0), 2)
            count_text = f"Objects crossed: {count}"
            cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            annotated_frame = results[0].plot()
            out.write(annotated_frame)
        else:
            # At the end of the video, append the count for the last interval
            frequency_data.append(count)
            break

    cap.release()
    out.release()

    # Send total count and frequency data to FE
    video_key = f"processed_{video.filename}"
    processed_videos[video_key] = {
        "count": total_count,
        "frequency_data": frequency_data,
        "video_url": f"/downloads/{video_key}"
    }

    return jsonify({
        "count": total_count,
        "frequency_data": frequency_data,
        "video_url": f"/downloads/{video_key}"
    })

@app.route('/downloads/<filename>', methods=['GET'])
def download_video(filename):
    file_path = os.path.join("processed_videos", filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype="video/mp4", as_attachment=False)
    else:
        return jsonify({"error": "File not found."}), 404

if __name__ == '__main__':
    app.run()
