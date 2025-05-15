# for working with imgs/video
import cv2

# takes video path
def read_video(video_path):
    # opens connection to video
    cap = cv2.VideoCapture(video_path)
    frames = []
    # loop until frames are done
    while True:
        ret, frame = cap.read()
        # ret will be false when there are no more frames to read
        if not ret:
            break
        # append frame to list of frames
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # 24 fps
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()