from ultralytics import YOLO

# import a yolo model made from the models folder (will be empty since in repo since it is big)
model = YOLO('yolov8x')

# to run model on an image within dir
result = model.track('input_videos/input_video.mp4', conf=0.2, save=True)
# print(result)
# print("Boxes:")
# for box in result[0].boxes:
#     print(box)

