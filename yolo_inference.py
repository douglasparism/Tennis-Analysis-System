from ultralytics import YOLO

# import a yolo model, 'x' stands for extra large
model = YOLO('yolov8x')

# to run model on an image within dir
result = model.predict('input_videos/input_video.mp4', save=True)
print(result)
print("Boxes:")
for box in result[0].boxes:
    print(box)

