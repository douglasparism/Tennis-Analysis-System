from ultralytics import YOLO

# import a yolo model, 'x' stands for extra large
model = YOLO('yolov8x')

# to run model on an image within dir
model.predict('input_video/image.png', save=True)


