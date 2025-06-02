## An FYI
Since the yolo models that were trained and used for this build are too big to be normally pushed onto github, here are the steps to aquire them: 

* Run yolo_inference.py, this will download the yolov8 model used for detecting players within the court.
* Run the tennis_ball_detector_training file within the training folder, you can use google colab's limited free gpu if met with a hardware limitation. Save both the last.pt and best.pt to the models folder and rename to yolo5_best.pt and yolo5_last.pt respectively (to be used for ball detection).
* Then run the other ipynb file within training folder, and move output model used for detecting court keypoints to the models folder.
* You should be set up to run the main file, just check to make sure your renamed models are being used in main. You also do not have to use the precomputed stub files and can make your own by setting them to false within main.

