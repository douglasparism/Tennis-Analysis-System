import torch
# for preprocessing images before feeding them into a model
import torchvision.transforms as transforms
import cv2
# for pre-trained models
from torchvision import models
import numpy as np

# class to detect court lines/keypoints in an img using a NN
class CourtLineDetector:
    def __init__(self, model_path):
        # load pre-trained model
        self.model = models.resnet50(pretrained=True)
        # replace final fully connected (fc) layer of nn to output 28 values 
        # 14 x-kps, 14 y-kps
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) 
        # load model weights
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        # transform the image (PIL, resize, tensify, normalize) for NN
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # predict kps using img (usually first one)
    def predict(self, image):
        # converts input img from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply transformations + batch dimension
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        # run model to predict keypoints from 1st img
        with torch.no_grad():
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]
        # scale/map kps back to original img size
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        # return keypoints as nummpy array
        return keypoints

    # to draw keypoints onto 1 image (takes in img instead of video frames)
    def draw_keypoints(self, image, keypoints):
        # Plot the keypoints returned by predict on the image
        for i in range(0, len(keypoints), 2):
            # iterate over each x & y index (thus by 2), used to draw keypoints with cv2
            # use int() to make sure any fractions are not used for pixel position
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            # label keypoint in img
            cv2.putText(image, str(i // 2), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # draw circle around keypoint in img
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        # return annotated frames
        return image
    
    # draw keypoints onto each frame of video
    def draw_keypoints_on_video(self, video_frames, keypoints):
        # list to store annotated video frames 
        output_video_frames = []
        # iterate over video frames
        for frame in video_frames:
            # uses prior function to draw kps
            frame = self.draw_keypoints(frame, keypoints)
            # append to list
            output_video_frames.append(frame)
        # return those annotated video frames
        return output_video_frames