# for yolo models
from ultralytics import YOLO 
# for img/video processing
import cv2
# for serializing/deserializing py obj's
import pickle
# manipulate python runtime env
import sys
# to allow imports from parent dir so that we can import utils
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

# class with methods that detect and track players in video frames
class PlayerTracker:
    def __init__(self, model_path):
        # takes in a path for which model to use
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        # choose/filter based on first frame
        player_detections_first_frame = player_detections[0]
        # selects players based on proximity and keypoints
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    # to select 2 players closest to the court keypoints 
    def choose_players(self, court_keypoints, player_dict):
        # to store the distances between players and court keypoints
        distances = []
        # iterate over detected players
        for track_id, bbox in player_dict.items():
            # gets center of player's bbox
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            # iterate over court keypoints
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                # calculate the distance
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            # add player id and min_dist to list
            distances.append((track_id, min_distance))
        
        # sort the distances in ascending order
        distances.sort(key = lambda x: x[1])
        # Choose the first 2 tracks (2 closest players)
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players

    # to detect players from multiple frames
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        # to store detections
        player_detections = []

        # Checks if detections should be loaded from a file with pickle.load(f)
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        # iterate over sequence of frames
        for frame in frames:
            # detect players and add to list
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        # save detections to a file if stub_path is provided
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        # return the detections
        return player_detections
    
    # to detect players in single frame
    def detect_frame(self, frame):
        # runs the YOLO tracking on frame
        results = self.model.track(frame, persist=True)[0]
        # map classID -> classnames
        id_name_dict = results.names

        # player id: bounding box (bbox)
        player_dict = {}
        # iterates over detected bboxes
        for box in results.boxes:
            # extract track ID
            track_id = int(box.id.tolist()[0])
            # extract bbox coordinates
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            # If of a person class, add player to dict
            if object_cls_name == "person":
                player_dict[track_id] = result
        
        # return player detections
        return player_dict

    # method to draw bbox and player ID onto the video frames
    def draw_bboxes(self, video_frames, player_detections):
        # for annotated frames
        output_video_frames = []
        # iterate over frames and the corresponding detections
        for frame, player_dict in zip(video_frames, player_detections):
            # iterate over detected players
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                # Draw the ID and Bounding Boxes with cv2
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            # add annotated frame to list
            output_video_frames.append(frame)
        
        # return the list
        return output_video_frames


    