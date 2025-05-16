from ultralytics import YOLO 
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        # load YOLO model
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe 
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])
        # interpolate the missing values with pandas interpolate function
        # useful between frames 
        df_ball_positions = df_ball_positions.interpolate()
        # fill any missing values with .bfill() (back fill) for first frames 
        df_ball_positions = df_ball_positions.bfill()
        # convert back to numpy since we just wanted to use the interpolate/bfill functions from pandas
        # a list of dicts where 1 is trackingId and x is bbox
        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]
        # return ball positions
        return ball_positions

    def get_ball_shot_frames(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Add column to mark ball hit frames
        df_ball_positions['ball_hit'] = 0

        # calculate vertical center
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        # rolling mean of mid_y
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window = 5, min_periods = 1, center=False).mean()
        # difference between consecutive rolling means
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit * 1.2) ):
            # Detect negative -> positive change
            negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i+1] < 0
            # Detect positive -> negative change
            positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i+1] > 0

            # if detecting a change in ball movement, initialize counter to zero
            if negative_position_change or positive_position_change:
                change_count = 0 
                # then look over next couple frames calculated from min # of frames for hit
                # see if direction change continues
                for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0

                    # increment if future change was same as past
                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1
            
                if change_count > minimum_change_frames_for_hit - 1:
                    # mark the frame as ball hit if count is larger than min change frames
                    df_ball_positions['ball_hit'].iloc[i] = 1 

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()

        # return list of hit frame indices
        return frame_nums_with_ball_hits

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        # to store detections
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                # Load detections from file if stub is used
                ball_detections = pickle.load(f)
            return ball_detections

        # iterate through video frames
        for frame in frames:
            # detect the ball in each frame
            player_dict = self.detect_frame(frame)
            # append the detection to the list
            ball_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                # Save detections to file if stub_path is provided
                pickle.dump(ball_detections, f)
        # return all detections
        return ball_detections

    def detect_frame(self, frame):
        # run YOLO prediction on the frame
        results = self.model.predict(frame,conf=0.15)[0]
        # to store ball detection
        ball_dict = {}
        # iterate through bboxes
        for box in results.boxes:
            # get bbox coordinates
            result = box.xyxy.tolist()[0]
            # Store with key 1 (assuming only one ball)
            ball_dict[1] = result
        # return the detection dict
        return ball_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames


    