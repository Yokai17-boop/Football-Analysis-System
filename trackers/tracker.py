import warnings
warnings.filterwarnings("ignore", message="The `ByteTrack` was deprecated since v0.28.0", category=FutureWarning)

from ultralytics import YOLO
import supervision
import pickle
import os
import cv2
import numpy as np
import pandas as pd

from utils import get_bbox_width, get_center_of_bbox,get_foot_position
from supervision.tracker import ByteTrack

class Tracker:
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = ByteTrack()


    def interpolate_ball_positions(self, ball_positions):
        extracted_bboxes = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        # Track which frames had a real ball detection
        was_detected = [len(bbox) == 4 for bbox in extracted_bboxes]

        formatted_bboxes = [
            bbox if len(bbox) == 4 else [np.nan, np.nan, np.nan, np.nan]
            for bbox in extracted_bboxes
        ]

        df_ball_positions = pd.DataFrame(formatted_bboxes, columns=['x1','y1','x2','y2'])

        #interpolate missing values 
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        df_ball_positions = df_ball_positions.fillna(0)

        ball_positions = [
            {1: {"bbox": x, "detected": detected}}
            for x, detected in zip(df_ball_positions.to_numpy().tolist(), was_detected)
        ]

        return ball_positions


    def detect_frames(self, frames):
        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detection_batch

        return detections


    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks 

        detections = self.detect_frames(frames)

        tracks = {
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_name = detection.names
            cls_name_inv = {v:k for k,v in cls_name.items()}

            player_cls_id = cls_name_inv.get('player', cls_name_inv.get('person'))
            referee_cls_id = cls_name_inv.get('referee')
            ball_cls_id = cls_name_inv.get('ball', cls_name_inv.get('sports ball'))

            # convert the detection to supervision detection format
            detection_supervision = supervision.Detections.from_ultralytics(detection)

            # Convert goalkeeper to player object if present
            if 'goalkeeper' in cls_name_inv:
                gk_cls_id = cls_name_inv['goalkeeper']
                for object_ind, class_id in enumerate(detection_supervision.class_id):
                    if class_id == gk_cls_id and player_cls_id is not None:
                        detection_supervision.class_id[object_ind] = player_cls_id

            # track objects 
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            if detection_with_tracks.tracker_id is not None:
                for bbox, cls_id, track_id in zip(detection_with_tracks.xyxy, detection_with_tracks.class_id, detection_with_tracks.tracker_id):
                    bbox_list = bbox.tolist()
                    if player_cls_id is not None and cls_id == player_cls_id:
                        tracks['players'][frame_num][track_id] = {"bbox": bbox_list}

                    if referee_cls_id is not None and cls_id == referee_cls_id:
                        tracks['referees'][frame_num][track_id] = {"bbox": bbox_list}

            for bbox, cls_id in zip(detection_supervision.xyxy, detection_supervision.class_id):
                if ball_cls_id is not None and cls_id == ball_cls_id:
                    tracks['ball'][frame_num][1] = {"bbox": bbox.tolist()}

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks    


    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])

        x_center, _  = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame, 
            center = (x_center, y2),
            axes = (int(width), int(0.35*width)),
            angle = 0.0,
            startAngle = -45,
            endAngle=235,
            color = color,
            thickness = 2,
            lineType = cv2.LINE_4,

        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2)+15
        y2_rect = (y2 + rectangle_height//2)+15

        if track_id is not None:
            cv2.rectangle(
                            frame,
                            (int(x1_rect),int(y1_rect)),
                            (int(x2_rect),int(y2_rect)),
                            color,
                            cv2.FILLED,
                          )
            
            x1_text = x1_rect+12
            if track_id > 9 and track_id <= 99:
                x1_text -= 5
            elif track_id > 99:
                x1_text -= 10

            cv2.putText(
                        frame,
                        f"{track_id}",
                        (int(x1_text), int(y1_rect+15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,0,0),
                        2,
                        )


        return frame
    

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10, y-20],
            [x+10, y-20],
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)

        return frame


    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparant rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350,850), (1900, 970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]

        # Get the number of time each team has the ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]

        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%", (1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%", (1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame


    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame_copy = frame.copy()

            # Safely access tracking data with bounds checking
            player_dict = {}
            referee_dict = {}
            ball_dict = {}

            if 'players' in tracks and len(tracks['players']) > frame_num:
                player_dict = tracks['players'][frame_num]

            if 'referees' in tracks and len(tracks['referees']) > frame_num:
                referee_dict = tracks['referees'][frame_num]

            if 'ball' in tracks and len(tracks['ball']) > frame_num:
                ball_dict = tracks['ball'][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0,0,255))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bbox'], (0,0,255))

            # Draw Referees
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0,255,255))

            # Draw balls (only when actually detected in the frame)
            for track_id, ball in ball_dict.items():
                if ball.get('detected', True):
                    frame = self.draw_triangle(frame, ball['bbox'], (0,255,0))

            # Draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames


    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                # Skip if track is not a dictionary (e.g., if it's a list or None)
                if not isinstance(track, dict):
                    continue
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position