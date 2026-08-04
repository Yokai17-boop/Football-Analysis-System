import os
import cv2
import numpy as np
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator

def main():
    input_video_path = os.path.join("input_videos", "football_sample_video.mp4")
    if not os.path.exists(input_video_path):
        input_video_path = os.path.join("input_videos", "08fd33_4.mp4")

    model_path = os.path.join("models", "best.pt")
    if not os.path.exists(model_path):
        model_path = "yolov8x.pt"

    track_stub_path = os.path.join("stubs", "track_stubs.pkl")
    camera_stub_path = os.path.join("stubs", "camera_movement_stub.pkl")
    output_video_path = os.path.join("output_videos", "output.avi")

    # Read Video
    video_frames = read_video(input_video_path)   
    if not video_frames:
        print(f"No frames read from {input_video_path}. Please check if the video file exists.")
        return

    #initialize tracker 
    tracker = Tracker(model_path)

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path=track_stub_path)
    
    # Get object positions
    tracker.add_position_to_tracks(tracks)
    
    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, 
                                                                              read_from_stub=True, 
                                                                              stub_path=camera_stub_path)
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # view transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    #interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and Distance Estimator 
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_sistance_to_tracks(tracks)


    # Assign player teams 
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])
    
    for frame_num, player_tracks in enumerate(tracks['players']):
        for player_id, track in player_tracks.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_color[team]

    # Assign ball acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_data = tracks['ball'][frame_num].get(1, {})
        ball_bbox = ball_data.get('bbox', [0,0,0,0])
        ball_detected = ball_data.get('detected', False)

        # Only assign ball possession when the ball is actually visible in the frame
        if ball_detected:
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        else:
            assigned_player = -1

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(1)
    team_ball_control = np.array(team_ball_control)


    # Draw Output
    ## Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    ## Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    ## Draw speed and distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save Video
    save_video(output_video_frames, output_video_path)
    
if __name__ == '__main__':
    main()