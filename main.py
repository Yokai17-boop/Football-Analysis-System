import os
import cv2
import numpy as np
import sys
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator

def main():
    # Get video path from command line argument
    if len(sys.argv) < 2:
        print("Error: Please provide a video file path as an argument.")
        print("Usage: python main.py <path_to_video_file>")
        print("Example: python main.py input_videos/my_video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    # Check if file exists
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        print("Please check the file path and try again.")
        sys.exit(1)

    model_path = os.path.join("models", "best.pt")
    if not os.path.exists(model_path):
        model_path = "yolov8x.pt"

    track_stub_path = os.path.join("stubs", "track_stubs.pkl")
    camera_stub_path = os.path.join("stubs", "camera_movement_stub.pkl")
    output_video_path = os.path.join("output_videos", "output.avi")

    # Read Video
    video_frames = read_video(video_path)
    if not video_frames:
        print(f"No frames read from {video_path}. Please check if the video file exists.")
        return

    #initialize tracker
    tracker = Tracker(model_path)

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path=track_stub_path)

    # Ensure tracks have the same length as video_frames
    def truncate_or_pad(lst, target_length, pad_value):
        if len(lst) > target_length:
            # Truncate
            return lst[:target_length]
        elif len(lst) < target_length:
            # Pad
            return lst + [pad_value] * (target_length - len(lst))
        else:
            return lst

    # Truncate or pad player tracks, referee tracks, and ball tracks
    tracks['players'] = truncate_or_pad(tracks['players'], len(video_frames), {})
    tracks['referees'] = truncate_or_pad(tracks['referees'], len(video_frames), {})
    tracks['ball'] = truncate_or_pad(tracks['ball'], len(video_frames), {})

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

    for frame_num in range(len(video_frames)):
        player_tracks = tracks['players'][frame_num]
        for player_id, track in player_tracks.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_color[team]

    # Assign ball acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num in range(len(video_frames)):
        player_track = tracks['players'][frame_num]
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