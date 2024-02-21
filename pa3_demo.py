# -*- coding: utf-8 -*-
import json
import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import math


#from google.colab.patches import cv2_imshow

# part 1:

def load_obj_each_frame(data_file):
    with open(data_file, 'r') as file:
        frame_dict = json.load(file)
    return frame_dict


def alpha_beta_filter(observed_centers, alpha=0.1, beta=0.2):
    filtered_centers = []
    prev_center = np.array(observed_centers[1])
    
    # Initialize the initial state of the filter based on the first observed center
    initial_center = observed_centers[1]
    
    for center in observed_centers:
        if center == [-1, -1]:  # If center is missing
            filtered_center = prev_center 
        else:
            if initial_center == [-1, -1]:  # Handle initial state when the first center is missing
                break
                
            predicted_center = prev_center + alpha * (np.array(center) - prev_center)
            filtered_center = predicted_center + beta * (np.array(center) - predicted_center)
            prev_center = filtered_center
        
        # Ensure the filtered center is within the bounds of the image
        filtered_center = np.clip(filtered_center, [0, 0], [700, 500])
        filtered_centers.append(filtered_center.tolist())
    #print(filtered_centers)
    return filtered_centers
    
def draw_target_object_center(video_file, obj_centers, smoothed_centers):
    count = 0
    cap = cv.VideoCapture(video_file)
    ok, image = cap.read()
    vidwrite = cv.VideoWriter("part_1_demo.mp4", cv.VideoWriter_fourcc(*'MP4V'), 30, (700, 500))
    observed_history = []  # Store the history of observed object centers
    smoothed_track = []  # Collect smoothed track coordinates
    
    for center in smoothed_centers:
        if center[0] != -1 and center[1] != -1:
            smoothed_track.append((int(center[0]), int(center[1])))
    while ok:
        pos_x, pos_y = obj_centers[count]
        count += 1
        image = cv.resize(image, (700, 500))
        
        # Draw the blue expected path line
        if smoothed_track:
            for i in range(len(smoothed_track) - 1):
                if(smoothed_track[i] > smoothed_track[i+1]):
                    cv.line(image, smoothed_track[2], smoothed_track[-1], (255, 0, 0), 1, cv.LINE_AA, 0)
        
      
        
        # Draw smoothed object centers (red circles) only when the object is missing in the current frame
        if count <= len(smoothed_centers):
            smooth_x, smooth_y = smoothed_centers[count - 1]  # Use count - 1 as index since count starts from 1
            #if pos_x == -1 or pos_y == -1:
            if smooth_x != -1 and smooth_y != -1:
                #cv.circle(image, (int(smooth_x), int(smooth_y)), 1, (0, 0, 255), 2)
                #smoothed_track.append((int(smooth_x), int(smooth_y)))  # Append to smoothed track
                observed_history.append((int(smooth_x), int(smooth_y)))
        
        # Draw observed object centers (red circles) for every frame
        observed_history.append((int(pos_x), int(pos_y)))
        for obs_x, obs_y in observed_history:
            cv.circle(image, (obs_x, obs_y), 1, (0, 0, 255), 2)

        vidwrite.write(image)
        ok, image = cap.read()
    vidwrite.release()

    # Write smoothed track coordinates to a JSON file
    with open('part_1_object_tracking.json', 'w') as outfile:
        json.dump({'obj': smoothed_track}, outfile)


frame_dict = load_obj_each_frame("object_to_track.json")
video_file = "commonwealth.mp4"
observed_centers = frame_dict['obj']
# Applying Alpha-Beta filter to smooth the trajectory
smoothed_centers = alpha_beta_filter(observed_centers)

draw_target_object_center(video_file, observed_centers, smoothed_centers)

# part 2:

def draw_object(object_dict, image, color=(0, 255, 0), thickness=2, font=cv.FONT_HERSHEY_SIMPLEX, font_scale=0.5, c_color=(255, 0, 0)):
    # Draw bounding box
    x = object_dict['x_min']
    y = object_dict['y_min']
    width = object_dict['width']
    height = object_dict['height']
    image = cv.rectangle(image, (x, y), (x + width, y + height), color, thickness)

    # Add object ID text
    text = f"ID: {object_dict['id']}"  # Get the unique ID assigned to the object
    text_size, _ = cv.getTextSize(text, font, font_scale, thickness)
    text_x = x + width // 2 - text_size[0] // 2
    text_y = y - 5
    image = cv.putText(image, text, (text_x, text_y), font, font_scale, c_color, thickness)

    return image

def draw_objects_in_video(video_file, frame_dict):
    ids = {}
    cap = cv.VideoCapture(video_file)
    vidwrite = cv.VideoWriter("part_2_demo.mp4", cv.VideoWriter_fourcc(*'MP4V'), 30, (700, 500))
    
    for i in range(len(frame_dict)):
        ok, image = cap.read()
        if not ok:
            break
        image = cv.resize(image, (700, 500))
        obj_list = frame_dict[str(i)]
        
        for obj in obj_list:
            best_id = None
            cur_dist_away = 30  # Threshold distance
            for id, coordinates in ids.items():
                dist_away = math.sqrt(((obj["x_min"] - coordinates[0]) ** 2) + ((obj["y_min"] - coordinates[1]) ** 2))
                if dist_away < cur_dist_away:
                    cur_dist_away = dist_away
                    best_id = id
            if best_id is None:
                next_id = max(ids.keys()) + 1 if ids else 1
                obj["id"] = str(next_id)
                ids[next_id] = (obj['x_min'], obj['y_min'])
            else:
                obj["id"] = str(best_id)
                ids[best_id] = (obj['x_min'], obj['y_min'])
                
            image = draw_object(obj, image)

        vidwrite.write(image)

    cap.release()
    vidwrite.release()
# Example usage:
frame_dict = load_obj_each_frame("frame_dict.json")
video_file = "commonwealth.mp4"
draw_objects_in_video(video_file, frame_dict)


def add_unique_ids_to_objects(frame_dict):
    object_ids = {}  # Dictionary to store object IDs
    for frame_key in frame_dict.keys():
        obj_list = frame_dict[frame_key]
        for obj in obj_list:
            obj_tuple = (obj['x_min'], obj['y_min'], obj['width'], obj['height'])
            if obj_tuple not in object_ids:
                object_ids[obj_tuple] = len(object_ids)  # Assign a unique ID to each object
            obj['id'] = object_ids[obj_tuple]  # Add the 'id' key to the object
    return frame_dict

# Load the original data file
with open("part_2_frame_dict.json", "r") as file:
    frame_dict = json.load(file)

# Modify the data file by adding unique IDs to objects
modified_frame_dict = add_unique_ids_to_objects(frame_dict)

# Save the modified data file
with open("part_2_frame_dict_modified.json", "w") as file:
    json.dump(modified_frame_dict, file, indent=4)