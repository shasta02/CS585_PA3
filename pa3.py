# -*- coding: utf-8 -*-
import json
import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict



#from google.colab.patches import cv2_imshow

# part 1:

def load_obj_each_frame(data_file):
    with open(data_file, 'r') as file:
        frame_dict = json.load(file)
    return frame_dict


def alpha_beta_filter(observed_centers, alpha=0.1, beta=0.1):
    filtered_centers = []
    prev_center = np.array(observed_centers[0])
    for center in observed_centers:
        if center == [-1, -1]:  # If center is missing
            filtered_center = prev_center
        else:
            predicted_center = prev_center + alpha * (np.array(center) - prev_center)
            filtered_center = predicted_center + beta * (np.array(center) - predicted_center)
            prev_center = filtered_center
        filtered_centers.append(filtered_center)
    return filtered_centers


def draw_target_object_center(video_file, obj_centers, smoothed_centers):
    count = 0
    cap = cv.VideoCapture(video_file)
    ok, image = cap.read()
    vidwrite = cv.VideoWriter("part_1_demo_with_filtered_circles.mp4", cv.VideoWriter_fourcc(*'MP4V'), 30, (700, 500))
    smoothed_line = [(int(x), int(y)) for x, y in smoothed_centers if x != -1 and y != -1]
    prev_smoothed = None
    smoothed_track = []  # Collect smoothed track coordinates
    while ok:
        pos_x, pos_y = obj_centers[count]
        count += 1
        image = cv.resize(image, (700, 500))

        # Draw observed track (red circles)
        for i in range(1, count):
            prev_x, prev_y = obj_centers[i-1]
            curr_x, curr_y = obj_centers[i]
            if prev_x != -1 and prev_y != -1 and curr_x != -1 and curr_y != -1:
                cv.circle(image, (int(prev_x), int(prev_y)), 1, (0, 0, 255), 2)

        # Draw smoothed track (blue line)
        if count > 1:
            for smoothed_point in smoothed_line:
                if prev_smoothed is not None:
                    cv.line(image, prev_smoothed, smoothed_point, (255, 0, 0), 1)
                    smoothed_track.append(smoothed_point)  # Add to smoothed track
                prev_smoothed = smoothed_point
        
        cv.circle(image, (int(pos_x), int(pos_y)), 1, (0, 0, 255), 2)  # Red: current observed position
        vidwrite.write(image)
        ok, image = cap.read()
    vidwrite.release()

    # Write smoothed track coordinates to a JSON file
    with open('smoothed_track.json', 'w') as outfile:
        json.dump({'smoothed_track': smoothed_track}, outfile)

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
    count = 0
    cap = cv.VideoCapture(video_file)
    ok, image = cap.read()
    vidwrite = cv.VideoWriter("part_2_demo_with_ids.mp4", cv.VideoWriter_fourcc(*'MP4V'), 30, (700, 500))

    object_ids = defaultdict(int)  # Dictionary to store object IDs
    objects_with_ids = {}  # Dictionary to store objects with assigned IDs

    while ok:
        image = cv.resize(image, (700, 500))
        obj_list = frame_dict[str(count)]

        # Assign unique IDs to objects
        for idx, obj in enumerate(obj_list):
            # Add 'id' key to object dictionary
            obj['id'] = idx
            # Calculate object's unique ID based on its bounding box coordinates
            obj_id = object_ids[(obj['x_min'], obj['y_min'], obj['width'], obj['height'])]
            objects_with_ids[(obj['x_min'], obj['y_min'], obj['width'], obj['height'])] = obj_id
            object_ids[(obj['x_min'], obj['y_min'], obj['width'], obj['height'])] += 1

        # Draw bounding boxes with unique colors based on assigned IDs
        for obj in obj_list:
            obj_id = objects_with_ids[(obj['x_min'], obj['y_min'], obj['width'], obj['height'])]
            color = (0, 0, 255) if obj_id == 0 else (int(255 * obj_id / len(obj_list)), 0, 0)
            image = draw_object(obj, image, color=color)

        vidwrite.write(image)
        count += 1
        ok, image = cap.read()

    vidwrite.release()

# Example usage:
frame_dict = load_obj_each_frame("frame_dict.json")
video_file = "commonwealth.mp4"
draw_objects_in_video(video_file, frame_dict)