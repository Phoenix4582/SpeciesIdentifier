import os
import cv2
import csv
import math
import numpy as np
import shutil
from datetime import datetime, timedelta

def reset_folder(dest):
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.mkdir(dest)

def handle_frames(frames, line, dest, timestamps):
    """
    For every line from csv, crop the frame target with given coordinates.
    Add implementation of writing a new csv under the subfolder to record timestamp info.
    """
    target_frames = {}
    rect_bbox = []
    selected_timestamps = []

    rotated_bbox_info = [rbox for rbox in line if len(eval(rbox)) != 0]
    for id, (frame, rbox, stamp) in enumerate(zip(frames, line, timestamps)):
        rect = convert_box(rbox)
        if rect is not None:
            target_frames.update({id:frame})
            rect_bbox.append(rect)
            selected_timestamps.append(stamp)

    # rect_bbox = [convert_box(rbox) for rbox in line]
    id = 0
    for (key, frame), rect in zip(target_frames.items(), rect_bbox):
        # current_timestamp = timestamps[timestamp_id]
        # timestamp_id = timestamp_id + 1
        # code modified from: https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/
        img = cv2.imread(frame)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # test if bbox is correct, usually omitted
        fused_img = cv2.drawContours(img, [box], 0, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(dest, "base_{}.jpg".format(str(key).zfill(5))), fused_img)

        # get width and height of the detected rectangle
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        # coordinate of the points in box points after the rectangle has been
        # straightened
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(img, M, (width, height))

        # force the rotated rectangle to format horizontally
        # may be omitted
        if not width > height:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        cv2.imwrite(os.path.join(dest, "{}.jpg".format(str(id).zfill(5))), warped)
        id = id + 1

        # append the timestamp to the list
        # selected_timestamps.append(current_timestamp)

    timestamp_csv = os.path.join(dest, "timestamps.csv")
    # header = [str(i).zfill(5) for i in range(id)]
    with open(timestamp_csv, "w+") as f:
        # header_writer = csv.DictWriter(f, fieldnames=header)
        # header_writer.writeheader()
        writer = csv.writer(f)
        writer.writerow(selected_timestamps)
        writer.writerow(rotated_bbox_info)

def convert_box(rbox):
    """
    Convert (cx, cy, w, h, theta) into rect coordinates. COUNTERCLOCKWISE as positive for theta.
    """
    rbox = eval(rbox)
    if len(rbox) != 0:
        # print(rbox)
        center_x, center_y = rbox[0], rbox[1]
        width, height, theta = rbox[2], rbox[3], rbox[4]
        theta = math.degrees(-1 * theta)

        # x_min = round(center_x - 0.5 * (width * math.cos(theta) + height * math.sin(theta)), 3)
        # y_min = round(center_y + 0.5 * (width * math.sin(theta) - height * math.cos(theta)), 3)
        #
        # x1, y1 = x_min, y_min
        # x2, y2 = round(x_min + height * math.sin(theta), 3), round(y_min + height * math.cos(theta), 3)
        # x4, y4 = round(x_min + width * math.cos(theta), 3), round(y_min - width * math.sin(theta), 3)
        # x3, y3 = round(x4 + height * math.sin(theta), 3), round(y4 + height * math.cos(theta), 3)
        rect = ((center_x, center_y), (width, height), theta)
        # rect_coords = np.array([[[x1, y1]], [[x2, y2]], [[x3, y3]], [[x4, y4]]], dtype='int32')
        # rect = cv2.minAreaRect(rect_coords)

        return rect
    else:
        return None

def crop_samples(root, dest):
    reset_folder(dest)
    frames = [os.path.join(root, img) for img in os.listdir(root) if img.endswith('.jpg')]
    csv_source = [os.path.join(root, c) for c in os.listdir(root) if c.endswith('.csv')]
    if len(csv_source) == 0:
        raise ValueError("No csv file found!")
    csv_source = csv_source[0]
    with open(csv_source, mode='r') as f:
        csv_dict_reader = csv.DictReader(f)
        timestamps = csv_dict_reader.fieldnames
        csv_file = csv.reader(f, delimiter=',')
        # next(csv_file, None)
        for id, line in enumerate(csv_file):
            subdir = os.path.join(dest, str(id).zfill(3))
            reset_folder(subdir)
            handle_frames(frames, line, subdir, timestamps)

def parse_time(video_filename:str):
    video_name = video_filename.split("/")[-1]
    # assert (video_name.endswith(".mp4"), "MP4 format only.")
    start_datetime = video_name[:-4]
    # print(start_datetime)
    start_date, start_time = start_datetime.split("T")[0], start_datetime.split("T")[-1]
    year, month, day = tuple(start_date.split("-"))
    hour, minute, second = tuple(start_time.split("-"))
    converted_datetime = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    # print(converted_datetime)
    return converted_datetime

def calculate_time(start:datetime, steps:float):
    result = start + timedelta(seconds=steps)
    return result

if __name__ == '__main__':
    # multi_mode = False
    # root = '../datasets/base/RGB_1/'
    # dest = '../datasets/cropped/RGB_1/'
    # try:
    #     crop_samples(root, dest)
    # except ValueError as e:
    #     print(e)
    # else:
    #     print(f"Successfully cropped targets under {dest} folder.")
    video_path = "../../Dataset_Preparation/wynd_videos/66.126/2023Mar21/2023-03-21T13-32-34.mp4"
    video_datetime = parse_time(video_path)
    print(calculate_time(video_datetime, 50))
