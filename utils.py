import os.path
import time
import json
import warnings
import signal
from datetime import datetime
from contextlib import contextmanager
from PIL import Image, ImageDraw
import requests
import numpy as np
import math
import torch
import cv2
# from nms import nms
from utils_iou import nms, polygon_iou
from utils_video import crop_samples, parse_time, calculate_time
import csv
import copy


def order_points(pts):
    pts_reorder = []

    for idx, pt in enumerate(pts):
        idx = torch.argsort(pt[:, 0])
        xSorted = pt[idx, :]
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        leftMost = leftMost[torch.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        D = torch.cdist(tl[np.newaxis], rightMost)[0]
        (br, tr) = rightMost[torch.argsort(D, descending=True), :]
        pts_reorder.append(torch.stack([tl, tr, br, bl]))

    return torch.stack([p for p in pts_reorder])

def rotate_boxes(boxes, points=False):
    '''
    Rotate target bounding boxes

    Input:
        Target boxes (xmin_ymin, width_height, theta)
    Output:
        boxes_axis (xmin_ymin, xmax_ymax, theta)
        boxes_rotated (xy0, xy1, xy2, xy3)
    '''

    # u = torch.stack([torch.cos(boxes[:,4]), torch.sin(boxes[:,4])], dim=1)
    # l = torch.stack([-torch.sin(boxes[:,4]), torch.cos(boxes[:,4])], dim=1)
    # R = torch.stack([u, l], dim=1)

    u = torch.stack([torch.cos(boxes[:,4]), -torch.sin(boxes[:,4])], dim=1)
    l = torch.stack([torch.sin(boxes[:,4]), torch.cos(boxes[:,4])], dim=1)
    R = torch.stack([u, l], dim=1)

    if points:
        cents = torch.stack([(boxes[:,0]+boxes[:,2])/2, (boxes[:,1]+boxes[:,3])/2],1).transpose(1,0)
        boxes_rotated = torch.stack([boxes[:,0],boxes[:,1],
            boxes[:,2], boxes[:,1],
            boxes[:,2], boxes[:,3],
            boxes[:,0], boxes[:,3],
            boxes[:,-2],
            boxes[:,-1]],1)

    else:
        cents = torch.stack([boxes[:,0]+(boxes[:,2])/2, boxes[:,1]+(boxes[:,3])/2],1).transpose(1,0)
        boxes_rotated = torch.stack([boxes[:,0],boxes[:,1],
            (boxes[:,0]+boxes[:,2]), boxes[:,1],
            (boxes[:,0]+boxes[:,2]), (boxes[:,1]+boxes[:,3]),
            boxes[:,0], (boxes[:,1]+boxes[:,3]),
            boxes[:,-2],
            boxes[:,-1]],1)

    xy0R = torch.matmul(R,boxes_rotated[:,:2].transpose(1,0) - cents) + cents
    xy1R = torch.matmul(R,boxes_rotated[:,2:4].transpose(1,0) - cents) + cents
    xy2R = torch.matmul(R,boxes_rotated[:,4:6].transpose(1,0) - cents) + cents
    xy3R = torch.matmul(R,boxes_rotated[:,6:8].transpose(1,0) - cents) + cents

    xy0R = torch.stack([xy0R[i,:,i] for i in range(xy0R.size(0))])
    xy1R = torch.stack([xy1R[i,:,i] for i in range(xy1R.size(0))])
    xy2R = torch.stack([xy2R[i,:,i] for i in range(xy2R.size(0))])
    xy3R = torch.stack([xy3R[i,:,i] for i in range(xy3R.size(0))])

    boxes_axis = torch.cat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:4] - 1,
        torch.sin(boxes[:,-1, None]), torch.cos(boxes[:,-1, None])], 1)
    boxes_rotated = order_points(torch.stack([xy0R,xy1R,xy2R,xy3R],dim = 1)).view(-1,8)

    return boxes_axis, boxes_rotated


def rotate_box(bbox):
    xmin, ymin, width, height, theta = bbox

    xy1 = xmin, ymin
    xy2 = xmin, ymin + height - 1
    xy3 = xmin + width - 1, ymin + height - 1
    xy4 = xmin + width - 1, ymin

    cents = np.array([xmin + (width - 1) / 2, ymin + (height - 1) / 2])

    corners = np.stack([xy1, xy2, xy3, xy4])

    # u = np.stack([np.cos(theta), -np.sin(theta)])
    # l = np.stack([np.sin(theta), np.cos(theta)])
    # R = np.vstack([u, l])

    u = np.stack([np.cos(theta), np.sin(theta)])
    l = np.stack([-np.sin(theta), np.cos(theta)])
    R = np.vstack([u, l])

    corners = np.matmul(R, (corners - cents).transpose(1, 0)).transpose(1, 0) + cents

    return corners.reshape(-1).tolist()


def show_detections(detections):
    'Show image with drawn detections'

    for image, detections in detections.items():
        im = Image.open(image).convert('RGBA')
        overlay = Image.new('RGBA', im.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        detections.sort(key=lambda d: d['score'])
        for detection in detections:
            box = detection['bbox']
            alpha = int(detection['score'] * 255)
            draw.rectangle(box, outline=(255, 255, 255, alpha))
            draw.text((box[0] + 2, box[1]), '[{}]'.format(detection['class']),
                      fill=(255, 255, 255, alpha))
            draw.text((box[0] + 2, box[1] + 10), '{:.2}'.format(detection['score']),
                      fill=(255, 255, 255, alpha))
        im = Image.alpha_composite(im, overlay)
        im.show()


def save_detections(path, detections):
    print('Writing detections to {}...'.format(os.path.basename(path)))
    with open(path, 'w') as f:
        json.dump(detections, f)


@contextmanager
def ignore_sigint():
    handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, handler)


class Profiler(object):
    def __init__(self, names=['main']):
        self.names = names
        self.lasts = {k: 0 for k in names}
        self.totals = self.lasts.copy()
        self.counts = self.lasts.copy()
        self.means = self.lasts.copy()
        self.reset()

    def reset(self):
        last = time.time()
        for name in self.names:
            self.lasts[name] = last
            self.totals[name] = 0
            self.counts[name] = 0
            self.means[name] = 0

    def start(self, name='main'):
        self.lasts[name] = time.time()

    def stop(self, name='main'):
        self.totals[name] += time.time() - self.lasts[name]
        self.counts[name] += 1
        self.means[name] = self.totals[name] / self.counts[name]

    def bump(self, name='main'):
        self.stop(name)
        self.start(name)


def post_metrics(url, metrics):
    try:
        for k, v in metrics.items():
            requests.post(url,
                          data={'time': int(datetime.now().timestamp() * 1e9),
                                'metric': k, 'value': v})
    except Exception as e:
        warnings.warn('Warning: posting metrics failed: {}'.format(e))


def capture_frames(video_source, dest_dir, freq):
    """
    Main function with the goal of capturing image frames at certain interval.
    input:
    args: arguments from ArgumentParser
    """
    try:
        os.mkdir(dest_dir)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(video_source)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    total = 0
    print (f"Converting video at frequency {freq} frame(s) interval..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            count = count + 1
            continue
        # Write the results back to output location.
        # Capture frames at a frequency of args.freq
        if (count % freq == 0):
            cv2.imwrite(dest_dir + "/%#05d.jpg" % (total), frame)
            total = total + 1
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print (f"Done extracting frames to {dest_dir}.\n{total} frames extracted")
            print (f"It took {time_end-time_start} seconds for conversion.")
            break

# def nms_mod(rbox_dataset, nms_threshold, debug=True):
#     """
#     Perform Non-Maximum Suppression(NMS) on inference. Threshold can be varied.
#     Input:
#     rbox_dataset: json-formatted inference dataset
#     nms_threshold: a variable to control the performance of NMS.
#     debug: boolean to flip angle direction
#
#     Output: the filtered inference dataset
#     """
#     result = {}
#     rboxes = [rbox['bbox'] for rbox in rbox_dataset]
#     scores = [rbox['score'] for rbox in rbox_dataset]
#     rbox_centered = []
#     for rbox in rboxes:
#         x, y, w, h, theta = rbox
#         if debug:
#             theta = -1 * theta
#         cx, cy = x + 0.5 * w, y + 0.5 * h
#         rbox_centered.append(((cx, cy), (w, h), theta))
#
#     indices = nms.rboxes(rbox_centered, scores, nms_threshold=nms_threshold)
#     filtered_dataset = [rbox_dataset[id] for id in indices]
#     return filtered_dataset

def nms_mod(rbox_dataset, nms_threshold, debug=True):
    """
    Perform Non-Maximum Suppression(NMS) on inference (or ground truth data for testing) per image. Threshold can be varied.
    Input:
    rbox_dataset: json-formatted inference dataset
    nms_threshold: a variable to control the performance of NMS.
    debug: boolean to flip angle direction

    Output: the filtered inference dataset
    """
    result = {}

    polygons = [rbox['segmentation'][0] for rbox in rbox_dataset]
    polygons = [[[polygon[0], polygon[1]], [polygon[2], polygon[3]], [polygon[4], polygon[5]], [polygon[6], polygon[7]]] for polygon in polygons]
    scores = [rbox['score'] if 'score' in rbox else 1 for rbox in rbox_dataset]
    indices = nms(polygons, scores, nms_threshold=nms_threshold)
    filtered_dataset = [rbox_dataset[id] for id in indices]
    return filtered_dataset

def save_video_inference(video_path, dest_dir, detections_file, freq, nms_thr, iou_thr, show_box, show_track, save_details):
    frame_ids = sorted([img.strip(".jpg") for img in os.listdir(dest_dir) if img.endswith(".jpg")])
    # frames = [cv2.imread(os.path.join(dest_dir, '{}.jpg'.format(id))) for id in frame_ids]
    # frame_ids = sorted([int(id) for id in frame_ids])
    # print(*frame_ids)
    # assert len(frame_ids) == len(frames)

    sample_frame = cv2.imread(os.path.join(dest_dir, '{}.jpg'.format(frame_ids[0])))
    height, width, layers = sample_frame.shape
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tracklet_info = {}
    global new_entry
    new_entry = []

    # For tracklets generation, mere spatial distance may not be enough.
    global DISTANCE
    DISTANCE = 120 # test with 80, 120, 150, 200, 300, 320, 360(X)
    global IOU_DISTANCE
    IOU_DISTANCE = 0.3

    # static_infer_dir = os.path.join(dest_dir, "static_infers")
    # if not os.path.exists(static_infer_dir):
    #     os.makedirs(static_infer_dir)

    current = get_time()
    if show_box and show_track:
        current = current + "_full"
    elif show_box:
        current = current + "_box"
    elif show_track:
        current = current + "_track"
    gen_video_dest = os.path.join(dest_dir, f"captured_video_{current}.mp4")

    print(f"Begin generating video inference......")
    if os.path.exists(gen_video_dest):
        os.remove(gen_video_dest)

    out = cv2.VideoWriter(gen_video_dest, fourcc, 30, size)

    for id in frame_ids:
        frame = cv2.imread(os.path.join(dest_dir, '{}.jpg'.format(id)))
        info = get_inference(detections_file, id, nms_thr)
        tracklet_info = update_info(info, tracklet_info)
        new_frame = augment_frame(frame, info, tracklet_info, show_box, show_track)
        out.write(new_frame)
        # cv2.imwrite(os.path.join(static_infer_dir, '{}.jpg'.format(id)), new_frame)

    out.release()
    print(f"Successfully generated video at {gen_video_dest}")

    # NEED UPDATE IN THE FUTURE
    if save_details:
        # Log Apr 25 2023: implement timestamp recording for csv file generation
        try:
            start_time = parse_time(video_path)
        except Exception:
            print("Unable to parse datetime information from video file.")
            print("Expected format: (year:xxxx)-(month:xx)-(day:xx)T(hour:xx)-(minute:xx)-(second:xx).mp4")
            print("Try recover datetime info and embed into video filename.")

        csv_dest = os.path.join(dest_dir, 'tracklet_{}.csv'.format(current))
        print("Generating csv file for tracklet info storage......")
        write_to_csv(csv_dest, start_time, len(frame_ids), freq, tracklet_info)
        print(f"Successfully wrote tracklet csv at {csv_dest}")
        cropped_dest = os.path.join(dest_dir, 'cropped')
        print("Cropping targets from tracklet info......")
        crop_samples(dest_dir, cropped_dest)
        print(f"Successfully gathered cropped targets under {cropped_dest}")
        if freq < 50:
            print("Clearing cached images to save space.....")
            clear_cached_images(dest_dir)


    print("All procedures done.")

def clear_cached_images(dir):
    frames = [os.path.join(dir, img) for img in os.listdir(dir) if img.endswith(".jpg")]
    for frame in frames:
        os.remove(frame)

def get_inference(detections_file, id, nms_thr) -> list:
    with open(detections_file, 'r') as json_source:
        detection_contents = json.load(json_source)
        proposals = [anno for anno in detection_contents['annotations'] if anno['image_id'] == int(id)]
        nms_filtered_proposals = nms_mod(proposals, nms_thr)
    return nms_filtered_proposals

def get_time():
    current = datetime.now()
    current_YMD = "{}{}{}".format(str(current.year).zfill(2), str(current.month).zfill(2), str(current.day).zfill(2))
    current_HMS = "{}{}{}".format(str(current.hour).zfill(2), str(current.minute).zfill(2), str(current.second).zfill(2))
    return f"{current_YMD}_{current_HMS}"

def update_info(info: list, tracklet_info: dict):
    """
    Update inference information for two domains:
    Inference Illustration(dots_dict): Save a temporary dictionary for visual representation of video inference.(Omitted since tracklet_info contains everything)
    Tracklet Recording(tracklet_info): Record all tracklet info ([cx, cy, w, h, theta] per element) for outputting a csv file.
    """
    bboxes = [anno['bbox'] for anno in info]
    tracklet_elements = find_center_coords(bboxes)

    # Section I: Inference Illustration && Section II: Tracklet Recording
    if len(tracklet_info.keys()) == 0:
        if len(tracklet_elements) != 0:
            for id, element in enumerate(tracklet_elements):
                new_entry_with_id = copy.deepcopy(new_entry)
                new_entry_with_id.append(element)
                tracklet_info.update({id:new_entry_with_id})
        new_entry.append(())
    else:
        if len(tracklet_elements) == 0:
            for key, value in tracklet_info.items():
                value.append(())
                tracklet_info.update({key:value})
        else:
            tracklet_info = pair_centers_with_dict(tracklet_elements, tracklet_info, calculate_distance)
            # tracklet_info = pair_centers_with_dict(tracklet_elements, tracklet_info, calculate_iou)

    return tracklet_info

# (Fixed Apr 30 2023)CONTAIN CRITICAL ERROR WHERE FRAME(WITH TIMESTAMPS) AND BBOX(WITH INFERENCE) DO NOT MATCH
def pair_centers_with_dict(centers: list, dots_dict: dict, policy):
    """
    # For each new center entry, find nearest previous dot,
    # Ordering of dots and centers are randomized, 2D np.array is required, focus on minimum distance in 2D array
    # Both Inference Illustration and Tracklet Recording section uses this module
    # Policy stands for the criterion used for selecting tracklets, require a function which take in two rboxes(cx, cy, w, h, angle) and return some distance(score)
    # Criteria may be adjusted for min or max
    """
    # element_length = len(list(dots_dict.values())[0])
    old_num_keys = len(list(dots_dict.keys()))
    element_length = max([len(value) for value in list(dots_dict.values())])
    # latest_boxes = {k: last_valid_box(dots_dict, k) for k in dots_dict.keys()}
    # latest_valid_boxes = {k: v[-1] for k, v in dots_dict.items() if len(v[-1]) != 0}
    latest_valid_boxes = {k: find_latest_nth_bbox(v) for k, v in dots_dict.items() if len(find_latest_nth_bbox(v)) != 0}

    queries = [center for center in centers]
    remainders = [center for center in centers]
    query_length = len(remainders)
    criteria = np.amin if policy == calculate_distance else np.amax
    # threshold = DISTANCE if policy == calculate_distance else IOU_DISTANCE
    infinity = float('inf') if policy == calculate_distance else float('-inf')
    for query in queries:
        if len(latest_valid_boxes.values()) == 0:
            break
        distance_table = [policy(box, query) for box in latest_valid_boxes.values()]
        # print(distance_table)
        distance_table = np.array(distance_table, dtype=np.float32)
        dict_keys = list(latest_valid_boxes.keys())
        found = False
        # available = True
        while (not found):
            best_distance = criteria(distance_table)
            if best_distance == infinity:
                # available = False
                break
            occurences = np.where(distance_table == best_distance)
            best_prev_box_id = list(occurences[0])[0]
            entry_query = dots_dict[dict_keys[best_prev_box_id]]

            # Only one bounding box available, possible for any directions
            # print("Entry query ", entry_query)
            if not_existing_tracklet(entry_query):
                found = True
            # Multiple bounding boxes available from a tracklet, check if the new entry follows the direction
            else:
                last_valid_entry_query = find_latest_nth_bbox(entry_query)
                second_last_valid_entry_query = find_latest_nth_bbox(entry_query, nth=2)
                vector_new = ((query[0] - last_valid_entry_query[0]), (query[1] - last_valid_entry_query[1]))
                vector_old = ((last_valid_entry_query[0] - second_last_valid_entry_query[0]),(last_valid_entry_query[1] - second_last_valid_entry_query[1]))
                if (math.fabs(angle_between(vector_new, vector_old)) > (math.pi / 4)) and calculate_iou(query, last_valid_entry_query) < 0.5:
                    distance_table[best_prev_box_id] = infinity
                else:
                    found = True

        if found and get_threshold(policy, best_distance):
            dots_dict[dict_keys[best_prev_box_id]].append(query)
            del latest_valid_boxes[dict_keys[best_prev_box_id]]
        else:
            # print("Inserted New Key Directly")
            new_dict_key = max([key for key in dots_dict.keys()]) + 1
            prev_elements = [() for _ in range(element_length)]
            prev_elements.append(query)
            dots_dict.update({new_dict_key: prev_elements})

        del remainders[0]

    for key in latest_valid_boxes.keys():
        prev_elements = dots_dict[key]
        prev_elements.append(())
        dots_dict.update({key: prev_elements})

    for remainder in remainders:
        largest_key = max([key for key in dots_dict.keys()])
        prev_elements = [() for _ in range(element_length)]
        prev_elements.append(remainder)
        dots_dict.update({(largest_key+1): prev_elements})

    # while (len(latest_boxes.keys()) != 0 and len(remainders) != 0):
    #     dots_centers_distances_table = np.array([[policy(box, center) if len(box) != 0 else infinity for box in latest_boxes.values()] for center in remainders], dtype=np.float32)
    #     dict_keys = list(latest_boxes.keys())
    #     found = False
    #     available = True
    #     while ((not found) and available):
    #         best_distance = criteria(dots_centers_distances_table)
    #         if best_distance == infinity:
    #             available = False
    #         occurences = np.where(dots_centers_distances_table == best_distance)
    #         first_occurence = list(zip(occurences[0], occurences[1]))[0]
    #         best_id, best_prev_box_id = first_occurence
    #         entry_query = dots_dict[dict_keys[best_prev_box_id]]
    #
    #         # Only one bounding box available, possible for any directions
    #         if not_existing_tracklet(entry_query):
    #             found = True
    #         # Multiple bounding boxes available from a tracklet, check if the new entry follows the direction
    #         else:
    #             vector_new = ((remainders[best_id][0] - entry_query[-1][0]), (remainders[best_id][1] - entry_query[-1][1]))
    #             vector_old = ((entry_query[-1][0] - entry_query[-2][0]),(entry_query[-1][1] - entry_query[-2][1]))
    #             if math.fabs(angle_between(vector_new, vector_old)) > (math.pi / 4):
    #                 dots_centers_distances_table[best_id, best_prev_box_id] = infinity
    #             else:
    #                 found = True
    #
    #     if found and get_threshold(policy, best_distance):
    #         dots_dict[dict_keys[best_prev_box_id]].append(remainders[best_id])
    #     else:
    #         # print("Inserted New Key Directly")
    #         new_dict_key = max([key for key in dots_dict.keys()]) + 1
    #         prev_elements = [() for _ in range(element_length)]
    #         prev_elements.append(remainders[best_id])
    #         dots_dict.update({new_dict_key: prev_elements})
    #
    #     del latest_boxes[dict_keys[best_prev_box_id]]
    #     del remainders[best_id]
    #
    # if len(remainders) != 0:
    #     # print("Inserted New Key From Remainders")
    #     for remainder in remainders:
    #         largest_key = max([key for key in dots_dict.keys()])
    #         prev_elements = [() for _ in range(element_length)]
    #         prev_elements.append(remainder)
    #         dots_dict.update({(largest_key+1): prev_elements})

    # assert(len(dots_dict.keys()) == max(old_num_keys, query_length))
    return dots_dict

def find_latest_nth_bbox(entry:list, nth:int = 1, boundary:int = 1):
    """
    For a list containing bbox queries, return the last nth valid bbox found, or () if not found
    """
    reverser = slice(None, None, -1)
    reversed_entry = entry[reverser]
    count = 0
    for item in reversed_entry[:boundary]:
        if len(item) != 0:
            count += 1
            if count == nth:
                return item
    return ()

def get_threshold(policy, best_distance):
    if policy == calculate_distance:
        return best_distance < DISTANCE
    else:
        return best_distance > IOU_DISTANCE

def not_existing_tracklet(query):
    if len(query) == 1:
        return True
    if all(len(q) == 0 for q in query):
        return True
    if len(find_latest_nth_bbox(query)) != 0 and len(find_latest_nth_bbox(query, nth=2)) == 0:
        return True
    # if all(len(q) == 0 for q in query[:-1]) and len(query[-1]) != 0:
    #     return True
    # assert(len(query[-1]) != 0)
    # return len(query[-2]) == 0
    return False

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def last_valid_box(dots_dict, key):
    coord_values = dots_dict[key]
    valid_values = list(filter(lambda x: len(x) != 0, coord_values))
    latest_valid_value = list(valid_values[-1])
    return latest_valid_value

def augment_frame(frame, info, dots_dict, show_box, show_track):
    height, width, layers = frame.shape
    circle_radius = 10
    circle_color = (200, 213, 48) # Torquoise for circle
    circle_thinkness = -1 # Filled Circle

    line_color = (0, 128, 255) # Orange for line
    line_thickness = 5

    if show_box:
        frame = draw_bbox(frame, info, debug=True)

    if show_track:
        frame = fill_prev_markings(frame, dots_dict, circle_radius, circle_color, circle_thinkness, line_color, line_thickness)

    return frame

def fill_prev_markings(frame, dots_dict,
                       circle_radius, circle_color, circle_thinkness,
                       line_color, line_thickness):
    for _, dots in dots_dict.items():
        prev_valid_center = None
        for id, dot in enumerate(dots):
            # draw dots and connect lines if new dot is available
            if len(dots) != 0:
                circle_center = (int(dots[0]), int(dots[1]))
                frame = cv2.circle(frame, circle_center, circle_radius, circle_color, circle_thinkness)
                # draw line
                if prev_valid_center is not None:
                    frame = cv2.line(frame, prev_valid_center, circle_center, line_color, line_thickness)

                prev_valid_center = circle_center

    return frame

# For results from OpenCV2(from GT), theta is positive as CLOCKWISE, testing over roLabelimg(GT original) is needed
# roLabelimg is limited within (0, pi), CLOCKWISE as positive
# May not need to look into roLabelimg direction, as involvement of head coordinates will correct the angle
# Only need conversion from clockwise to counterclockwise
# For results from inference(IR), theta is positive as COUNTERCLOCKWISE
def draw_bbox(img, dataset, debug):
    if len(dataset) == 0:
        return img

    color = (226, 181, 0)
    arrow_color = (255, 111, 255)
    for data in dataset:
        label = "IR"
        x, y, w, h, theta = data['bbox']
        if debug:
            theta = -1 * theta
        x_center = int(x + 0.5 * w)
        y_center = int(y + 0.5 * h)
        pointer = max(w, h)
        pt1 = (x_center, y_center)
        pt2 = (int(x_center + 0.5*(pointer+100)*math.cos(theta)), int(y_center + 0.5*(pointer+100)*math.sin(theta)))

        label = label + ":{}%".format(round(data['score'] * 100, 3))

        angle = math.degrees(theta)
        rot_rectangle = ((x_center, y_center), (w, h), angle)
        box = cv2.boxPoints(rot_rectangle)
        box = np.int0(box)

        infused_img = cv2.drawContours(img, [box], 0, color, 2)
        infused_img = cv2.arrowedLine(infused_img, pt1, pt2, arrow_color, 2)
        # infused_img = cv2.polylines(img, [points], True, color, 2)
        # infused_img = cv2.putText(infused_img, label, (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)

        ((txt_w, txt_h), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)
        x1 = int(box[0][0])
        y1 = int(box[0][1])
        infused_img = cv2.rectangle(infused_img, (x1, y1), (x1 + int(txt_w), y1 + int(txt_h*1.8)), color, thickness=-1)
        infused_img = cv2.putText(infused_img, label, (x1, y1 + int(txt_h*1.5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    return infused_img

# def find_center_coords(bboxes, tracklet_info=False):
#     if tracklet_info:
#         return [(box[0] + 0.5 * box[2], box[1] + 0.5 * box[3], box[2], box[3], box[4]) for box in bboxes]
#     return [(int(box[0] + 0.5 * box[2]), int(box[1] + 0.5 * box[3])) for box in bboxes]

def find_center_coords(bboxes):
    return [(box[0] + 0.5 * box[2], box[1] + 0.5 * box[3], box[2], box[3], box[4]) for box in bboxes]

# Policies
def calculate_distance(box1, box2):
    x1, y1 = box1[0], box1[1]
    x2, y2 = box2[0], box2[1]
    return ((x1-x2)**2 + (y1-y2)**2) ** 0.5

def calculate_iou(box1, box2):
    polygon1 = convert_rbox_to_polygon(box1)
    polygon2 = convert_rbox_to_polygon(box2)
    return polygon_iou(polygon1, polygon2)

def convert_rbox_to_polygon(rbox):
    """
    Convert a rotated bbox (cx, cy, w, h, angle) to a Polygon
    """
    center_x, center_y, width, height, theta = rbox
    theta = -1 * theta
    x_min = round(center_x - 0.5 * (width * math.cos(theta) + height * math.sin(theta)), 3)
    y_min = round(center_y + 0.5 * (width * math.sin(theta) - height * math.cos(theta)), 3)

    x1, y1 = x_min, y_min
    x2, y2 = round(x_min + height * math.sin(theta), 3), round(y_min + height * math.cos(theta), 3)
    x4, y4 = round(x_min + width * math.cos(theta), 3), round(y_min - width * math.sin(theta), 3)
    x3, y3 = round(x4 + height * math.sin(theta), 3), round(y4 + height * math.cos(theta), 3)
    return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]


def write_to_csv(file_name, start_time, num_frames, freq, contents: dict):
    """
    Write the pre-recorded tracklet information onto a csv file.
    Contents should already contain all tracklets plus their coordinates by frames
    (keep csv cell empty if specific tracklet(s) are not found)
    """
    # Video FPS=25, may double check video metadata
    header = [str(calculate_time(start_time, frame*freq / 25)) for frame in range(num_frames)]
    full_length = len(header)
    # list of keys
    keys = contents.keys()
    # list of values
    bbox_info = list(contents.values())
    # bbox_info = list(zip(*contents.values()))
    with open(file_name, 'w+', encoding='UTF8') as f:
        header_writer = csv.DictWriter(f, fieldnames=header)
        header_writer.writeheader()
        writer = csv.writer(f)
        for info in bbox_info:
            info_length = len(info)
            if info_length < full_length:
                for _ in range(full_length - info_length):
                    info.append(())

            writer.writerow(info)
