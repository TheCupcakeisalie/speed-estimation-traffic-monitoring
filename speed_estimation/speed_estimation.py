from datetime import datetime
from importlib import reload
from multiprocessing import Pool
import sys
import cv2
import os
import configparser
from modules.object_detection.yolov5.object_detection import (
    ObjectDetection as ObjectDetectionCustom,
)
from modules.object_detection.yolov4.object_detection import ObjectDetection
import math
from get_fps import give_me_fps
import logging
import json
from modules.shake_detection.shake_detection import ShakeDetection
from paths import session_path
import time
import uuid
from utils.speed_estimation import (
    Direction,
    TrackingBox,
    Car,
    calculate_car_direction,
)
from modules.evaluation.evaluate import plot_absolute_error
from modules.depth_map.depth_map_utils import DepthModel
from collections import defaultdict
from modules.scaling_factor.scaling_factor_extraction import (
    GeometricModel,
    CameraPoint,
    get_ground_truth_events,
    offline_scaling_factor_estimation_from_least_squares,
)
import torch
from typing import Dict, List
from tqdm import tqdm

config = configparser.ConfigParser()
config.read("speed_estimation/config.ini")


def run(
    path_to_video: str,
    data_dir: str,
    fps: int = 0,
    max_frames: int = 0,
    custom_object_detection: bool = False,
):
    reload(logging)

    run_id = uuid.uuid4().hex[:10]
    print(f"Run No.: {run_id}")

    # Initialize logging
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_name = f"logs/{now_str}_run_{run_id}.log"
    logging.basicConfig(
        filename=f"logs/{now_str}_run_{run_id}.log", level=logging.DEBUG
    )
    logging.info(f"Run No.: {run_id}, Video: {data_dir}")

    start = time.time()

    # Initialize Object Detection
    if custom_object_detection:
        weights = "speed_estimation/model_weights/yolov5/best.pt"
        od = ObjectDetectionCustom(weights=weights)
    else:
        od = ObjectDetection()

    input_video = cv2.VideoCapture(path_to_video)

    fps = give_me_fps(path_to_video) if fps == 0 else fps

    SLIDING_WINDOW_SEC = config.getint("main", "sliding_window_sec")
    sliding_window = SLIDING_WINDOW_SEC * fps

    # Initialize running variables
    frame_count = 0
    track_id = 0
    tracking_objects: Dict[int, TrackingBox] = {}
    tracked_cars: Dict[int, Car] = {}
    tracked_boxes: Dict[int, List[TrackingBox]] = defaultdict(list)
    depth_model = DepthModel(data_dir)
    geo_model = GeometricModel(depth_model)
    is_calibrated = False
    text_color = (255, 255, 255)

    MAX_TRACKING_MATCH_DISTANCE = config.getint("tracker", "max_match_distance")
    CAR_CLASS_ID = config.getint("tracker", "car_class_id")

    NUM_TRACKED_CARS = config.getint("calibration", "num_tracked_cars")
    NUM_GT_EVENTS = config.getint("calibration", "num_gt_events")

    # meta_appr
    AVG_FRAME_COUNT = config.getfloat("analyzer", "avg_frame_count")
    SPEED_LIMIT = config.getint("analyzer", "speed_limit")

    # for shake_detection
    shake_detection = ShakeDetection()

    if custom_object_detection:
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    progress_bar = tqdm(total=NUM_TRACKED_CARS)
    progress_bar.set_description("Calibrating")

    while True:
        ############################
        # load frame, shake detection and object detection
        ############################
        ret, frame = input_video.read()

        if frame_count == 0:
            # set normalization axes once at beginning
            c_u = int(frame.shape[1] / 2)
            c_v = int(frame.shape[0] / 2)
            geo_model.set_normalization_axes(c_u, c_v)

        if not ret:
            break

        if custom_object_detection:
            frame = fgbg.apply(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            path_to_frame = f"speed_estimation/frames_detected/frame_{run_id}.jpg"
            cv2.imwrite(path_to_frame, frame)

        # for shake_detection
        if shake_detection.is_hard_move(frame):
            logging.info(
                f"Run No.: {run_id}, Video: {data_dir}, Hard Move Detected Frame: {frame_count}"
            )

        ############################
        # Detect cars on frame
        ############################
        if custom_object_detection:
            boxes = od.detect(path_to_frame)
            if len(boxes) == 0:
                continue
        else:
            # TODO: look into scores
            (class_ids, scores, boxes) = od.detect(frame)
            boxes = [boxes[i] for i, class_id in enumerate(class_ids) if class_id == CAR_CLASS_ID]

        # collect tracking boxes
        tracking_boxes_cur_frame: List[TrackingBox] = []
        for box in boxes:
            (x, y, w, h) = box.astype(int)
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)

            tracking_boxes_cur_frame.append(
                TrackingBox(cx, cy, x, y, w, h, frame_count)
            )
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        ############################
        # assign tracking box IDs
        ############################
        for object_id, tracking_box_prev in tracking_objects.copy().items():
            min_distance = math.inf
            min_track_box = None

            # Find closest bounding box
            for tracking_box_cur in tracking_boxes_cur_frame:
                distance = math.hypot(
                    tracking_box_prev.x - tracking_box_cur.x,
                    tracking_box_prev.y - tracking_box_cur.y,
                )

                # Only take bounding box if it is closest AND somewhat close to bounding box (closer than MAX_TRACKING_...)
                if distance < min_distance and distance < MAX_TRACKING_MATCH_DISTANCE:
                    min_distance = distance
                    min_track_box = tracking_box_cur

            if min_track_box is not None:
                # Update tracking box for object if close box found
                tracking_objects[object_id] = min_track_box
                tracking_boxes_cur_frame.remove(min_track_box)
            else:
                # Remove IDs lost
                tracking_objects.pop(object_id)

        # Add new IDs found
        for tracking_box_cur in tracking_boxes_cur_frame:
            tracking_objects[track_id] = tracking_box_cur
            track_id += 1

        ############################
        # scaling factor estimation
        ############################
        if not is_calibrated:
            if len(tracked_boxes) >= NUM_TRACKED_CARS:
                # more than x cars were tracked
                ground_truth_events = get_ground_truth_events(tracked_boxes)
                print("Number of GT events: ", len(ground_truth_events))
                if len(ground_truth_events) >= NUM_GT_EVENTS:
                    # could extract more than x ground truth events
                    geo_model.scale_factor = 2 * (
                        offline_scaling_factor_estimation_from_least_squares(
                            geo_model, ground_truth_events
                        )
                    )
                    logging.info(
                        f"Is calibrated: scale_factor: {geo_model.scale_factor}"
                    )
                    print(
                        f"Is calibrated: scale_factor: {geo_model.scale_factor}",
                        flush=True,
                    )
                    is_calibrated = True
                    progress_bar.close()
                    torch.cuda.empty_cache()
                    od = ObjectDetection()

            progress_bar.update(len(tracked_boxes) - progress_bar.n)
            for object_id, tracking_box in tracking_objects.items():
                tracked_boxes[object_id].append(tracking_box)
        else:
            ############################
            # track cars
            ############################
            for object_id, tracking_box in tracking_objects.items():
                cv2.putText(
                    frame,
                    f"ID:{object_id}",
                    (
                        tracking_box.x + tracking_box.w + 5,
                        tracking_box.y + tracking_box.h,
                    ),
                    0,
                    1,
                    (255, 255, 255),
                    2,
                )
                if object_id in tracked_cars:
                    tracked_cars[object_id].tracked_boxes.append(tracking_box)
                    tracked_cars[object_id].frames_seen += 1
                    tracked_cars[object_id].frame_end += 1
                else:
                    tracked_cars[object_id] = Car(
                        [tracking_box], 1, frame_count, frame_count
                    )

            ############################
            # speed estimation
            ############################
            if frame_count >= fps and frame_count % (15 * fps) == 0:
                # every x seconds
                car_count_towards = 0
                car_count_away = 0
                total_speed_towards = 0
                total_speed_away = 0
                total_speed_meta_appr_towards = 0
                total_speed_meta_appr_away = 0
                ids_to_drop = []

                for car_id, car in tracked_cars.items():
                    if car.frame_end >= frame_count - sliding_window:
                        if 5 < car.frames_seen < 750:
                            car.direction = calculate_car_direction(car)
                            car_first_box = car.tracked_boxes[0]
                            car_last_box = car.tracked_boxes[-1]
                            meters_moved = geo_model.get_distance_from_camera_points(
                                CameraPoint(
                                    car_first_box.frame_count,
                                    car_first_box.center_x,
                                    car_first_box.center_y,
                                ),
                                CameraPoint(
                                    car_last_box.frame_count,
                                    car_last_box.center_x,
                                    car_last_box.center_y,
                                ),
                            )
                            if meters_moved <= 6:
                                continue

                            if car.direction == Direction.towards:
                                car_count_towards += 1
                                total_speed_towards += (meters_moved) / (
                                    car.frames_seen / fps
                                )
                                total_speed_meta_appr_towards += (
                                    AVG_FRAME_COUNT / int(car.frames_seen)
                                ) * SPEED_LIMIT
                            else:
                                car_count_away += 1
                                total_speed_away += (meters_moved) / (
                                    car.frames_seen / fps
                                )
                                total_speed_meta_appr_away += (
                                    AVG_FRAME_COUNT / int(car.frames_seen)
                                ) * SPEED_LIMIT

                    else:
                        # car is too old, drop from tracked_cars
                        ids_to_drop.append(car_id)

                for car_id in ids_to_drop:
                    del tracked_cars[car_id]

                if car_count_towards > 0:
                    avg_speed = round(
                        (total_speed_towards / car_count_towards) * 3.6, 2
                    )
                    print(f"Average speed towards: {avg_speed} km/h")
                    print(
                        f"Average META speed towards: {(total_speed_meta_appr_towards / car_count_towards)} km/h"
                    )
                    logging.info(
                        json.dumps(dict(frameId=frame_count, avgSpeedTowards=avg_speed))
                    )

                if car_count_away > 0:
                    avg_speed = round((total_speed_away / car_count_away) * 3.6, 2)
                    print(f"Average speed away: {avg_speed} km/h")
                    print(
                        f"Average META speed away: {(total_speed_meta_appr_away / car_count_away)} km/h"
                    )
                    logging.info(
                        json.dumps(dict(frameId=frame_count, avgSpeedAway=avg_speed))
                    )

        ############################
        # output text on video stream
        ############################
        cv2.putText(
            frame,
            f"Timestamp: {(frame_count / fps):.2f} s",
            (7, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            text_color,
            2,
        )
        cv2.putText(
            frame, f"FPS: {fps}", (7, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2
        )
        cv2.imwrite(
            f"speed_estimation/frames_detected/frame_after_detection.jpg", frame
        )

        if frame_count % 500 == 0:
            print(
                f"Frame no. {frame_count} time since start: {(time.time()-start):.2f}s"
            )
        frame_count += 1
        if max_frames != 0 and frame_count >= max_frames:
            if not is_calibrated:
                log_name = None
            break

    input_video.release()
    cv2.destroyAllWindows()
    logging.shutdown()
    return log_name


def main():
    FPS = config.getint("main", "fps")
    CUSTOM_OBJECT_DETECTION = config.getboolean("main", "custom_object_detection")

    max_frames = FPS * 60 * 20  # fps * sec * min

    session_path_local = sys.argv[1] if len(sys.argv) > 1 else session_path
    log_name = run(
        os.path.join(session_path_local, "video.mp4"),
        session_path_local,
        FPS,
        max_frames=max_frames,
        custom_object_detection=CUSTOM_OBJECT_DETECTION,
    )

    if log_name is None:
        print("Calibration did not finish, skip evaluation.")
    else:
        ### Evaluation
        plot_absolute_error([log_name], "logs/")


if __name__ == "__main__":
    main()