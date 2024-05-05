from typing import List, Tuple
from ultralytics import YOLO
import supervision as sv
from rtree import index
import numpy as np
import torch
import cv2
import numpy as np
from paths import YOLOV4_WEIGHTS, YOLOV4_CLASSES, YOLOV4_CONFIG
YOLOV8_WEIGHTS = "model_weights/yolov8x.pt"
class YOLOv8Wrapper:
    def __init__(self, model_path, mask_points_file=None, fps=10, config=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)
        self.model = YOLO(model_path, task='detect')
        self.model.to(self.device)
        self.model.fuse()
        #self.mask_points = self.load_mask_points(mask_points_file) if mask_points_file else None
        self.mask_points = None
        self.byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=fps)
        self.config = config
        if config is None:
            print("NO CONFIG FOUND")
        self.vehicle_classes = [2, 3, 5, 7] if config is None else config["DETECT_CLASSES"]# bus, car, motorcycle, truck
        self.pedestrian_classes = [0, 1] if config is None else config["PEDESTRIAN_CLASSES"] # person, bicycle
        self.OVERLAP_EXCLUDE = 0.95
        CLASS_NAMES_DICT = self.get_class_names()
        try:
            CLASS_NAMES_DICT[len(CLASS_NAMES_DICT)] = "vehicle" # add class_name for class_id 80
        except IndexError:
            CLASS_NAMES_DICT.append("vehicle") # add class_name for class_id 80
        self.class_names_dict = CLASS_NAMES_DICT
        self.original_class_id = None
        self.final_mask = None

        self.times = []

    def get_class_names(self):
        return self.model.model.names

    def load_mask_points(self, file_path):
        points = []
        with open(file_path, 'r') as file:
            for line in file:
                x, y = line.strip().split(',')
                points.append((int(x), int(y)))
        return np.array(points, np.int32).reshape((-1, 1, 2))

    def _calculate_overlap_area(self, box1, box2):
        dx = min(box1[2], box2[2]) - max(box1[0], box2[0])
        dy = min(box1[3], box2[3]) - max(box1[1], box2[1])
        if (dx>=0) and (dy>=0):
            return dx*dy
        return 0

    def apply_mask_to_image(self, image):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self.mask_points], 255)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image

    def predict(self, image, conf_threshold=0.2, iou=0.7, verbose=False):
        if self.mask_points:
            image = self.apply_mask_to_image(image)
        # Perform inference on the masked image
        # It seems that the model itself is RGB, but opencv uses BGR internally
        # Unfortunately this means we end up with multiple copy ops on the box (bayer2BGR -> BGR2RGB)
        return self.model(image, conf=conf_threshold, iou=iou, verbose=verbose)

    def rtree_filter(self, detections):
        self.original_class_id = detections.class_id.copy()
        class_filter = self.vehicle_classes + self.pedestrian_classes
        mask = np.array([class_id in class_filter for class_id in detections.class_id], dtype=bool)
        ped_mask = np.array([class_id in self.pedestrian_classes for class_id in detections.class_id], dtype=bool)
        idx = index.Index()
        bbox_mask = []

        for n in range(len(detections.xyxy)):
            bboxes = detections.xyxy[n]

            # Check if the current detection is in pedestrian classes, skip R-tree if true
            if ped_mask[n]:
                bbox_mask.append(True)  # Always keep this bounding box
                continue

            overlaps = list(idx.intersection(bboxes))
            overlap_exists = False
            for overlap_bbox_index in overlaps:
                overlap_bbox = detections.xyxy[overlap_bbox_index]
                overlap_area = self._calculate_overlap_area(bboxes, overlap_bbox)
                area_bbox = (bboxes[2]-bboxes[0])*(bboxes[3]-bboxes[1])
                area_overlap_bbox = (overlap_bbox[2]-overlap_bbox[0])*(overlap_bbox[3]-overlap_bbox[1])
                if overlap_area >= self.OVERLAP_EXCLUDE * area_bbox or overlap_area >= self.OVERLAP_EXCLUDE * area_overlap_bbox:
                    overlap_exists = True
                    break

            if not overlap_exists:
                idx.insert(n, bboxes)
                bbox_mask.append(True)  # keep this bounding box
            else:
                bbox_mask.append(False)  # remove this bounding box

        bbox_mask = np.array(bbox_mask)
        # Combine class filter mask and bounding box filter mask
        self.final_mask = np.logical_and(mask, bbox_mask)

        detections.class_id[self.final_mask] = len(self.class_names_dict) - 1
        detections = detections[self.final_mask]

        return detections

    def update_original_class_IDs(self, detections):
        detections.class_id = self.original_class_id[self.final_mask]
        return detections

    def __call__(self, frame, verbose=False, conf=0.2):
        '''Returns the tracked detections'''

        # t0 = time.time()
        results = self.predict(frame, conf_threshold=conf, verbose=verbose)[0]


        detections = sv.Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, self.vehicle_classes + self.pedestrian_classes)]

        # detections = self.rtree_filter(detections)
        detections = self.byte_tracker.update_with_detections(detections)
        # detections = self.update_original_class_IDs(detections)

        # endtime = time.time()
        # self.times.append(endtime-t0)
        # if len(self.times) > 100:
        #     average_time = sum(self.times) / len(self.times)
        #     print(f"Average model inference time per frame: {average_time:.4f} seconds")
        #     self.times.clear()
        # RTree filter for type overlap overlap

        return detections
def convert_to_xywh(bounding_boxes):
    """
    Convert bounding boxes from (x1, y1, x2, y2) format to (x, y, width, height) format.
    
    Parameters:
        bounding_boxes (numpy.ndarray): 2D numpy array of bounding boxes, where each row is (x1, y1, x2, y2).
    
    Returns:
        numpy.ndarray: 2D numpy array of bounding boxes, where each row is (x, y, width, height).
    """
    # Ensure the input is a numpy array
    bounding_boxes = np.asarray(bounding_boxes)
    
    # Compute the top-left corner coordinates (x, y)
    x = bounding_boxes[:, 0]
    y = bounding_boxes[:, 1]
    
    # Compute width and height
    width = bounding_boxes[:, 2] - bounding_boxes[:, 0]
    height = bounding_boxes[:, 3] - bounding_boxes[:, 1]
    
    # Stack the results to get the output in the desired format
    xywh_boxes = np.stack([x, y, width, height], axis=1)
    
    return xywh_boxes
class ObjectDetection:
    """This class is used to detect the cars in a frame."""

    def __init__(self, weights_path=YOLOV8_WEIGHTS):
        """Create an instance of ObjectDetection.

        @param weights_path:
            The path to the model weights.
        @param cfg_path:
            The path to config file.
        """
        print("Loading Object Detection")
        print("Running YOLOV8")
        
        self.colors = np.random.uniform(0, 255, size=(80, 3))
        self.v8 = YOLOv8Wrapper(weights_path, fps=25)
        self.load_class_names()
        


    def load_class_names(self, classes_path: str = YOLOV4_CLASSES) -> List[str]:
        """Get all classes the model can classify in an image.

        @param classes_path: The path to the classes.txt file
            (e.g., `speed_estimation/model_weights/classes.txt`).

        @return: Returns a list of all classes the model can detect.
        """

        return self.v8.get_class_names()

    def detect(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Detect cars in frame and put bounding boxes around.

        This function detects all cars in the frame and puts 2D bounding boxes around.
        A YoloV4 model is used.

        @param frame:
            The frame that should be analyzed.

        @return:
            Returns a tuple of class_ids, scores, and boxes.
            The class_id identifies which object was detected, while the scores indicate the
            confidence level of the prediction.
            The boxes are represented as a list of NumPy ndarrays, where each array corresponds to
            a bounding box that identifies a car in the frame.
            Each ndarray in the list holds the following information:
            (x_coordinate, y_coordinate, width, height).
        """
        detections = self.v8(
            frame
        )
        xywh_boxes = convert_to_xywh(detections.xyxy)
        detections_tuple = (detections.class_id, detections.confidence, xywh_boxes)
        return detections_tuple

class ObjectDetectionOld:
    """This class is used to detect the cars in a frame."""

    def __init__(self, weights_path=YOLOV4_WEIGHTS, cfg_path=YOLOV4_CONFIG):
        """Create an instance of ObjectDetection.

        @param weights_path:
            The path to the model weights.
        @param cfg_path:
            The path to config file.
        """
        print("Loading Object Detection")
        print("Running opencv dnn with YOLOv4")
        self.nmsThreshold = 0.4
        self.confThreshold = 0.5
        self.image_size = 608

        # Load Network
        net = cv2.dnn.readNet(weights_path, cfg_path)

        # Enable GPU CUDA
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.model = cv2.dnn_DetectionModel(net)

        self.classes = []
        self.load_class_names()
        self.colors = np.random.uniform(0, 255, size=(80, 3))

        self.model.setInputParams(
            size=(self.image_size, self.image_size), scale=1 / 255
        )

    def load_class_names(self, classes_path: str = YOLOV4_CLASSES) -> List[str]:
        """Get all classes the model can classify in an image.

        @param classes_path: The path to the classes.txt file
            (e.g., `speed_estimation/model_weights/classes.txt`).

        @return: Returns a list of all classes the model can detect.
        """
        with open(classes_path, "r", encoding="UTF-8") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        self.colors = np.random.uniform(0, 255, size=(80, 3))

        return self.classes

    def detect(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Detect cars in frame and put bounding boxes around.

        This function detects all cars in the frame and puts 2D bounding boxes around.
        A YoloV4 model is used.

        @param frame:
            The frame that should be analyzed.

        @return:
            Returns a tuple of class_ids, scores, and boxes.
            The class_id identifies which object was detected, while the scores indicate the
            confidence level of the prediction.
            The boxes are represented as a list of NumPy ndarrays, where each array corresponds to
            a bounding box that identifies a car in the frame.
            Each ndarray in the list holds the following information:
            (x_coordinate, y_coordinate, width, height).
        """
        detections = self.model.detect(
            frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold
        )
        return detections
