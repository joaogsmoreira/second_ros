#!/usr/bin/env python3
import rospy
import numpy as np
from typing import List
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Pose, Point, Vector3
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# Tracking hyper-parameters
_DETECTOR_STD_MEAS      = 0.1 # meters
_DETECTOR_FREQUENCY     = 0.1 # seconds
_ASSOCIATION_MAX_COST   = 1.0 # meters
_ASSOCIATION_MAX_AGE    = 20  # frames
_TRACKING_MAX_AGE       = 10  # frames

class BBoxTracker:
    def __init__(self, bbox: BoundingBox, dt = _DETECTOR_FREQUENCY, std_meas = _DETECTOR_STD_MEAS):
        """
        Initializes a BBoxTracker object with the specified parameters.

        Parameters:
        - bbox (object): The bounding box object containing the initial state.
        - dt (float): The time interval between measurements.
        - x_std_meas (float): The standard deviation of the measurement noise in the x-axis.
        - y_std_meas (float): The standard deviation of the measurement noise in the y-axis.
        - z_std_meas (float): The standard deviation of the measurement noise in the z-axis.
        """
        self.id = bbox.label
        self.bbox_dimensions = bbox.dimensions
        self.dt = dt
        self.age = 0
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.initialize(bbox, dt, std_meas)
        
    def initialize(self, bbox : BoundingBox, dt, std_meas):
        # Define state variables: x, y, z, vx, vy, vz
        self.kf.x = np.array([bbox.pose.position.x, bbox.pose.position.y, bbox.pose.position.z, 0.1, 0.1, 0.1])
        
        # Define measurement function
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0]])
        
        # Define measurement noise
        self.kf.R = np.diag([std_meas**2, std_meas**2, std_meas**2])
        
        # Define process noise
        self.kf.Q = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        
        # Define state transition function
        self.kf.F = np.array([[1, 0, 0, dt, 0, 0],
                              [0, 1, 0, 0, dt, 0],
                              [0, 0, 1, 0, 0, dt],
                              [0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1]])
        
    def predict(self):
        """
        Predicts the state of the filter at the next time step.
        """
        self.kf.predict()
    
    def update(self, bbox: BoundingBox):
        """
        Updates the state of the filter based on the position measurements from the given bounding box.

        Parameters:
        - bbox (object): The bounding box containing the position measurements.
        """
        self.kf.update(np.array([bbox.pose.position.x, bbox.pose.position.y, bbox.pose.position.z]))
    
    def get_state(self):
        """
        Returns the current estimated state of the Kalman filter as a ROS marker.
        """
        # Extract position and velocity from Kalman filter state
        pos = Point(self.kf.x[0], self.kf.x[1], self.kf.x[2])
        vel = Vector3(self.kf.x[3], self.kf.x[4], self.kf.x[5])

        # Create a tracking box
        box                 = BoundingBox()
        box.header.frame_id = 'rslidar'
        box.header.stamp    = rospy.Time.now()
        box.label           = self.id
        box.dimensions      = self.bbox_dimensions
        box.pose            = Pose(position=Point(self.kf.x[0], self.kf.x[1], self.kf.x[2]))
        return box


class BBoxMatcher:
    def __init__(self, cost_thresh = _ASSOCIATION_MAX_COST):
        self.cost_thresh = cost_thresh
        self.history_len = _ASSOCIATION_MAX_AGE

    def __match(self, bboxes_curr, bboxes_prev):
        """
        Performs data association using the Hungarian algorithm.
        :param bboxes_curr: A list of BoundingBox messages from the current frame.
        :param bboxes_prev: A list of BoundingBox messages from the previous frame.
        :return: A list of tuples where each tuple contains the indices of the associated BoundingBox messages from the
        previous and current frames, respectively.
        """
        cost_matrix = np.zeros((len(bboxes_prev), len(bboxes_curr)))
        for i, bbox_prev in enumerate(bboxes_prev):
            for j, bbox_curr in enumerate(bboxes_curr):
                # Calculate the Euclidean distance between the centers of the previous and current bounding boxes.
                center_prev         = np.array([bbox_prev.pose.position.x, bbox_prev.pose.position.y, bbox_prev.pose.position.z])
                center_curr         = np.array([bbox_curr.pose.position.x, bbox_curr.pose.position.y, bbox_curr.pose.position.z])
                cost_matrix[i, j]   = np.linalg.norm(center_prev - center_curr)
                # Add the difference in size between the previous and current bounding boxes to the cost.
                size_prev           = np.array([bbox_prev.dimensions.x, bbox_prev.dimensions.y, bbox_prev.dimensions.z])
                size_curr           = np.array([bbox_curr.dimensions.x, bbox_curr.dimensions.y, bbox_curr.dimensions.z])
                size_diff           = np.linalg.norm(size_prev - size_curr)
                cost_matrix[i, j]   += size_diff
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] > self.cost_thresh:
                rospy.loginfo("Bounding box matched above threshold!")
                continue
            matches.append((i, j))
        return matches

    def associate(self, bboxes_curr : List[BoundingBox], bboxes_prev : List[BoundingBox]):
        last_frame_bboxes = bboxes_prev[-1]
        matches = self.__match(bboxes_curr, last_frame_bboxes)
        # Assign the matches to the label field of the current frame's bounding boxes.
        _new_boxes = 0
        for i, bbox in enumerate(bboxes_curr):
            bbox.label = 0
            for match in matches:
                if match[1] == i:
                    # Assign the label from the previous frame's matching bounding box.
                    bbox.label = last_frame_bboxes[match[0]].label
            # If a box was not matched, check if it has appeared in the recent frames
            if bbox.label == 0:
                _appeared = False
                for j in range(1, min(self.history_len, len(bboxes_prev))):
                    prev_bboxes = bboxes_prev[-j]
                    for prev_bbox in prev_bboxes:
                        if np.linalg.norm(np.array([bbox.pose.position.x, bbox.pose.position.y, bbox.pose.position.z]) -
                                          np.array([prev_bbox.pose.position.x, prev_bbox.pose.position.y, prev_bbox.pose.position.z])) < self.cost_thresh:
                            # Assign the label from the previous frame's matching bounding box.
                            bbox.label = prev_bbox.label
                            _appeared = True
                            break
                    if _appeared:
                        break
            # This is a new bounding box
            if bbox.label == 0:
                prev_labels = [prev_bbox.label for prev_bboxes in bboxes_prev for prev_bbox in prev_bboxes]
                bbox.label = max(prev_labels, default=1) + 1 + _new_boxes
                _new_boxes += 1

        return bboxes_curr


class MultipleBBoxTracker:
    def __init__(self) -> None:
        self.bboxes_prev    = []
        self.tracks         = {}
        self.estimated_velo = {}
        self.matcher        = BBoxMatcher()
        self.update_trigger = 0

    def get_marker(self, bbox : BoundingBox):
        marker                  = Marker()
        marker.header.frame_id  = 'rslidar'
        marker.header.stamp     = rospy.Time.now()
        marker.lifetime         = rospy.Duration(secs = 0.1)
        marker.type             = Marker.TEXT_VIEW_FACING
        marker.action           = Marker.ADD
        marker.scale            = Vector3(0.5, 0.5, 0.5)
        marker.color            = ColorRGBA(1.0, 1.0, 1.0, 1.0)
        marker.id               = bbox.label
        marker.text             = str(bbox.label)
        marker.pose             = bbox.pose
        return marker
    
    def get_velocities(self, curr_boxes : List[BoundingBox]):
        for curbox in curr_boxes:
            for prevbox in self.bboxes_prev[-1]:
                if curbox.label == prevbox.label:
                    dT      = curbox.header.stamp.to_sec() - prevbox.header.stamp.to_sec()
                    poseX   = curbox.pose.position.x - prevbox.pose.position.x
                    poseY   = curbox.pose.position.y - prevbox.pose.position.y
                    poseZ   = curbox.pose.position.z - prevbox.pose.position.z
                    velX    = poseX / dT
                    velY    = poseY / dT
                    velZ    = poseZ / dT
                    self.estimated_velo[curbox.label] = [velX, velY, velZ]
                    if curbox.label == 1:
                        print("BBox Velocity: ", velX, velY, velZ)

    def __from_obj_to_list(self, bounding_boxes : BoundingBoxArray) -> List[BoundingBox]:
        return bounding_boxes.boxes
    
    def track(self, curr_boxes : List[BoundingBox]):
        # Create a copy of the tracks dictionary to avoid modifying it during iteration
        tracks_copy = self.tracks.copy()
        
        # Keep track of labels that appear in the current frame
        curr_ids = set(box.label for box in curr_boxes)
        
        # Create box array message
        box_msg = BoundingBoxArray()
        box_msg.header.frame_id = 'rslidar'
        box_msg.header.stamp    = rospy.Time.now()

        # Adding box id to the center of bbox
        marker_arr = MarkerArray()

        # Cross-check track id's with current label id's
        for id, tracker in tracks_copy.items():
            if id in curr_ids:
                tracker.predict()
                tracker.update([box for box in curr_boxes if box.label == id][0])
                tracker.age = 0
                box_msg.boxes.append(tracker.get_state())
                marker_arr.markers.append(self.get_marker(tracker.get_state()))
            else:
                # Remove track if label is not in current frame and track has reached max_age
                if tracker.age > _TRACKING_MAX_AGE:
                    print("Deleting track: ", id)
                    del self.tracks[id]
                else:
                    tracker.age +=1
                    tracker.predict()
                    box_msg.boxes.append(tracker.get_state())
                    marker_arr.markers.append(self.get_marker(tracker.get_state()))
        # Check if its a new track
        for box in curr_boxes:
            if box.label not in self.tracks:
                print("Creating track: ", box.label)
                self.tracks[box.label] = BBoxTracker(box)
        
        print(f"Currently tracking {len(self.tracks)} boxes.")

        # Clearing previous texts markers
        marker_delete= MarkerArray()
        marker = Marker()
        marker.id = 0
        marker.action = Marker.DELETEALL
        marker_delete.markers.append(marker)
        pub_marker.publish(marker_delete)

        # Publish tracks & ids
        pub_predict.publish(box_msg)
        pub_marker.publish(marker_arr)


    def callback(self, bounding_boxes : BoundingBoxArray):
        bboxes = self.__from_obj_to_list(bounding_boxes)

        # Initialize the system with the first bounding boxes
        if not self.bboxes_prev:
            self.bboxes_prev.append(bboxes)
            self.tracks = {bbox.label: BBoxTracker(bbox) for bbox in bboxes}
            return
        
        # Data association with Hungarian
        bboxes = self.matcher.associate(bboxes, self.bboxes_prev)

        # Matched bounding boxes
        matched_bboxes          = BoundingBoxArray()
        matched_bboxes.header   = bounding_boxes.header
        matched_bboxes.boxes    = bboxes

        # Get estimated velocities to feed into the kalman filter
        # self.get_velocities(matched_bboxes.boxes)

        # Updating kalman filter state and markers
        self.track(matched_bboxes.boxes)

        # Publish    
        pub_bbox.publish(matched_bboxes)

        # Store the current frame's bounding boxes for the next iteration.
        self.bboxes_prev.append(bboxes)
        if len(self.bboxes_prev) > self.matcher.history_len:
            self.bboxes_prev.pop(0)

if __name__ == "__main__":
    rospy.init_node('tracker_node')
    mbboxtracker = MultipleBBoxTracker()
    rospy.Subscriber('/detections', BoundingBoxArray, mbboxtracker.callback)
    pub_bbox    = rospy.Publisher("/tracked_bbox", BoundingBoxArray, queue_size=100)
    pub_marker  = rospy.Publisher("/tracked_bbox_id", MarkerArray, queue_size=100)
    pub_predict = rospy.Publisher("/predicted_bbox", BoundingBoxArray, queue_size=100)
    rospy.spin()