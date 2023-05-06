#!/usr/bin/env python3
import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class BBoxTracker:
    def __init__(self, bbox : BoundingBox, 
                 dt=0.1, 
                 x_std_meas=0.1, 
                 y_std_meas=0.1, 
                 z_std_meas=0.1):
        """
        Initializes a BBoxTracker object with the specified parameters.

        Parameters:
        - bbox (object): The bounding box object containing the initial state.
        - dt (float): The time interval between measurements.
        - x_std_meas (float): The standard deviation of the measurement noise in the x-axis.
        - y_std_meas (float): The standard deviation of the measurement noise in the y-axis.
        - z_std_meas (float): The standard deviation of the measurement noise in the z-axis.
        """
        self.kf = KalmanFilter(dim_x=3, dim_z=3)
        
        # define state variables: x, y, z
        self.kf.x = np.array([bbox.pose.position.x,
                              bbox.pose.position.y,
                              bbox.pose.position.z])
        
        # define measurement function
        self.kf.H = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])
        
        # define measurement noise
        self.kf.R = np.diag([x_std_meas**2, y_std_meas**2, z_std_meas**2])
        
        # define process noise
        self.kf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=0.1)
        
        # define state transition function
        self.kf.F = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])
        
        # define control function
        self.kf.B = np.zeros((3, 0))
        
        # define control input noise
        self.kf.R[0:3, 0:3] = np.diag([x_std_meas**2, y_std_meas**2, z_std_meas**2])

    def predict(self):
        """
        Predicts the state of the filter at the next time step.
        """
        self.kf.predict()

    def update(self, bbox : BoundingBox):
        """
        Updates the state of the filter based on the measurements from the given bounding box.

        Parameters:
        - bbox (object): The bounding box containing the measurements.
        """
        z = np.array([bbox.pose.position.x,
                      bbox.pose.position.y,
                      bbox.pose.position.z])
        
        self.kf.update(z)

class BBoxMatcher:
    def __init__(self, cost_thresh=1.0):
        self.cost_thresh = cost_thresh
        self.history_len = 50 #frames

    def __match(self, bboxes_curr, bboxes_prev):
        """
        Performs data association using the Hungarian algorithm.
        :param bboxes: A list of BoundingBox messages from the current frame.
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

    def associate(self, bboxes_curr, bboxes_prev):
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
        self.tracks         = []
        self.matcher        = BBoxMatcher(cost_thresh=1.0)

    def get_marker(self, bbox : BoundingBox, header):
        marker          = Marker()
        marker.type     = Marker.TEXT_VIEW_FACING
        marker.action   = Marker.ADD
        marker.scale.x  = 0.5
        marker.scale.y  = 0.5
        marker.scale.z  = 0.5
        marker.color.a  = 1.0
        marker.color.r  = 1.0
        marker.color.g  = 1.0
        marker.color.b  = 1.0
        marker.header   = header
        marker.id       = bbox.label
        marker.text     = str(bbox.label)
        marker.pose     = bbox.pose
        return marker

    def callback(self, bounding_boxes : BoundingBoxArray):
        bboxes = bounding_boxes.boxes

        if not self.bboxes_prev:
            self.bboxes_prev.append(bboxes)
            #self.tracks = [(bbox.label, BBoxTracker(bbox)) for bbox in bboxes]
            return

        # Data association with Hungarian
        bboxes = self.matcher.associate(bboxes, self.bboxes_prev)

        # Matched bounding boxes
        matched_bboxes          = BoundingBoxArray()
        matched_bboxes.header   = bounding_boxes.header
        matched_bboxes.boxes    = bboxes

        # Clearing previous texts markers
        marker_delete= MarkerArray()
        marker = Marker()
        marker.id = 0
        marker.action = Marker.DELETEALL
        marker_delete.markers.append(marker)
        pub_marker.publish(marker_delete)

        # Adding text to the center of bbox
        marker_arr = MarkerArray()
        for bbox in matched_bboxes.boxes:
            marker_arr.markers.append(self.get_marker(bbox, matched_bboxes.header))

        # Publish    
        pub_bbox.publish(matched_bboxes)
        pub_marker.publish(marker_arr)

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
    rospy.spin()