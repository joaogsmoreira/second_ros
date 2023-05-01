#!/usr/bin/env python3

import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from scipy.optimize import linear_sum_assignment
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox

class BoundingBoxMatcher:
    def __init__(self, cost_thresh=1.0):
        self.bboxes_prev = []
        self.cost_thresh = cost_thresh

    def match(self, bboxes):
        """
        Performs data association using the Hungarian algorithm.
        :param bboxes: A list of BoundingBox messages from the current frame.
        :return: A list of tuples where each tuple contains the indices of the associated BoundingBox messages from the
        previous and current frames, respectively.
        """
        cost_matrix = np.zeros((len(self.bboxes_prev), len(bboxes)))
        for i, bbox_prev in enumerate(self.bboxes_prev):
            for j, bbox_curr in enumerate(bboxes):
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

def get_marker(bbox, header):
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

def callback(bounding_boxes : BoundingBoxArray):
    bboxes = bounding_boxes.boxes
    matches = bbox_matcher.match(bboxes)
    # Assign the matches to the label field of the current frame's bounding boxes.
    for i, bbox in enumerate(bboxes):
        for match in matches:
            if match[1] == i:
                # Assign the label from the previous frame's matching bounding box.
                bbox.label = bbox_matcher.bboxes_prev[match[0]].label

    matched_bboxes          = BoundingBoxArray()
    matched_bboxes.header   = bounding_boxes.header
    matched_bboxes.boxes    = bboxes

    pub_bbox.publish(matched_bboxes)

    # Store the current frame's bounding boxes for the next iteration.
    bbox_matcher.bboxes_prev = bboxes

    marker_arr = MarkerArray()
    for bbox in matched_bboxes.boxes:
        marker_arr.markers.append(get_marker(bbox, matched_bboxes.header))
        
    pub_marker.publish(marker_arr)

if __name__ == "__main__":
    rospy.init_node('tracker_node')
    bbox_matcher = BoundingBoxMatcher(cost_thresh=1.0)
    
    rospy.Subscriber('/detections', BoundingBoxArray, callback)
    pub_bbox    = rospy.Publisher("/tracked_bbox", BoundingBoxArray, queue_size=100)
    pub_marker  = rospy.Publisher("/tracked_bbox_id", MarkerArray, queue_size=100)
    
    rospy.spin()