#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Vector3
from pyquaternion import Quaternion
import sensor_msgs.point_cloud2 as pc2
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
import logging

import sys
import time
sys.path.append("/home/johny/catkin_ws/src/second_ros/second.pytorch")

import numpy as np

import torch
from google.protobuf import text_format
from second.pytorch.train import build_network
from second.protos import pipeline_pb2
from second.utils import config_tool

# This value represents the difference in height from our custom dataset to the Kitti dataset
# since Kitti use a height of 1,73m, and our dataset was taken with a height of ~0.90m
# we added the difference to both labeling and in the bounding box representation code (it might require manual tuning)
_LIDAR_HEIGHT = 0.80

class Logger:
    def __init__(self, log_name : str) -> None:
        self.log = logging.getLogger(log_name)
        self.log.setLevel(logging.INFO)

        # Create a file handler to write the log to a file
        file_handler = logging.FileHandler("/home/johny/catkin_ws/time.log")
        file_handler.setLevel(logging.INFO)

        # Create a formatter to define the log message format
        formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        self.log.addHandler(file_handler)
    
    def log_info(self, timestamp: int, fps: float) -> None:
        self.log.info("%d %f", timestamp, fps)


class Second_ROS:
    def __init__(self):
        config_path, ckpt_path = self.init_ros()
        self.init_second(config_path, ckpt_path)
        self.logger = Logger("FPSLogger")
        self.iteration = 0

    def init_second(self, config_path, ckpt_path):
        """ Initialize second model """
        
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
        input_cfg = config.eval_input_reader
        model_cfg = config.model.second
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = build_network(model_cfg).to(self.device).eval()
        self.net.load_state_dict(torch.load(ckpt_path))
        target_assigner = self.net.target_assigner
        self.voxel_generator = self.net.voxel_generator

        grid_size = self.voxel_generator.grid_size
        feature_map_size = grid_size[:2] // config_tool.get_downsample_factor(model_cfg)
        feature_map_size = [*feature_map_size, 1][::-1]

        self.anchors = target_assigner.generate_anchors(feature_map_size)["anchors"]
        self.anchors = torch.tensor(self.anchors, dtype=torch.float32, device=self.device)
        self.anchors = self.anchors.view(1, -1, 7)
        print("[second_ros] Model Initialized")


    def init_ros(self):
        """ Initialize ros parameters """

        self.sub_velo   = rospy.Subscriber("/point_cloud", PointCloud2, self.lidar_callback, queue_size=1, buff_size=2**24)
        self.pub_bbox   = rospy.Publisher("/detections", BoundingBoxArray, queue_size=1)
        self.pub_cloud  = rospy.Publisher("/synced_cloud", PointCloud2, queue_size=1)
        self.pub_marker = rospy.Publisher("/detections_score", MarkerArray, queue_size=100)
        
        # Trained for all classes on KITTI
        #config_path = rospy.get_param("/config_path", "/home/johny/catkin_ws/src/second_ros/config/all.fhd.config")
        #ckpt_path = rospy.get_param("/ckpt_path", "/home/johny/catkin_ws/src/second_ros/trained_models/voxelnet-99040.tckpt")
        
        # Trained for pedestrians on custom dataset
        config_path = rospy.get_param("/config_path", "/home/johny/catkin_ws/src/second_ros/config/people.fhd.config")
        ckpt_path = rospy.get_param("/ckpt_path", "/home/johny/catkin_ws/src/second_ros/trained_custom_v2/voxelnet-55710.tckpt")
        
        return config_path, ckpt_path

    def inference(self, points):
        num_features = 4
        points = points.reshape([-1, num_features])
        rospy.logdebug("[second_ros] inference points shape: ", points.shape)
        dic = self.voxel_generator.generate(points)
        voxels, coords, num_points = dic['voxels'], dic['coordinates'], dic['num_points_per_voxel']
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)
        voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=self.device)
        
        input_points = {
        "anchors": self.anchors,
        "voxels": voxels,
        "num_points": num_points,
        "coordinates": coords,
        }

        pred = self.net(input_points)[0]
        boxes_lidar = pred["box3d_lidar"].detach().cpu().numpy()
        scores = pred["scores"].detach().cpu().numpy()
        label = pred["label_preds"].detach().cpu().numpy()

        return boxes_lidar, scores, label

    def get_score_marker(self, bbox : BoundingBox):
        marker                  = Marker()
        marker.header.frame_id  = 'rslidar'
        marker.header.stamp     = rospy.Time.now()
        marker.type             = Marker.TEXT_VIEW_FACING
        marker.action           = Marker.ADD
        marker.scale            = Vector3(0.5, 0.5, 0.5)
        marker.color            = ColorRGBA(1.0, 1.0, 1.0, 1.0)
        marker.id               = bbox.label
        marker.text             = "score:" + str(bbox.value.round(decimals=2))
        marker.pose.position.x  = bbox.pose.position.x
        marker.pose.position.y  = bbox.pose.position.y
        marker.pose.position.z  = bbox.pose.position.z + 1.4
        return marker

    def lidar_callback(self, msg):
        """ Captures pointcloud data and feed into second model for inference """

        pcl_msg = pc2.read_points(msg, skip_nans=False, field_names=("x", "y", "z", "intensity", "ring"))
        np_p = np.array(list(pcl_msg), dtype=np.float32)
        
        # convert to xyzi point cloud
        x = np_p[:, 0].reshape(-1)
        y = np_p[:, 1].reshape(-1)
        z = np_p[:, 2].reshape(-1)
        if np_p.shape[1] == 4: # if intensity field exists
            i = np_p[:, 3].reshape(-1)
        else:
            i = np.zeros((np_p.shape[0], 1)).reshape(-1)
        cloud = np.stack((x, y, z, i)).T
        
        # start processing
        tic = time.time()
        boxes_lidar, scores, label = self.inference(cloud)
        toc = time.time()
        fps = 1/(toc-tic)
        self.iteration += 1
        
        # Log the timestamp and FPS value
        #self.logger.log_info(self.iteration, toc-tic)

        num_detections = len(boxes_lidar)
        arr_bbox = BoundingBoxArray()
        arr_marker = MarkerArray()

        # Clearing previous texts markers
        marker_delete = MarkerArray()
        marker = Marker()
        marker.id = 0
        marker.action = Marker.DELETEALL
        marker_delete.markers.append(marker)
        self.pub_marker.publish(marker_delete)

        for i in range(num_detections):
            #if label[i] != 2:               # Checking for pedestrian only
                #continue
            #if  scores[i] < 0.50:          # With confidence level of at least 50%
                #continue
            
            bbox = BoundingBox()
            bbox.header.frame_id = msg.header.frame_id
            bbox.header.stamp    = rospy.Time.now()
            bbox.pose.position.x = float(boxes_lidar[i][0])
            bbox.pose.position.y = float(boxes_lidar[i][1])
            bbox.pose.position.z = float(boxes_lidar[i][2]) + float(boxes_lidar[i][5]) / 2 - _LIDAR_HEIGHT
            bbox.dimensions.x    = float(boxes_lidar[i][3])  # width
            bbox.dimensions.y    = float(boxes_lidar[i][4])  # length
            bbox.dimensions.z    = float(boxes_lidar[i][5])  # height

            q = Quaternion(axis=(0, 0, 1), radians=float(boxes_lidar[i][6]))
            bbox.pose.orientation.x = q.x
            bbox.pose.orientation.y = q.y
            bbox.pose.orientation.z = q.z
            bbox.pose.orientation.w = q.w
            bbox.value = scores[i]
            # Since we're assuming every bounding box is related to the class Pedestrian, we're overwriting
            # the jsk bounding box variable "label" to represent the unique id of the bounding box
            bbox.label = i+1
            arr_bbox.boxes.append(bbox)
            arr_marker.markers.append(self.get_score_marker(bbox))
            
            #rospy.loginfo("Label: %d\tScore: %f", label[i], scores[i])
        
        arr_bbox.header.frame_id = msg.header.frame_id
        arr_bbox.header.stamp = rospy.Time.now()
        
        # This way we garantee that the bounding box we're seeing is relative to the current point cloud we're seeing
        self.pub_cloud.publish(msg)
        self.pub_bbox.publish(arr_bbox)
        self.pub_marker.publish(arr_marker)


if __name__ == '__main__':
    sec = Second_ROS()
    rospy.init_node('second_ros_node')
    # Configure the logging settings
    logging.basicConfig(filename='fps_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        del sec
        rospy.loginfo("[second_ros] Shutting down")
