#include "ros/ros.h"
#include <sensor_msgs/PointCloud2.h>
#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <cmath>

#include <ros/console.h>

#define DISPLACEMENT_THRESHOLD      3.0
#define IOU_THRESHOLD               1.0

class Tracking {
public:
    // Constructor
    Tracking() {}

    bool compareBoxes(const jsk_recognition_msgs::BoundingBox& a, const jsk_recognition_msgs::BoundingBox& b);

    std::vector<std::vector<uint32_t>> associateBoxes(const jsk_recognition_msgs::BoundingBoxArray& curr_boxes);

    std::vector<std::vector<uint32_t>> connectionMatrix(const std::vector<std::vector<uint32_t>>& connection_pairs, std::vector<uint32_t>& left, std::vector<uint32_t>& right);

    bool hungarianFind(const int i, const std::vector<std::vector<uint32_t>>& connection_matrix, std::vector<bool>& right_connected, std::vector<int>& right_pair);

    std::vector<int> hungarian(const std::vector<std::vector<uint32_t>>& connection_matrix);

    int searchBoxIndex(const jsk_recognition_msgs::BoundingBoxArray& boxes, const int id);
    
    void callback(const jsk_recognition_msgs::BoundingBoxArray& msg);

    jsk_recognition_msgs::BoundingBoxArray curr_boxes;
    jsk_recognition_msgs::BoundingBoxArray prev_boxes;
    // Publisher
    ros::Publisher publisher;
};


bool Tracking::compareBoxes(const jsk_recognition_msgs::BoundingBox& a, const jsk_recognition_msgs::BoundingBox& b) {
    // Percentage Displacements ranging between [0.0, +oo]
    const float dis = sqrt( 
        pow((a.pose.position.x - b.pose.position.x), 2) + 
        pow((a.pose.position.y - b.pose.position.y), 2) + 
        pow((a.pose.position.z - b.pose.position.z), 2)
    );

    const float a_max_dim = std::max(a.pose.position.x, std::max(a.pose.position.y, a.pose.position.z));
    const float b_max_dim = std::max(b.pose.position.x, std::max(b.pose.position.y, b.pose.position.z));
    const float ctr_dis = dis / std::min(a_max_dim, b_max_dim);

    // Dimension similiarity values between [0.0, 1.0]
    const float x_dim = 2 * (a.pose.position.x - b.pose.position.x) / (a.pose.position.x + b.pose.position.x);
    const float y_dim = 2 * (a.pose.position.y - b.pose.position.y) / (a.pose.position.y + b.pose.position.y);
    const float z_dim = 2 * (a.pose.position.z - b.pose.position.z) / (a.pose.position.z + b.pose.position.z);

    //ROS_DEBUG_STREAM("ctr_dis: " << ctr_dis << "x_dim: " << x_dim << "y_dim: " << y_dim << "z_dim: " << z_dim);

    if (ctr_dis <= DISPLACEMENT_THRESHOLD && x_dim <= IOU_THRESHOLD && y_dim <= IOU_THRESHOLD && z_dim <= IOU_THRESHOLD) {
        return true;
    } else {
        return false;
    }
}


std::vector<std::vector<uint32_t>> Tracking::associateBoxes(const jsk_recognition_msgs::BoundingBoxArray& curr_boxes) {
    std::vector<std::vector<uint32_t>> connection_pairs;

    for (auto& prev_box : this->prev_boxes.boxes) {
        for (auto& curBox : curr_boxes.boxes) {
            // Add the indecies of a pair of similiar boxes to the matrix
            if (this->compareBoxes(curBox, prev_box)) {
                connection_pairs.push_back({prev_box.label, curBox.label});
            }
        }
    }

    return connection_pairs;
}

std::vector<std::vector<uint32_t>> Tracking::connectionMatrix(const std::vector<std::vector<uint32_t>>& connection_pairs, std::vector<uint32_t>& left, std::vector<uint32_t>& right) {
  // Hash the box ids in the connection_pairs to two vectors(sets), left and right
    for (auto& pair : connection_pairs) {
        bool left_found = false;
        for (auto i : left) {
            if (i == pair[0])
            left_found = true;
        }
        if (!left_found)
            left.push_back(pair[0]);

        bool right_found = false;
        for (auto j : right) {
            if (j == pair[1])
            right_found = true;
        }
        if (!right_found)
            right.push_back(pair[1]);
    }

    std::vector<std::vector<uint32_t>> connection_matrix(left.size(), std::vector<uint32_t>(right.size(), 0));

    for (auto& pair : connection_pairs) {
        int left_index = -1;
        for (int i = 0; i < left.size(); ++i) {
            if (pair[0] == left[i])
            left_index = i;
        }

        int right_index = -1;
        for (int i = 0; i < right.size(); ++i) {
            if (pair[1] == right[i]) {
                right_index = i;
            }
        }

        if (left_index != -1 && right_index != -1) {
            connection_matrix[left_index][right_index] = 1;
        }
    }

    return connection_matrix;
}

bool Tracking::hungarianFind(const int i, const std::vector<std::vector<uint32_t>>& connection_matrix, std::vector<bool>& right_connected, std::vector<int>& right_pair) {
    for (int j = 0; j < connection_matrix[0].size(); ++j) {
        if (connection_matrix[i][j] == 1 && right_connected[j] == false) {
            right_connected[j] = true;
            if (right_pair[j] == -1 || hungarianFind(right_pair[j], connection_matrix, right_connected, right_pair)) {
                right_pair[j] = i;
                return true;
            }
        }
    }
    return false;
}

std::vector<int> Tracking::hungarian(const std::vector<std::vector<uint32_t>>& connection_matrix) {
    std::vector<bool> right_connected(connection_matrix[0].size(), false);
    std::vector<int> right_pair(connection_matrix[0].size(), -1);

    uint32_t count = 0;
    for (int i = 0; i < connection_matrix.size(); ++i) {
        if (hungarianFind(i, connection_matrix, right_connected, right_pair)) {
            count++;
        }
    }

    std::cout << "For: " << right_pair.size() << " current frame bounding boxes, found: " << count << " matches in previous frame! " << std::endl;

    return right_pair;
}

int Tracking::searchBoxIndex(const jsk_recognition_msgs::BoundingBoxArray& boxes, const int id) {
    for (int i = 0; i < boxes.boxes.size(); i++) {
        if (boxes.boxes[i].label == id) {
            return i;
        }
    }
    return -1;
}

void Tracking::callback(const jsk_recognition_msgs::BoundingBoxArray& latest_boxes) {
    ROS_DEBUG_STREAM("BoundingBoxArray received");
    curr_boxes = latest_boxes;
    //  Since we're assuming every bounding box is related to the class Pedestrian, we're overwriting
    //  the jsk bounding box variable "label" to represent the unique id of the bounding box
    uint32_t fake_id = 0;
    for (auto& box : curr_boxes.boxes) {
        box.label = fake_id++;
    }

    if (prev_boxes.boxes.empty()) {
        prev_boxes = curr_boxes;
    }

    //  Associate Boxes that are similar in two frames
    auto connection_pairs = associateBoxes(curr_boxes);
    
    if (connection_pairs.empty()) {
        ROS_ERROR_STREAM("Connection pairs empty");
        return;
    }

    //  Vectors containing the id of boxes in left and right sets
    std::vector<uint32_t> prev_ids;
    std::vector<uint32_t> curr_ids;
    std::vector<int> matches;

    //  Construct the connection matrix for Hungarian Algorithm
    auto connection_matrix = connectionMatrix(connection_pairs, prev_ids, curr_ids);

    //  Use Hungarian Algorithm to solve for max-matching
    matches = hungarian(connection_matrix);

    for (int j = 0; j < matches.size(); ++j) {
        //  Find the index of the previous box that the current box corresponds to
        const uint32_t prev_id = prev_ids[matches[j]];
        const uint32_t prev_index = searchBoxIndex(prev_boxes, prev_id);
        //  Find the index of the current box that needs to be changed
        const uint32_t curr_id = curr_ids[j]; // right and matches has the same size
        const auto curr_index = searchBoxIndex(curr_boxes, curr_id);

        if (prev_index > -1 && curr_index > -1) {
        //  Change the id of the current box to the same as the previous box
            curr_boxes.boxes[curr_index].label = prev_boxes.boxes[prev_index].label;
        }
    }
    
    prev_boxes = curr_boxes;
    this->publisher.publish(curr_boxes);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "tracking_node");
    ros::NodeHandle a;
    Tracking tracker;

    ros::Subscriber pcl = a.subscribe("/detections", 10000, &Tracking::callback, &tracker);
    tracker.publisher = a.advertise<jsk_recognition_msgs::BoundingBoxArray>("/tracking", 10000);


    while (ros::ok()) {
        ros::spinOnce();
    }
}