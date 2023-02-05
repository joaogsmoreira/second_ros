# SECOND: Sparsely Embedded Convolutional Detection with ROS

This package contains the second model that performs 3D object detection from pointcloud data.

![Video_Result2](docs/results.gif)

---
## Setting up the environment

### General Environment

- Ubuntu 20.04
- ROS Noetic

### Compute Environment
- NVIDIA GeForce 3060Ti
- Driver Version 515.86.01
- CUDA 11.1
- CUDDN 8.0.5

### SECOND Environment
Note: It is higly advised to use anaconda/miniconda as an environment manager in this project.

Create the environment from the environment.yml file:
```
conda env create -f environment.yml
```
Although all dependencies should be covered in the environment.yaml file, necessary model dependencies are listed here [second.pytorch model.](https://github.com/traveller59/second.pytorch)

Please note that this repo requires Python 3.6+. If you are using ROS Melodic, please build ROS from source to ensure rospy runs with Python3.

### SpConv: PyTorch Spatially Sparse Convolution Library
For the complete version of the installation, please refer to [spconv.](https://github.com/traveller59/second.pytorch)

Note: this repo was implemented using version 1.2.1.
``` 
$ sudo apt-get install libboost-all-dev
$ cd spconv
$ python setup.py bdist_wheel
$ pip install dist/spconv-1.2.1-*
```


## Setting up the package

### Clone project into catkin_ws and build it

``` 
$ cd ~/catkin_ws && catkin_make
$ source devel/setup.bash
```

### Download rosbag files for testing

To download rosbags for testing, please follow the [link](https://github.com/tomas789/kitti2bag) to get a kitti.bag for testing.

## Using the package

### Running the package

```
$ roslaunch second_ros second_kitti.launch
```

## Licenses and References
I referenced original implementation of [second.pytorch](https://github.com/traveller59/second.pytorch) which is under MIT License
