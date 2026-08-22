# VINS-Fisheye

## 20260818 数据：Kalibr EUCM → VINS-Fisheye-loop

本数据使用 Kalibr `cam1 + cam2`（约 6.56 cm 基线）、压缩图像
`/cam1/image/compressed`、`/cam2/image/compressed` 和 IMU
`/imu/data_raw`。配置可重复生成：

```bash
cd /home/lhk/workspace/VINS-Fisheye
./scripts/prepare_vins_fisheye_loop.sh
```

原始 bag 的 LZ4 frame 能被标准 LZ4 解码，但 ROS Noetic 的 `roslz4`
会报 `ROSLZ4_DATA_ERROR`。首次运行前生成一个不覆盖原文件的修复版：

```bash
python3 scripts/repair_rosbag_lz4.py \
  /home/lhk/data/20260818-160433.bag \
  /home/lhk/data/20260818-160433.repaired.bag
```

容器内一键编译（可选）、启动 VINS + loop_fusion 并播放 bag：

```bash
docker exec -it vins-fisheye-gpu bash -lc '
  cd /root/catkin_ws/src/VINS-Fisheye &&
  REBUILD=1 VIZ=true TRACKING_VIZ=true \
  ./scripts/run_vins_fisheye_loop_container.sh \
  /data/20260818-160433.repaired.bag
'
```

无界面运行时将 `VIZ=false TRACKING_VIZ=false`。输出为
`data/vio.csv`（VIO）和 `data/vio_loop.csv`（回环位姿图轨迹）。

This repository is a Fisheye version of [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) with GPU and Visionworks acceleration. It can run on Nvidia TX2 in real-time, also provide depth estimation based on fisheye. This project stands as a part of __[Omni-swarm](https://arxiv.org/abs/2103.04131): A Decentralized Omnidirectional Visual-Inertial-UWB State Estimation System for Aerial Swarm__. You may use it alone on any type of robot or as a part of Omni-swarm for swarm robots.

Only stereo visual-inertial-odometry is supported for fisheye cameras now. Loop closure module for fisheye camera will release later.

![Image of PCL](support_files/point_cloud.png)
*Drone path and RGB point cloud estimation*

![Image of fisheye](support_files/feature_track.png)
*Feature tracker for fisheye*

![Image of Disparity](support_files/disparity.png)
*Disparity estimation for depth estimation*

## 运行(Docker)

docker compose --profile gpu build

```
CPU 版(一行)

xhost +local:root && docker run -it --network=host --privileged -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v /home/lhk/workspace/VINS-Fisheye:/root/catkin_ws/src/VINS-Fisheye -v /home/lhk/data:/data --name vins-fisheye-cpu -w /root/catkin_ws vins-fisheye:cpu

GPU 版(一行)

xhost +local:root && docker run -it --gpus all --network=host --privileged -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v /home/lhk/workspace/VINS-Fisheye:/root/catkin_ws/src/VINS-Fisheye -v /home/lhk/data:/data --name vins-fisheye-gpu -w /root/catkin_ws vins-fisheye:gpu
```

## 编译

catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -j8

常用命令

 标准化用法

  每次 my_kalibr_result/ 里的 yaml 变了,只需:

  cd /home/lhk/workspace/VINS-Fisheye
  ./scripts/regen_vins_config.sh

  输出会完全重写 config/my_kalibr_fisheye/ 和 launch 文件,幂等可重入。

  脚本能力

  kalibr_to_vins.py 自动处理的事:

  ┌──────────────────────┬─────────────────────────────────────────────────────────────────────────────────────┐
  │         行为         │                                        说明                                         │
  ├──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ 自动发现 kalibr 文件 │ 用 glob 匹配 *camchain-imucam*.yaml 等,文件名变了也能找到                           │
  ├──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ 自动检测相机数量     │ 看 camchain 里有几个 camN,逐个生成 yaml                                             │
  ├──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ 多种相机模型         │ omni-radtan → MEI;pinhole-radtan → PINHOLE;pinhole-equi → KANNALA_BRANDT            │
  ├──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ 外参链式计算         │ body_T_camN = T_imu_cam0 · inv(T_camN_cam0),T_camN_cam0 通过逐级累乘 T_cn_cnm1 得到 │
  ├──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ 自动写 OpenCV YAML   │ 保留 %YAML:1.0 + !!opencv-matrix 格式(PyYAML 自动 dump 不行)                        │
  ├──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ 时间偏移             │ 直接读 kalibr timeshift_cam_imu 写成 td                                             │
  ├──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ IMU 噪声放大         │ --imu-noise-scale 默认 ×10(VIO 实战经验)                                            │
  └──────────────────────┴─────────────────────────────────────────────────────────────────────────────────────┘

  常见调整(用环境变量或 CLI 覆盖)

# 换成后向相机对

  CAM_PAIR=2,3 ./scripts/regen_vins_config.sh

# 换 IMU topic

  IMU_TOPIC=/imu/data ./scripts/regen_vins_config.sh

# 换图像 topic 映射

  TOPIC_MAP="0:/cam_a/image,1:/cam_b/image" ./scripts/regen_vins_config.sh

# 不放大 IMU 噪声

  IMU_NOISE_SCALE=1.0 ./scripts/regen_vins_config.sh

# 关闭压缩图模式(需自己 republish)

  ./scripts/regen_vins_config.sh --no-compressed

# 直接看完整选项

  python3 scripts/kalibr_to_vins.py --help

  运行

# Terminal 1

```
source ~/catkin_ws/devel/setup.bash
roslaunch vins my_kalibr_fisheye.launch
```

```
source ~/catkin_ws/devel/setup.bash
roslaunch vins my_4cam_kalibr_fisheye.launch
```

  catkin_make && source devel/setup.bash
  roslaunch vins my_kalibr_fisheye.launch

# Terminal 2

```
source ~/catkin_ws/devel/setup.bash
rosbag play /home/lhk/data/Calibration/20260528-164331.bag --clock
```

source ～/catkin_ws/devel/setup.bash
rosbag play /home/lhk/data/Calibration/20260528-164331.bag --clock

# ---- 构建 ----

  docker compose --profile cpu build                       # 只构建 CPU
  docker compose --profile gpu build                       # 只构建 GPU(默认 CUDA_ARCH=7.5)
  CUDA_ARCH=8.6 docker compose --profile gpu build         # RTX 30 系
  CUDA_ARCH="7.5;8.6" docker compose --profile gpu build   # 多 arch

# ---- 一次性运行(用完即销毁)----

  xhost +local:root
  docker compose --profile cpu run --rm cpu
  docker compose --profile gpu run --rm gpu

# ---- 长驻容器(可多次 exec)----

  docker compose --profile gpu up -d gpu
  docker compose exec gpu bash                             # 第二个终端进入同一容器
  docker compose --profile gpu down                        # 关闭

# ---- 调试/检查 ----

  docker compose --profile gpu config                      # 看渲染后的最终配置
  docker compose --profile gpu logs -f                     # 看日志

## 1. Prerequisites

The essential software environment is same as VINS-Fusion. Besides, it requires OpenCV cuda version.(Only test it on OpenCV 3.4.1).
Visionworks: Optional

If you want to CUDA mode of this package, [libSGM](https://github.com/fixstars/libSGM) is required for depth estimation.

## 2. Usage

### 2.1 Change the opencv path in the CMakeLists

This package support CUDA mode and CPU mode with OpenMP enabled. If you are using on embedded device, I strongly recommend you to use this package with CUDA to achieve best performance.

By default, CUDA is automatically detected, however you can disable it by set

```cmake
set(DETECT_CUDA false)
```

When using OpenCV, I recommend that you to compile and install opencv 3.4 with CUDA to /usr/local/

If your opencv with CUDA is installed in other localization, modify the

```cmake
SET("OpenCV_DIR"  "/usr/local/share/OpenCV/")
```

If you don't have visionworks, please

```cmake
set(ENABLE_VWORKS false)
```

NVIDIA VisionWorks gives slightly better performance, however, the VisionWorks support for this package is not stable yet.

### Fisheye usage

Term 1

```bash
#If use CUDA
roslaunch vins fisheye_split.launch config_file:=/home/your_name/your_ws/src/VINS-Fusion-Fisheye/config/fisheye_ptgrey_n3/fisheye_cuda.yaml
#If use CPU
roslaunch vins fisheye_split.launch config_file:=/home/your_name/your_ws/src/VINS-Fusion-Fisheye/config/fisheye_ptgrey_n3/fisheye_cpu.yaml
```

Term 2

```bash
rosbag play fishey_vins_2020-01-30-10-38-14.bag --clock -s 12
```

Term 3(for visuallization only)

```bash
roslaunch vins vins_rviz.launch
```

GPU is default enabled, if you are not using CUDA, disable it in yaml config file.

For rosbag, you can download from https://www.dropbox.com/s/kmakksca3ns6cav/fisheye_vins_2020-01-30-10-38-14.bag?dl=0

### Parameters for fisheye

```yaml
depth_config: "depth_cpu.yaml" # config path for depth estimation, depth_cpu.yaml uses opencv SGBM, depth.yaml uses visionworks SGM, you must install visionworks before use visionworks sgm
image_width: 600 # For fisheye, this indicate the flattened image width; min 100; 300 - 500 is good for vins
fisheye_fov: 235 # Your fisheye fov
enable_up_top: 1 #Choose direction you use
enable_down_top: 1
enable_up_side: 1
enable_down_side: 1
enable_rear_side: 1
thres_outlier : 5.0 # outlier thres for backend
tri_max_err: 3.0 #outlier thres for triangulate

depth_estimate_baseline: 0.05 # mini baseline for pts initialization
top_cnt: 30 #number of track point for top view
side_cnt: 30 #number of track point for side view
max_solve_cnt: 30 # Max Point for solve; highly influence performace
show_track: 0 # if display track
use_vxworks: 0 #use vision works for front-end; not as stable as CUDA now

enable_depth: 1 # If estimate depth cloud; only available for dual fisheye now
rgb_depth_cloud: 0 # -1: point no texture,  0 depth cloud will be gray, 1 depth cloud will be colored;
#Note that textured and colored depth cloud will slow down whole system
use_gpu: 1 # If using GPU
use_vworks: 0 # If using visionworks
```

Parameter for depth estimation

```yaml
#choose the depth you want estimate
enable_front: 1
enable_left: 1
enable_right: 1
enable_rear: 0
#downsample ration
downsample_ratio: 0.5
#choose use cpu or visionworks
use_vworks: 0

#Publish cloud jump step
pub_cloud_step: 1
#If show dispartity
show_disparity: 0
#If publish depth map image
pub_depth_map: 1
#Publish cloud in radius
depth_cloud_radius: 10
#If publish all depth cloud in a topic
pub_cloud_all: 1
#If publish all depth cloud in every direction
pub_cloud_per_direction: 0
```

# Related Paper

__Omni-swarm: A Decentralized Omnidirectional Visual-Inertial-UWB State Estimation System for Aerial Swarm__ The VINS-Fisheye is a part of Omni-swarm. If you want use VIN-Fisheye as a part of your research project, please cite this paper.

# See also

__Autonomous aerial robot using dual‐fisheye cameras, Wenliang Gao, Kaixuan Wang, Wenchao Ding, Fei Gao, Tong Qin, Shaojie Shen, 2020 Journal of Field Robotics (JFR)__ for Fisheye camera navigation. The basic idea of this project is from this paper.

__H. Xu, L. Wang, Y. Zhang and S. Shen. (2020) Decentralized visual-inertial-UWB fusion for relative state estimation of aerial swarm. in 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE.__ for swarm localization which is this project developed for.

# VINS-Fusion

## An optimization-based multi-sensor state estimator

<img src="https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/blob/master/support_files/image/vins_logo.png" width = 55% height = 55% div align=left />
<img src="https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/blob/master/support_files/image/kitti.png" width = 34% height = 34% div align=center />

VINS-Fusion is an optimization-based multi-sensor state estimator, which achieves accurate self-localization for autonomous applications (drones, cars, and AR/VR). VINS-Fusion is an extension of [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono), which supports multiple visual-inertial sensor types (mono camera + IMU, stereo cameras + IMU, even stereo cameras only). We also show a toy example of fusing VINS with GPS.
**Features:**

- multiple sensors support (stereo cameras / mono camera+IMU / stereo cameras+IMU)
- online spatial calibration (transformation between camera and IMU)
- online temporal calibration (time offset between camera and IMU)
- visual loop closure

<img src="https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/blob/master/support_files/image/kitti_rank.png" width = 80% height = 80% />

We are the **top** open-sourced stereo algorithm on [KITTI Odometry Benchmark](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) (12.Jan.2019).

**Authors:** [Tong Qin](http://www.qintonguav.com), Shaozu Cao, Jie Pan, [Peiliang Li](https://peiliangli.github.io/), and [Shaojie Shen](http://www.ece.ust.hk/ece.php/profile/facultydetail/eeshaojie) from the [Aerial Robotics Group](http://uav.ust.hk/), [HKUST](https://www.ust.hk/)

**Videos:**

`<a href="https://www.youtube.com/embed/1qye82aW7nI" target="_blank"><img src="http://img.youtube.com/vi/1qye82aW7nI/0.jpg"  alt="VINS" width="320" height="240" border="10" />``</a>`

**Related Papers:** (papers are not exactly same with code)

* **A General Optimization-based Framework for Local Odometry Estimation with Multiple Sensors**, Tong Qin, Jie Pan, Shaozu Cao, Shaojie Shen, aiXiv [pdf](https://arxiv.org/abs/1901.03638)
* **A General Optimization-based Framework for Global Pose Estimation with Multiple Sensors**, Tong Qin, Shaozu Cao, Jie Pan, Shaojie Shen, aiXiv [pdf](https://arxiv.org/abs/1901.03642)
* **Online Temporal Calibration for Monocular Visual-Inertial Systems**, Tong Qin, Shaojie Shen, IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS, 2018), **best student paper award** [pdf](https://ieeexplore.ieee.org/abstract/document/8593603)
* **VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator**, Tong Qin, Peiliang Li, Shaojie Shen, IEEE Transactions on Robotics [pdf](https://ieeexplore.ieee.org/document/8421746/?arnumber=8421746&source=authoralert)

*If you use VINS-Fusion for your academic research, please cite our related papers. [bib](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/blob/master/support_files/paper_bib.txt)*

## 1. Prerequisites

### 1.1 **Ubuntu** and **ROS**

Ubuntu 64-bit 16.04 or 18.04.
ROS Kinetic or Melodic. [ROS Installation](http://wiki.ros.org/ROS/Installation)

### 1.2. **Ceres Solver**

Follow [Ceres Installation](http://ceres-solver.org/installation.html).

## 2. Build VINS-Fusion

Clone the repository and catkin_make:

```
    cd ~/catkin_ws/src
    git clone https://github.com/HKUST-Aerial-Robotics/VINS-Fusion.git
    cd ../
    catkin_make
    source ~/catkin_ws/devel/setup.bash
```

(if you fail in this step, try to find another computer with clean system or reinstall Ubuntu and ROS)

## 3. EuRoC Example

Download [EuRoC MAV Dataset](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) to YOUR_DATASET_FOLDER. Take MH_01 for example, you can run VINS-Fusion with three sensor types (monocular camera + IMU, stereo cameras + IMU and stereo cameras).
Open four terminals, run vins odometry, visual loop closure(optional), rviz and play the bag file respectively.
Green path is VIO odometry; red path is odometry under visual loop closure.

### 3.1 Monocualr camera + IMU

```
    roslaunch vins vins_rviz.launch
    rosrun vins vins_node ~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_mono_imu_config.yaml 
    (optional) rosrun loop_fusion loop_fusion_node ~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_mono_imu_config.yaml 
    rosbag play YOUR_DATASET_FOLDER/MH_01_easy.bag
```

### 3.2 Stereo cameras + IMU

```
    roslaunch vins vins_rviz.launch
    rosrun vins vins_node ~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml 
    (optional) rosrun loop_fusion loop_fusion_node ~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml 
    rosbag play YOUR_DATASET_FOLDER/MH_01_easy.bag
```

### 3.3 Stereo cameras

```
    roslaunch vins vins_rviz.launch
    rosrun vins vins_node ~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_config.yaml 
    (optional) rosrun loop_fusion loop_fusion_node ~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_config.yaml 
    rosbag play YOUR_DATASET_FOLDER/MH_01_easy.bag
```

<img src="https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/blob/master/support_files/image/euroc.gif" width = 430 height = 240 />

## 4. KITTI Example

### 4.1 KITTI Odometry (Stereo)

Download [KITTI Odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) to YOUR_DATASET_FOLDER. Take sequences 00 for example,
Open two terminals, run vins and rviz respectively.
(We evaluated odometry on KITTI benchmark without loop closure funtion)

```
    roslaunch vins vins_rviz.launch
    (optional) rosrun loop_fusion loop_fusion_node ~/catkin_ws/src/VINS-Fusion/config/kitti_odom/kitti_config00-02.yaml
    rosrun vins kitti_odom_test ~/catkin_ws/src/VINS-Fusion/config/kitti_odom/kitti_config00-02.yaml YOUR_DATASET_FOLDER/sequences/00/ 
```

### 4.2 KITTI GPS Fusion (Stereo + GPS)

Download [KITTI raw dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) to YOUR_DATASET_FOLDER. Take [2011_10_03_drive_0027_synced](https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0027/2011_10_03_drive_0027_sync.zip) for example.
Open three terminals, run vins, global fusion and rviz respectively.
Green path is VIO odometry; blue path is odometry under GPS global fusion.

```
    roslaunch vins vins_rviz.launch
    rosrun vins kitti_gps_test ~/catkin_ws/src/VINS-Fusion/config/kitti_raw/kitti_10_03_config.yaml YOUR_DATASET_FOLDER/2011_10_03_drive_0027_sync/ 
    rosrun global_fusion global_fusion_node
```

<img src="https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/blob/master/support_files/image/kitti.gif" width = 430 height = 240 />

## 5. VINS-Fusion on car demonstration

Download [car bag](https://drive.google.com/open?id=10t9H1u8pMGDOI6Q2w2uezEq5Ib-Z8tLz) to YOUR_DATASET_FOLDER.
Open four terminals, run vins odometry, visual loop closure(optional), rviz and play the bag file respectively.
Green path is VIO odometry; red path is odometry under visual loop closure.

```
    roslaunch vins vins_rviz.launch
    rosrun vins vins_node ~/catkin_ws/src/VINS-Fusion/config/vi_car/vi_car.yaml 
    (optional) rosrun loop_fusion loop_fusion_node ~/catkin_ws/src/VINS-Fusion/config/vi_car/vi_car.yaml 
    rosbag play YOUR_DATASET_FOLDER/car.bag
```

<img src="https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/blob/master/support_files/image/car_gif.gif" width = 430 height = 240  />

## 6. Run with your devices

VIO is not only a software algorithm, it heavily relies on hardware quality. For beginners, we recommend you to run VIO with professional equipment, which contains global shutter cameras and hardware synchronization.

### 6.1 Configuration file

Write a config file for your device. You can take config files of EuRoC and KITTI as the example.

### 6.2 Camera calibration

VINS-Fusion support several camera models (pinhole, mei, equidistant). You can use [camera model](https://github.com/hengli/camodocal) to calibrate your cameras. We put some example data under /camera_models/calibrationdata to tell you how to calibrate.

```
cd ~/catkin_ws/src/VINS-Fusion/camera_models/camera_calib_example/
rosrun camera_models Calibrations -w 12 -h 8 -s 80 -i calibrationdata --camera-model pinhole
```

## 7. Acknowledgements

We use [ceres solver](http://ceres-solver.org/) for non-linear optimization and [DBoW2](https://github.com/dorian3d/DBoW2) for loop detection, a generic [camera model](https://github.com/hengli/camodocal) and [GeographicLib](https://geographiclib.sourceforge.io/).

## 8. License

The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.

We are still working on improving the code reliability. For any technical issues, please contact Tong Qin <qintonguavATgmail.com>.

For commercial inquiries, please contact Shaojie Shen <eeshaojieATust.hk>.
