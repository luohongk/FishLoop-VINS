<div align="center">
  <img src="image/logo.png" alt="FishLoop-VINS logo" width="86%">
</div>

---

## 项目简介

FishLoop-VINS 是一个运行于 ROS 1 的双鱼眼视觉惯性定位系统。在鱼眼 VIO 的基础上，系统加入了面向超广角相机的回环检测、几何验证与位姿图优化，使用双目鱼眼图像和 IMU 进行实时状态估计，并通过多视图回环融合抑制长期运行中的累计漂移。

当前仓库已完整适配双鱼眼 EUCM（Extended Unified Camera Model）。每台鱼眼相机可展开为多个规范透视视图，用于特征跟踪、回环候选检索和匹配；候选回环最终回到原始 EUCM 像素与 bearing 上完成几何验证，避免直接将鱼眼图像作为普通针孔图像处理。

## 运行效果

<div align="center">
  <img src="image/run_images.png" alt="FishLoop-VINS runtime visualization" width="100%">
</div>

[观看演示视频](video/FishLoop_VINS.mp4)

## 软件与硬件环境

推荐使用仓库提供的 Docker 环境，以避免 ROS、OpenCV、CUDA 和 Ceres 版本冲突。本仓库当前仅支持配备 NVIDIA GPU 的设备。

- Ubuntu 20.04
- ROS Noetic
- NVIDIA GPU 与可用的宿主机驱动
- [Docker](https://www.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- CUDA 11.8
- OpenCV 4.5.5 + CUDA
- Ceres Solver、Eigen3、Boost、OpenMP
- libSGM 3.0.0，用于 CUDA 深度估计路径

运行前必须安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

## 快速开始：Docker GPU 版本

### 1. 获取代码

```bash
git clone https://github.com/luohongk/FishLoop-VINS.git
cd FishLoop-VINS
```

### 2. 获取镜像

#### 方式一：本地构建

默认 GPU 架构为 Ada `8.9`，适用于 RTX 40 系列：

```bash
docker compose --profile gpu build
```

如需适配其他显卡，请根据下表设置 `CUDA_ARCH` 和 `CUDA_ARCH_PTX`：

| GPU 架构 | 常见显卡       | 架构值 |
| -------- | -------------- | ------ |
| Pascal   | GTX 10 系列    | `6.1`  |
| Volta    | V100           | `7.0`  |
| Turing   | RTX 20、T4     | `7.5`  |
| Ampere   | RTX 30、A 系列 | `8.6`  |
| Ada      | RTX 40 系列    | `8.9`  |

例如，Ampere 架构的构建命令为：

```bash
CUDA_ARCH=8.6 CUDA_ARCH_PTX=8.6 docker compose --profile gpu build
```

#### 方式二：从 Docker Hub 拉取

```bash
docker pull luohongkun0715/fishloop_vins:gpu
docker tag luohongkun0715/fishloop_vins:gpu fishloop_vins:gpu
```

### 3. 启动容器

允许容器访问宿主机 X11：

```bash
xhost +local:root
```

将仓库和数据目录挂载到容器：

```bash
docker run -it --rm \
  --gpus all \
  --network=host \
  --privileged \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -e DISPLAY="$DISPLAY" \
  -e QT_X11_NO_MITSHM=1 \
  -v /path/to/FishLoop-VINS:/root/catkin_ws/src/fishloop_vins \
  -v /path/to/data:/data \
  --name fishloop_vins_gpu \
  -w /root/catkin_ws \
  fishloop_vins:gpu
```

请替换以下路径：

- `/path/to/FishLoop-VINS`：FishLoop-VINS 仓库的绝对路径
- `/path/to/data`：存放 rosbag 数据的绝对路径

本仓库开发环境中的命令示例：

```bash
xhost +local:root && docker run -it --rm \
  --gpus all \
  --network=host \
  --privileged \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -e DISPLAY="$DISPLAY" \
  -e QT_X11_NO_MITSHM=1 \
  -v /home/lhk/workspace/FishLoop-VINS:/root/catkin_ws/src/fishloop_vins \
  -v /home/lhk/data:/data \
  --name fishloop_vins_gpu \
  -w /root/catkin_ws \
  fishloop_vins:gpu
```

### 4. 在容器内编译

GPU Dockerfile 已准备完整依赖，但挂载源码后仍需在容器中编译当前工作区：

```bash
cd /root/catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -j8
source devel/setup.bash
```

### 5. 启动 FishLoop-VINS

继续在当前容器终端中运行：

```bash
cd /root/catkin_ws
source devel/setup.bash
roslaunch fishloop_vins vins_fisheye_loop.launch
```

该 launch 文件默认同时启动：

- `fishloop_vins_node_fisheye`：双鱼眼 VIO
- `loop_fusion_node`：回环检测和位姿图优化
- RViz：轨迹、回环和地图可视化

### 6. 播放 rosbag

样例数据：[Google Drive：example.bag](https://drive.google.com/file/d/173pJxOWuc-TGzYazSCw-WAJ2W-pfcllE/view?usp=sharing)

在宿主机另开一个终端，进入同一容器并播放数据：

```bash
docker exec -it fishloop_vins_gpu bash
rosbag play /data/example.bag --clock -r 0.3
```

## 轨迹输出

默认输出目录：

```text
/root/catkin_ws/src/fishloop_vins/data
```

由于仓库被挂载到容器中，文件会同步出现在宿主机仓库的 `data/` 目录中。

| 文件                | 内容                                   |
| ------------------- | -------------------------------------- |
| `data/vio.csv`      | 原始 VIO 逐帧轨迹，并包含速度信息      |
| `data/vio_loop.csv` | 回环位姿图优化后的关键帧轨迹           |
| `data/pose_graph/`  | 可选的位姿图、关键帧及描述子持久化数据 |

`vio.csv` 字段格式：

```text
timestamp_ns,px,py,pz,qw,qx,qy,qz,vx,vy,vz
```

`vio_loop.csv` 字段格式：

```text
timestamp_ns,px,py,pz,qw,qx,qy,qz
```

## 致谢

FishLoop-VINS 基于并借鉴了以下优秀开源项目：

- [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion)：视觉惯性估计与回环融合基础框架
- [VINS-Fisheye](https://github.com/HKUST-Aerial-Robotics/VINS-Fisheye)：鱼眼 VINS 基础版
- [DBoW2](https://github.com/dorian3d/DBoW2)：基于词袋模型的场景检索
- [Ceres Solver](http://ceres-solver.org/)：非线性最小二乘优化
- [libSGM](https://github.com/fixstars/libSGM)：CUDA 半全局立体匹配

## 许可证

仓库根目录的 [LICENSE](LICENSE) 文件声明为 Apache License 2.0。需要注意，部分 ROS package 元数据以及继承或修改自上游项目的源码可能标注了不同许可证；在公开分发或商业使用前，请核对并统一项目许可证元数据，同时遵循所有第三方组件的许可要求。
