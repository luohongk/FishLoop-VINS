# VINS-Fisheye-loop 运行说明

本文对应以下环境和数据：

- 仓库：`/home/lhk/workspace/VINS-Fisheye`
- Docker 容器：`vins-fisheye-gpu`
- Docker 镜像：`vins-fisheye:gpu`
- 原始数据：`/home/lhk/data/20260818-160433.bag`
- Kalibr 结果：`my_kalibr_result/`
- ROS：Noetic
- VINS 输入相机：Kalibr `cam1 + cam2`
- VINS 统一图像 topic：左 `/cam0/image/compressed`、右 `/cam1/image/compressed`
- bag 原始 topic：`/cam1/image/compressed`、`/cam2/image/compressed`（播放时自动重映射）
- IMU topic：`/imu/data_raw`

容器中的主要挂载关系：

```text
宿主机 /home/lhk/workspace/VINS-Fisheye
    -> 容器 /root/catkin_ws/src/VINS-Fisheye

宿主机 /home/lhk/data
    -> 容器 /data
```

## 1. 修复原始 bag

原始 bag 的 LZ4 数据本身可以被标准 LZ4 解码，但 ROS Noetic 的
`roslz4` 播放时会报错：

```text
ROSLZ4_DATA_ERROR: malformed data to decompress
```

在宿主机运行下面的命令，生成一个不覆盖原文件的修复版 bag：

```bash
cd /home/lhk/workspace/VINS-Fisheye

python3 scripts/repair_rosbag_lz4.py \
  /home/lhk/data/20260818-160433.bag \
  /home/lhk/data/20260818-160433.repaired.bag
```

修复后的文件应为：

```text
/home/lhk/data/20260818-160433.repaired.bag
```

检查 bag：

```bash
docker exec vins-fisheye-gpu bash -lc '
  source /opt/ros/noetic/setup.bash
  rosbag info /data/20260818-160433.repaired.bag
'
```

预期信息：

```text
duration:    3:12s
messages:    50016
compression: none
/cam1/image/compressed    5726 msgs
/cam2/image/compressed    5726 msgs
/imu/data_raw            38564 msgs
```

如果修复版文件已经存在，不需要重复执行本步骤。

## 2. 从 Kalibr 结果重新生成配置

在宿主机执行：

```bash
cd /home/lhk/workspace/VINS-Fisheye
./scripts/prepare_vins_fisheye_loop.sh
```

该脚本会完成以下工作：

1. 读取 `my_kalibr_result/` 中的相机、IMU 和相机-IMU 标定结果。
2. 使用 Kalibr `cam1 + cam2`，基线约为 `0.0656 m`。
3. 原样保留 Kalibr EUCM 相机模型参数。
4. 设置图像和 IMU topic。
5. 生成 VINS 主配置；loop_fusion 直接加载 Kalibr cam1/cam2 的 EUCM 内参。

主要输出文件：

```text
config/vins_fisheye_loop/stereo_imu_config.yaml
config/vins_fisheye_loop/cam1_eucm.yaml
config/vins_fisheye_loop/cam2_eucm.yaml
config/vins_fisheye_loop/loop_fusion_config.yaml
```

当前关键参数：

```yaml
cam0_calib: "cam1_eucm.yaml"
cam1_calib: "cam2_eucm.yaml"

compressed_image0_topic: "/cam0/image/compressed"  # 左，Kalibr cam1
compressed_image1_topic: "/cam1/image/compressed"  # 右，Kalibr cam2
imu_topic: "/imu/data_raw"

estimate_td: 0
td: -0.0044974052495221515

fisheye_fov: 250
use_gpu: 0
```

回环配置采用原始 EUCM 双目方案：

```yaml
image0_topic: "/vins_estimator/fisheye/left/image_raw"
image1_topic: "/vins_estimator/fisheye/right/image_raw"
cam0_calib: "cam1_eucm.yaml"
cam1_calib: "cam2_eucm.yaml"
loop_view_size: 320
loop_view_fov_deg: 100.0
loop_view_features: 80
loop_min_query_gap: 120
loop_min_inliers: 30
loop_max_translation: 3.0
```

候选检索会为两个鱼眼分别生成前、左、右、上、下 5 个规范视图，
将各视图限额后的 BRIEF 描述子集合融合进 DBoW。候选选出后，使用
cam0 原始 EUCM 像素提升得到的 bearing 执行 PnP RANSAC 几何验证。

上述回环阈值来自完整 bag 初测：排除少于等于 30 个 PnP 内点、相对平移
达到 3 m 上限，以及不足约 8 秒时间间隔的候选，以降低重复近邻和错误
闭环进入位姿图的概率。

VINS 在进入非线性求解后还会等待连续稳定 30 帧，才开始写入 `vio.csv`
并向 loop_fusion 发布关键帧。这样首次初始化若因 IMU 偏置异常被自动重试，
短暂的失败局部坐标系不会污染最终轨迹和位姿图。

`use_gpu: 0` 表示当前采用已经验证过的 CPU/OpenMP 前端。运行环境仍然是
GPU 镜像；需要试验 CUDA 路径时可以这样重新生成：

```bash
USE_GPU=1 ./scripts/prepare_vins_fisheye_loop.sh
```

## 3. 编译

在宿主机执行：

```bash
docker exec -it vins-fisheye-gpu bash -lc '
  source /opt/ros/noetic/setup.bash
  cd /root/catkin_ws
  catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -j8
'
```

编译成功后应存在：

```text
/root/catkin_ws/devel/lib/vins/vins_node_fisheye
/root/catkin_ws/devel/lib/loop_fusion/loop_fusion_node
```

## 4. 推荐：容器内一键运行

### 4.1 带 RViz 和跟踪窗口

宿主机需要允许容器访问 X11：

```bash
xhost +local:root
```

然后运行：

```bash
docker exec -it vins-fisheye-gpu bash -lc '
  cd /root/catkin_ws/src/VINS-Fisheye
  VIZ=true TRACKING_VIZ=true LOOP_FUSION=true \
  ./scripts/run_vins_fisheye_loop_container.sh \
  /data/20260818-160433.repaired.bag
'
```

### 4.2 无界面运行

服务器、SSH 或不需要 RViz 时使用：

```bash
docker exec -it vins-fisheye-gpu bash -lc '
  cd /root/catkin_ws/src/VINS-Fisheye
  VIZ=false TRACKING_VIZ=false LOOP_FUSION=true \
  ./scripts/run_vins_fisheye_loop_container.sh \
  /data/20260818-160433.repaired.bag
'
```

### 4.3 运行前自动重新编译

```bash
docker exec -it vins-fisheye-gpu bash -lc '
  cd /root/catkin_ws/src/VINS-Fisheye
  REBUILD=1 BUILD_JOBS=8 \
  VIZ=false TRACKING_VIZ=false LOOP_FUSION=true \
  ./scripts/run_vins_fisheye_loop_container.sh \
  /data/20260818-160433.repaired.bag
'
```

一键脚本会自动：

1. 检查 bag 是否存在。
2. 在需要时启动 `roscore`。
3. 启动 `vins_estimator`。
4. 启动 `loop_fusion`。
5. 默认等待 5 秒，让相机模型和回环词典完成初始化。
6. 按统一 topic 映射播放修复版 bag。
7. bag 播放完成后等待节点处理剩余消息。
8. 清理本次启动的 ROS 进程。

若机器加载词典较慢，可以增大启动等待时间；如需显示 `rosbag play`
进度，可关闭 quiet 模式：

```bash
STARTUP_WAIT_SECONDS=8 ROSBAG_QUIET=0 \
./scripts/run_vins_fisheye_loop_container.sh /data/20260818-160433.repaired.bag
```

## 5. 手工分终端运行

如果需要分别查看每个进程的终端输出，可使用下面的方式。

手工启动前也必须先在宿主机生成配置：

```bash
cd /home/lhk/workspace/VINS-Fisheye
./scripts/prepare_vins_fisheye_loop.sh
```

回环直接使用 `cam1_eucm.yaml` 和 `cam2_eucm.yaml`，不再需要额外的
`cam0_pinhole_undist_full.yaml`。

### 终端 1：ROS master

```bash
docker exec -it vins-fisheye-gpu bash
source /opt/ros/noetic/setup.bash
roscore
```

如果已有 `roscore`，本步骤可以跳过。

### 终端 2：VINS 和 loop_fusion

带界面：

```bash
docker exec -it vins-fisheye-gpu bash
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
roslaunch fishloop_vins vins_fisheye_loop.launch
```

无界面：

```bash
docker exec -it vins-fisheye-gpu bash
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
roslaunch fishloop_vins vins_fisheye_loop.launch \
  viz:=false tracking_viz:=false loop_fusion:=true
```

### 终端 3：播放 bag

```bash
docker exec -it vins-fisheye-gpu bash
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
rosbag play /data/20260818-160433.repaired.bag --clock \
  /cam1/image/compressed:=/cam0/image/compressed \
  /cam2/image/compressed:=/cam1/image/compressed
```

## 6. 验证运行状态

### 6.1 检查节点

```bash
docker exec vins-fisheye-gpu bash -lc '
  source /opt/ros/noetic/setup.bash
  source /root/catkin_ws/devel/setup.bash
  rosnode list
'
```

至少应看到：

```text
/vins_estimator
/loop_fusion
/rosout
```

### 6.2 检查输入订阅

```bash
docker exec vins-fisheye-gpu bash -lc '
  source /opt/ros/noetic/setup.bash
  source /root/catkin_ws/devel/setup.bash
  rostopic info /cam0/image/compressed
  rostopic info /cam1/image/compressed
  rostopic info /imu/data_raw
'
```

三个 topic 的 subscriber 中都应包含 `/vins_estimator`。

### 6.3 检查里程计

```bash
docker exec vins-fisheye-gpu bash -lc '
  source /opt/ros/noetic/setup.bash
  source /root/catkin_ws/devel/setup.bash
  rostopic echo -n 1 /vins_estimator/odometry/header
'
```

检查回环优化后的里程计：

```bash
docker exec vins-fisheye-gpu bash -lc '
  source /opt/ros/noetic/setup.bash
  source /root/catkin_ws/devel/setup.bash
  rostopic echo -n 1 /loop_fusion/odometry_rect/header
'
```

### 6.4 检查关键帧和回环输入

```bash
docker exec vins-fisheye-gpu bash -lc '
  source /opt/ros/noetic/setup.bash
  source /root/catkin_ws/devel/setup.bash
  rostopic hz /vins_estimator/keyframe_pose
'
```

loop_fusion 依赖以下 VINS 输出和原始双目重发布 topic：

```text
/vins_estimator/odometry
/vins_estimator/keyframe_pose
/vins_estimator/keyframe_point
/vins_estimator/extrinsic
/vins_estimator/fisheye/left/image_raw
/vins_estimator/fisheye/right/image_raw
```

## 7. 输出文件

宿主机输出目录：

```text
/home/lhk/workspace/VINS-Fisheye/data
```

主要文件：

```text
data/vio.csv
data/vio_loop.csv
```

- `vio.csv`：VINS 视觉惯性里程计轨迹。
- `vio_loop.csv`：loop_fusion 位姿图轨迹。

检查文件是否持续增长：

```bash
watch -n 1 'wc -l \
  /home/lhk/workspace/VINS-Fisheye/data/vio.csv \
  /home/lhk/workspace/VINS-Fisheye/data/vio_loop.csv'
```

## 8. 日志位置

使用一键脚本运行时：

```text
/tmp/vins_fisheye_roscore.log
/tmp/vins_fisheye_loop_run.log
```

查看最新日志：

```bash
docker exec vins-fisheye-gpu tail -200 /tmp/vins_fisheye_loop_run.log
```

重点成功日志包括：

```text
Initialization finish!
Will directly receive compressed images /cam0/image/compressed and /cam1/image/compressed
loop start load vocabulary
[loop_fusion] stereo multi-view retrieval: 2 cameras x 5 views
[loop_fusion][EUCM-PnP] ... descriptor_matches=...
[loop_fusion] accepted EUCM loop ...
```

## 9. 常见问题

### 9.1 `ROSLZ4_DATA_ERROR`

不要直接播放原始文件：

```text
/data/20260818-160433.bag
```

应播放：

```text
/data/20260818-160433.repaired.bag
```

### 9.2 `package 'vins' not found`

运行前需要加载工作区：

```bash
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
```

### 9.3 没有 `/vins_estimator/odometry`

依次检查：

1. 是否播放了修复版 bag。
2. 三个输入 topic 是否存在 publisher。
3. 配置是否使用逻辑输入 `/cam0`、`/cam1` 和 `/imu/data_raw`，并在
   `rosbag play` 时将 bag 的 `/cam1`、`/cam2` 分别重映射过去。
4. 日志中是否出现 `Initialization finish!`。
5. 是否有足够的运动和图像特征用于初始化。

### 9.4 RViz 或 OpenCV 窗口无法打开

先运行：

```bash
xhost +local:root
```

仍无法显示时，先用无界面模式验证算法：

```text
VIZ=false TRACKING_VIZ=false
```

### 9.5 清理残留进程

先查看精确进程：

```bash
docker exec vins-fisheye-gpu bash -lc '
  pgrep -af "roslaunch|rosbag play|vins_node_fisheye|loop_fusion_node"
'
```

正常情况下，使用一键脚本并按 `Ctrl-C` 即可自动清理，不需要手工结束进程。

## 10. 最短运行流程

首次运行：

```bash
cd /home/lhk/workspace/VINS-Fisheye

python3 scripts/repair_rosbag_lz4.py \
  /home/lhk/data/20260818-160433.bag \
  /home/lhk/data/20260818-160433.repaired.bag

./scripts/prepare_vins_fisheye_loop.sh

docker exec -it vins-fisheye-gpu bash -lc '
  cd /root/catkin_ws/src/VINS-Fisheye
  REBUILD=1 VIZ=false TRACKING_VIZ=false LOOP_FUSION=true \
  ./scripts/run_vins_fisheye_loop_container.sh \
  /data/20260818-160433.repaired.bag
'
```

之后重复运行只需要最后一条 `docker exec` 命令。

## 11. 本数据完整验证结果

2026-08-18 已在 `vins-fisheye-gpu` 中按本文无界面命令完整播放：

```text
/data/20260818-160433.repaired.bag
bag duration: 192.653 s
```

最终结果：

```text
vio.csv:       2794 行，轨迹时间跨度 186.200 s
vio_loop.csv:  1943 行，轨迹时间跨度 184.733 s
接受 EUCM 回环: 15 次
回环 PnP 内点: 32~57
回环相对平移: 全部 < 3 m（完整日志行中的最大值为 2.12 m）
节点崩溃:      0
非有限状态:    0
```

首次视觉惯性初始化曾因加速度偏置异常自动重试 1 次；这是估计器的安全恢复，
不是进程崩溃。稳定发布等待保证了该失败段没有写入 CSV，也没有进入
loop_fusion。第二次初始化后直到 bag 结束没有再次重启。

轨迹连续性检查：

```text
vio.csv 最大相邻位移:       0.231 m
vio.csv 最大速度:           1.852 m/s
vio_loop.csv 最大相邻位移:  0.579 m
```

`vio_loop.csv` 的最大相邻位移对应末尾 1.6 秒的输出时间间隔，等效速度约
`0.362 m/s`，不是坐标系跳变。验证日志保存在容器中：

```text
/tmp/vins_eucm_verified_full.log
```
