rosbag play /data/20260818-161025-cam1-cam2-imu-jpeg-q90.bag --clock

docker-compose.yml可以用来构建docker镜像

```

xhost +local:root && docker run -it --network=host --privileged -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v /home/lhk/workspace/VINS-Fisheye:/root/catkin_ws/src/VINS-Fisheye -v /home/lhk/data:/data --name vins-fisheye-cpu -w /root/catkin_ws vins-fisheye:cpu

GPU 版(一行)

xhost +local:root && docker run -it --gpus all --network=host --privileged -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v /home/lhk/workspace/VINS-Fisheye:/root/catkin_ws/src/VINS-Fisheye -v /home/lhk/data:/data --name vins-fisheye-gpu -w /root/catkin_ws vins-fisheye:gpu
```

编译：

catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -j8

运行：

终端1：

```
source ~/catkin_ws/devel/setup.bash
roslaunch vins vins_fisheye_loop.launch
```

终端2：

```
source ~/catkin_ws/devel/setup.bash
rosbag play /data/20260528-164331.bag --clock
rosbag play /data/20260818-161025-cam1-cam2-imu-jpeg-q90.bag --clock
```
