1：构建镜像

  CPU 镜像：

  docker compose --profile cpu build

  GPU 镜像：

  docker compose --profile gpu build

2：运行镜像：

```

xhost +local:root && docker run -it --network=host --privileged -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v /home/lhk/workspace/FishLoop-VINS:/root/catkin_ws/src/FishLoop-VINS -v /home/lhk/data:/data --name fishloop_vins_cpu -w /root/catkin_ws fishloop_vins:cpu

GPU 版(一行)

  xhost +local:root && docker run -it --rm \
    --gpus all \
    --network=host \
    --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY="$DISPLAY" \
    -v /home/lhk/workspace/FishLoop-VINS:/root/catkin_ws/src/fishloop_vins \
    -v /home/lhk/data:/data \
    --name fishloop_vins_gpu \
    -w /root/catkin_ws \
    fishloop_vins:gpu
```

3:docker中编译：

cd ~/catkin_ws

catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -j8

运行：


  终端 1：

  source ~/catkin_ws/devel/setup.bash
  roslaunch fishloop_vins vins_fisheye_loop.launch

  终端 2 的 rosbag play 不涉及包名，因此不用改：

  source ~/catkin_ws/devel/setup.bash
  rosbag play /data/20260528-164331.bag --clock

  或者播放另一份数据：

  rosbag play /data/20260818-161025-cam1-cam2-imu-jpeg-q90.bag --clock

  注意：同一终端连续写两条 rosbag play，第二条会等第一条播放完才执行。一般是两份数据任选一份，不要同时播放。
