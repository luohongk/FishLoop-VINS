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
