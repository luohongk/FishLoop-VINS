#!/usr/bin/env python3
"""
pad_fisheye_node.py
===================

A tiny ROS1 republisher: subscribes to a CompressedImage (or raw Image)
topic, decompresses it, pads top + bottom with black, and republishes as
sensor_msgs/Image.

This exists because VINS-Fisheye's FisheyeUndist assumes a radially
symmetric fisheye (single fisheye_fov scalar), but our cameras are a
1920x1920 sensor cropped to 1920x1200. Padding 360px back on top/bottom
restores the geometry assumed by FisheyeUndist; the padded regions are
just black (no information is invented).

Usage (run two instances, one per camera):
    rosrun fishloop_vins pad_fisheye_node.py \
        _input:=/fisheye/left/image_raw/compressed \
        _output:=/fisheye/left/image_padded \
        _input_compressed:=true \
        _pad_top:=360 _pad_bottom:=360

Or via the auto-generated launch file:
    roslaunch fishloop_vins my_kalibr_fisheye.launch
"""
from __future__ import annotations

import sys

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image


class FisheyePadder:
    def __init__(self) -> None:
        self.bridge = CvBridge()

        self.input_topic        = rospy.get_param("~input")
        self.output_topic       = rospy.get_param("~output")
        self.input_compressed   = bool(rospy.get_param("~input_compressed", True))
        self.pad_top            = int(rospy.get_param("~pad_top", 360))
        self.pad_bottom         = int(rospy.get_param("~pad_bottom", 360))
        self.pad_left           = int(rospy.get_param("~pad_left", 0))
        self.pad_right          = int(rospy.get_param("~pad_right", 0))
        # Border value: 0 = black, anything else = grey/white
        self.border_value       = int(rospy.get_param("~border_value", 0))

        # Throttled stats
        self._last_log_t = 0.0
        self._frame_count = 0

        self.pub = rospy.Publisher(self.output_topic, Image, queue_size=10)

        if self.input_compressed:
            self.sub = rospy.Subscriber(
                self.input_topic, CompressedImage, self.cb_compressed, queue_size=10,
                tcp_nodelay=True,
            )
        else:
            self.sub = rospy.Subscriber(
                self.input_topic, Image, self.cb_raw, queue_size=10,
                tcp_nodelay=True,
            )

        rospy.loginfo(
            "[fisheye_padder] %s (%s) -> %s   pad t=%d b=%d l=%d r=%d val=%d",
            self.input_topic,
            "CompressedImage" if self.input_compressed else "Image",
            self.output_topic,
            self.pad_top, self.pad_bottom, self.pad_left, self.pad_right,
            self.border_value,
        )

    def _pad_and_publish(self, cv_img: np.ndarray, header) -> None:
        padded = cv2.copyMakeBorder(
            cv_img,
            self.pad_top, self.pad_bottom, self.pad_left, self.pad_right,
            cv2.BORDER_CONSTANT,
            value=(self.border_value,) * (cv_img.shape[2] if cv_img.ndim == 3 else 1),
        )
        # Preserve encoding (mono8 / bgr8 / rgb8)
        if padded.ndim == 2:
            encoding = "mono8"
        else:
            encoding = "bgr8"
        out = self.bridge.cv2_to_imgmsg(padded, encoding=encoding)
        out.header = header
        self.pub.publish(out)

        # Throttled stats every ~2 s
        self._frame_count += 1
        now = rospy.get_time()
        if now - self._last_log_t >= 2.0:
            dt = max(now - self._last_log_t, 1e-6)
            rate = self._frame_count / dt
            rospy.loginfo(
                "[VINS-DBG][padder] %s -> %s   in=%dx%d -> out=%dx%d   "
                "rate=%.1f Hz   stamp=%.3f",
                self.input_topic, self.output_topic,
                cv_img.shape[1], cv_img.shape[0],
                padded.shape[1], padded.shape[0],
                rate, header.stamp.to_sec(),
            )
            self._last_log_t = now
            self._frame_count = 0

    def cb_compressed(self, msg: CompressedImage) -> None:
        try:
            cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logerr_throttle(5.0, f"[fisheye_padder] decompress failed: {e}")
            return
        if cv_img is None:
            rospy.logwarn_throttle(5.0, "[fisheye_padder] cv_bridge returned None")
            return
        self._pad_and_publish(cv_img, msg.header)

    def cb_raw(self, msg: Image) -> None:
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logerr_throttle(5.0, f"[fisheye_padder] convert failed: {e}")
            return
        self._pad_and_publish(cv_img, msg.header)


def main() -> int:
    rospy.init_node("fisheye_padder", anonymous=True)
    FisheyePadder()
    rospy.spin()
    return 0


if __name__ == "__main__":
    sys.exit(main())
