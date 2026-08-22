#!/usr/bin/env python3
"""
viz_fisheye_tracking.py
=======================

A polished stereo-fisheye tracking + VINS trajectory visualizer.

Layout (single OpenCV window, also published as sensor_msgs/Image):

    +---------------------------------------------------------------+
    | STEREO FISHEYE TRACKING + VINS    L:180  R:185  29.4fps  ...  |
    +-------------------------------+-------------------------------+
    |                               |                               |
    |       LEFT  fisheye           |       RIGHT  fisheye          |
    |       (mask + tracks)         |       (mask + tracks)         |
    |                               |                               |
    +-------------------------------+-------------------------------+
    |                                                               |
    |             VINS  trajectory  (top-down X-Y)                  |
    |                                                               |
    +---------------------------------------------------------------+

Tracks are colored by age: green = freshly detected, red = long-tracked.
Trail tails fade out. Mask is applied to the image FIRST and the eroded
inner region is used for detection (so no spurious corners on the mask
boundary).

Usage:  rosrun vins viz_fisheye_tracking.py _<param>:=<value> ...
        (the auto-generated launch file plumbs everything for you)
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import cv2
import message_filters
import numpy as np
import rospy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import CompressedImage, Image


# ============================================================================
# Color helpers
# ============================================================================

def _track_color(age: int, max_age: int) -> Tuple[int, int, int]:
    """Green (new) -> yellow -> orange -> red (long-tracked) gradient (BGR)."""
    t = min(max(age, 1) / float(max_age), 1.0)
    # Use HSV for a smoother gradient: 60 (green) -> 0 (red)
    # OpenCV HSV: H in [0, 179]
    h = int(60 * (1.0 - t))           # 60..0
    s = 255
    v = 255
    rgb = cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


# ============================================================================
# Per-camera tracker (LK + GFTT with persistent track IDs)
# ============================================================================

class CameraTracker:
    def __init__(self, name: str, max_corners: int, min_distance: int,
                 trail_max: int):
        self.name = name
        self.max_corners = max_corners
        self.min_distance = min_distance
        self.trail_max = trail_max

        self.prev_gray: Optional[np.ndarray] = None
        self.cur_pts:   Optional[np.ndarray] = None       # Nx1x2 float32
        self.cur_ids:   Optional[np.ndarray] = None       # (N,) int32
        self.next_id = 0
        self.trails: Dict[int, Deque[Tuple[float, float]]] = {}
        self.mask: Optional[np.ndarray] = None
        self.inner_mask: Optional[np.ndarray] = None      # eroded for detection

    def set_mask(self, mask: Optional[np.ndarray]) -> None:
        self.mask = mask
        if mask is None:
            self.inner_mask = None
        else:
            er = max(1, self.min_distance // 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er, er))
            self.inner_mask = cv2.erode(mask, kernel)

    def _alloc_ids(self, n: int) -> np.ndarray:
        ids = np.arange(self.next_id, self.next_id + n, dtype=np.int32)
        self.next_id += n
        return ids

    def step(self, cur_gray: np.ndarray) -> None:
        # Mask the image for detection (per user request)
        if self.inner_mask is not None:
            cur_gray_masked = cv2.bitwise_and(cur_gray, self.inner_mask)
        else:
            cur_gray_masked = cur_gray

        # First frame
        if self.prev_gray is None or self.cur_pts is None or len(self.cur_pts) == 0:
            new = cv2.goodFeaturesToTrack(
                cur_gray_masked, self.max_corners, 0.01, self.min_distance,
                mask=self.inner_mask)
            if new is None:
                self.cur_pts = np.empty((0, 1, 2), dtype=np.float32)
                self.cur_ids = np.empty((0,), dtype=np.int32)
            else:
                self.cur_pts = new.astype(np.float32)
                self.cur_ids = self._alloc_ids(len(new))
            self.prev_gray = cur_gray
            self._update_trails()
            return

        # LK on ORIGINAL (unmasked) frames — masking would create fake gradients
        cur_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, cur_gray, self.cur_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        ok = status.flatten() == 1
        good_pts = cur_pts[ok]
        good_ids = self.cur_ids[ok] if self.cur_ids is not None else np.empty((0,), np.int32)

        # Drop tracks outside the (eroded) mask
        if self.inner_mask is not None and len(good_pts) > 0:
            flat = good_pts.reshape(-1, 2)
            u = flat[:, 0].astype(np.int32).clip(0, self.inner_mask.shape[1] - 1)
            v = flat[:, 1].astype(np.int32).clip(0, self.inner_mask.shape[0] - 1)
            keep = self.inner_mask[v, u] > 0
            good_pts = good_pts[keep]
            good_ids = good_ids[keep]

        # Top up with new corners
        if len(good_pts) < self.max_corners // 2:
            feat_mask = 255 * np.ones_like(cur_gray)
            if self.inner_mask is not None:
                feat_mask = cv2.bitwise_and(feat_mask, self.inner_mask)
            for p in good_pts.reshape(-1, 2):
                cv2.circle(feat_mask, (int(p[0]), int(p[1])),
                           self.min_distance, 0, -1)
            need = self.max_corners - len(good_pts)
            extra = cv2.goodFeaturesToTrack(
                cur_gray_masked, need, 0.01, self.min_distance, mask=feat_mask)
            if extra is not None:
                new_ids = self._alloc_ids(len(extra))
                good_pts = np.vstack([good_pts.reshape(-1, 2),
                                       extra.reshape(-1, 2)]).reshape(-1, 1, 2).astype(np.float32)
                good_ids = np.concatenate([good_ids, new_ids])

        self.cur_pts = good_pts.astype(np.float32) if len(good_pts) else np.empty((0, 1, 2), np.float32)
        self.cur_ids = good_ids
        self.prev_gray = cur_gray
        self._update_trails()

    def _update_trails(self) -> None:
        if self.cur_pts is None or self.cur_ids is None:
            self.trails = {}
            return
        cur_set = set(int(i) for i in self.cur_ids.tolist())
        for tid in list(self.trails.keys()):
            if tid not in cur_set:
                del self.trails[tid]
        for p, tid in zip(self.cur_pts.reshape(-1, 2), self.cur_ids.tolist()):
            t = int(tid)
            if t not in self.trails:
                self.trails[t] = deque(maxlen=self.trail_max)
            self.trails[t].append((float(p[0]), float(p[1])))


# ============================================================================
# Drawing
# ============================================================================

def draw_camera_panel(gray_full: np.ndarray, tracker: CameraTracker,
                      label: str, target_w: Optional[int] = None,
                      font_scale: float = 0.7) -> np.ndarray:
    """Draw the per-camera visualization panel. If target_w is given AND
    smaller than the input width, resize the gray image FIRST and scale all
    point coordinates accordingly — this avoids drawing thousands of circles
    on a 1920x1920 canvas only to shrink it afterwards (the dominant cost
    in the unscaled path)."""
    if target_w is not None and target_w < gray_full.shape[1]:
        scale = target_w / float(gray_full.shape[1])
        target_h = int(round(gray_full.shape[0] * scale))
        gray = cv2.resize(gray_full, (target_w, target_h),
                          interpolation=cv2.INTER_AREA)
        if tracker.mask is not None:
            mask = cv2.resize(tracker.mask, (target_w, target_h),
                              interpolation=cv2.INTER_NEAREST)
        else:
            mask = None
    else:
        scale = 1.0
        gray = gray_full
        mask = tracker.mask

    # Mask the image first (user-requested semantics)
    if mask is not None:
        gray = cv2.bitwise_and(gray, mask)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Mask outline
    if mask is not None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (60, 220, 60), 2,
                         lineType=cv2.LINE_AA)

    # Trails (anti-aliased fading polylines), points scaled to display size
    if tracker.cur_ids is not None:
        for tid in tracker.cur_ids.tolist():
            trail = tracker.trails.get(int(tid))
            if not trail or len(trail) < 2:
                continue
            base = _track_color(len(trail), tracker.trail_max)
            pts_arr = np.array([(int(p[0] * scale), int(p[1] * scale))
                                for p in trail], dtype=np.int32)
            for i in range(1, len(pts_arr)):
                a = i / float(len(pts_arr))   # 0..1, older = dimmer
                col = (int(base[0] * a), int(base[1] * a), int(base[2] * a))
                cv2.line(canvas, tuple(pts_arr[i - 1]), tuple(pts_arr[i]),
                         col, 2, lineType=cv2.LINE_AA)

    # Current points (scaled to display)
    if tracker.cur_pts is not None and tracker.cur_ids is not None:
        for p, tid in zip(tracker.cur_pts.reshape(-1, 2), tracker.cur_ids.tolist()):
            age = len(tracker.trails.get(int(tid), []))
            color = _track_color(age, tracker.trail_max)
            xy = (int(p[0] * scale), int(p[1] * scale))
            cv2.circle(canvas, xy, 5, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, xy, 5, (10, 10, 10), 1, lineType=cv2.LINE_AA)

    # Camera label badge
    n = 0 if tracker.cur_pts is None else len(tracker.cur_pts)
    badge = f" {label}  pts={n} "
    (tw, th), bl = cv2.getTextSize(badge, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)
    cv2.rectangle(canvas, (10, 10), (10 + tw + 10, 10 + th + bl + 10),
                  (40, 40, 40), -1)
    cv2.rectangle(canvas, (10, 10), (10 + tw + 10, 10 + th + bl + 10),
                  (180, 180, 180), 1)
    cv2.putText(canvas, badge, (15, 10 + th + 5),
                cv2.FONT_HERSHEY_DUPLEX, font_scale, (220, 220, 220), 1,
                lineType=cv2.LINE_AA)
    return canvas


def draw_trajectory_panel(width: int, height: int,
                          path_xy: list, cur_pose: Optional[tuple],
                          status_line: str = "") -> np.ndarray:
    canvas = np.full((height, width, 3), 22, dtype=np.uint8)
    title = "VINS  trajectory  (top-down X-Y)"
    cv2.putText(canvas, title, (15, 25),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 1,
                lineType=cv2.LINE_AA)
    if status_line:
        cv2.putText(canvas, status_line, (15, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 200, 220), 1,
                    lineType=cv2.LINE_AA)

    if not path_xy:
        cv2.putText(canvas, "[Waiting for /vins_estimator/path ...]",
                    (15, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1,
                    lineType=cv2.LINE_AA)
        return canvas

    arr = np.asarray(path_xy, dtype=np.float32)
    if cur_pose is not None:
        arr = np.vstack([arr, np.array([[cur_pose[0], cur_pose[1]]], dtype=np.float32)])

    pad_px = 35
    x_min, y_min = arr.min(axis=0)
    x_max, y_max = arr.max(axis=0)
    cx_w = (x_min + x_max) * 0.5
    cy_w = (y_min + y_max) * 0.5
    span = max(x_max - x_min, y_max - y_min, 1.0) * 1.15
    s = min((width - 2 * pad_px) / span, (height - 2 * pad_px) / span)

    def proj(x: float, y: float) -> Tuple[int, int]:
        u = width  * 0.5 + s * (x - cx_w)
        v = height * 0.5 - s * (y - cy_w)   # flip Y so up = +Y
        return int(u), int(v)

    # Auto-pick grid step
    if   span > 100: grid = 20.0
    elif span > 30:  grid = 5.0
    elif span > 10:  grid = 2.0
    elif span > 3:   grid = 1.0
    elif span > 1:   grid = 0.5
    else:            grid = 0.1
    grid_color = (45, 45, 45)
    gx = np.floor((cx_w - span / 2) / grid) * grid
    while gx <= cx_w + span / 2:
        u, _ = proj(gx, 0.0)
        cv2.line(canvas, (u, 30), (u, height - 25), grid_color, 1)
        gx += grid
    gy = np.floor((cy_w - span / 2) / grid) * grid
    while gy <= cy_w + span / 2:
        _, v = proj(0.0, gy)
        cv2.line(canvas, (5, v), (width - 5, v), grid_color, 1)
        gy += grid

    # Origin marker
    ou, ov = proj(0.0, 0.0)
    if 0 <= ou < width and 0 <= ov < height:
        cv2.drawMarker(canvas, (ou, ov), (90, 200, 200),
                       cv2.MARKER_CROSS, 14, 2)
        cv2.putText(canvas, "0,0", (ou + 6, ov - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 200, 200), 1,
                    lineType=cv2.LINE_AA)

    # Path polyline (gradient color along the path)
    if len(path_xy) >= 2:
        pts = np.array([proj(p[0], p[1]) for p in path_xy], dtype=np.int32)
        for i in range(1, len(pts)):
            t = i / float(len(pts))
            col = (int(60 + 100 * (1 - t)),
                   int(180 - 60 * (1 - t)),
                   int(255))
            cv2.line(canvas, tuple(pts[i - 1]), tuple(pts[i]), col, 2,
                     lineType=cv2.LINE_AA)

    # Current pose triangle pointing in heading
    if cur_pose is not None:
        x, y, yaw = cur_pose
        cu, cv_ = proj(x, y)
        size = 14
        ang = -yaw   # screen-Y is flipped
        cos_a, sin_a = float(np.cos(ang)), float(np.sin(ang))
        tip = (int(cu + size * cos_a),
               int(cv_ + size * sin_a))
        bl  = (int(cu - 0.6 * size * cos_a + 0.6 * size * sin_a),
               int(cv_ - 0.6 * size * sin_a - 0.6 * size * cos_a))
        br  = (int(cu - 0.6 * size * cos_a - 0.6 * size * sin_a),
               int(cv_ - 0.6 * size * sin_a + 0.6 * size * cos_a))
        tri = np.array([tip, bl, br], dtype=np.int32)
        cv2.fillConvexPoly(canvas, tri, (60, 60, 240))
        cv2.polylines(canvas, [tri], True, (240, 240, 240), 1,
                      lineType=cv2.LINE_AA)

    # Scale bar (one grid step)
    bar_px = int(s * grid)
    by = height - 30
    cv2.line(canvas, (15, by), (15 + bar_px, by), (220, 220, 220), 2,
             lineType=cv2.LINE_AA)
    cv2.line(canvas, (15, by - 4),         (15, by + 4),         (220, 220, 220), 1)
    cv2.line(canvas, (15 + bar_px, by - 4), (15 + bar_px, by + 4), (220, 220, 220), 1)
    cv2.putText(canvas, f"{grid:g} m",
                (15 + bar_px + 8, by + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1,
                lineType=cv2.LINE_AA)
    return canvas


def draw_header(width: int, height: int, stats: dict) -> np.ndarray:
    canvas = np.full((height, width, 3), 35, dtype=np.uint8)
    cv2.line(canvas, (0, height - 1), (width, height - 1),
             (90, 90, 90), 1)
    cv2.putText(canvas, "STEREO  FISHEYE  +  VINS  TRAJECTORY",
                (15, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (230, 230, 230), 1,
                lineType=cv2.LINE_AA)
    info1 = (f"L:{stats['left_pts']:3d}  R:{stats['right_pts']:3d}  "
             f"{stats['fps']:.1f} fps")
    cv2.putText(canvas, info1, (width - 480, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 120), 1,
                lineType=cv2.LINE_AA)
    info2 = stats.get("pose_str", "")
    if info2:
        cv2.putText(canvas, info2, (width - 270, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 200, 240), 1,
                    lineType=cv2.LINE_AA)
    return canvas


# ============================================================================
# Main visualizer
# ============================================================================

class StereoFisheyeVisualizer:
    def __init__(self) -> None:
        self.bridge = CvBridge()

        self.left_topic       = rospy.get_param("~left_topic",  "/fisheye/left/image_padded")
        self.right_topic      = rospy.get_param("~right_topic", "/fisheye/right/image_padded")
        self.left_compressed  = bool(rospy.get_param("~left_compressed",  False))
        self.right_compressed = bool(rospy.get_param("~right_compressed", False))
        self.output_topic     = rospy.get_param("~output_topic", "/vins/fisheye_track")
        self.path_topic       = rospy.get_param("~path_topic",   "/vins_estimator/path")
        self.odom_topic       = rospy.get_param("~odom_topic",   "/vins_estimator/odometry")
        self.display          = bool(rospy.get_param("~display", True))
        self.window_name      = rospy.get_param("~window_name", "VINS Fisheye Viewer")
        self.show_width       = int(rospy.get_param("~show_width", 1600))
        self.traj_height      = int(rospy.get_param("~traj_height", 360))
        self.header_height    = int(rospy.get_param("~header_height", 44))
        self.max_corners      = int(rospy.get_param("~max_corners", 200))
        self.min_distance     = int(rospy.get_param("~min_distance", 25))
        self.trail_length     = int(rospy.get_param("~trail_length", 12))

        self.mask_path        = rospy.get_param("~mask_path", "")
        self.mask_left_path   = rospy.get_param("~mask_left_path",  self.mask_path)
        self.mask_right_path  = rospy.get_param("~mask_right_path", self.mask_path)

        self.tracker_l = CameraTracker("left",  self.max_corners,
                                       self.min_distance, self.trail_length)
        self.tracker_r = CameraTracker("right", self.max_corners,
                                       self.min_distance, self.trail_length)

        self._mask_left_loaded  = False
        self._mask_right_loaded = False

        # Trajectory state
        self._lock = threading.Lock()
        self._path_xy: list = []
        self._cur_pose: Optional[tuple] = None
        self._path_len_m: float = 0.0

        # Latest fully-composed frame for the main-thread render loop
        # (cv2.imshow / cv2.waitKey MUST run on the main thread on Linux,
        # otherwise the window flickers / refreshes erratically).
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._got_frame_event = threading.Event()

        # FPS smoothing
        self._frame_times: Deque[float] = deque(maxlen=30)

        self.pub = rospy.Publisher(self.output_topic, Image, queue_size=2)

        SubL = CompressedImage if self.left_compressed  else Image
        SubR = CompressedImage if self.right_compressed else Image
        sub_l = message_filters.Subscriber(self.left_topic,  SubL, queue_size=10)
        sub_r = message_filters.Subscriber(self.right_topic, SubR, queue_size=10)
        sync = message_filters.ApproximateTimeSynchronizer(
            [sub_l, sub_r], queue_size=10, slop=0.05)
        sync.registerCallback(self._image_cb)

        rospy.Subscriber(self.path_topic, Path,     self._path_cb,
                         queue_size=2,  tcp_nodelay=True)
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb,
                         queue_size=10, tcp_nodelay=True)

        rospy.loginfo(
            "[viz_fisheye] L=%s (%s)  R=%s (%s)  path=%s  odom=%s  -> %s",
            self.left_topic,  "comp" if self.left_compressed  else "raw",
            self.right_topic, "comp" if self.right_compressed else "raw",
            self.path_topic, self.odom_topic, self.output_topic,
        )

    # ------------------- subscriber callbacks -------------------

    def _path_cb(self, msg: Path) -> None:
        xy = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        # Compute travelled length
        length = 0.0
        for i in range(1, len(xy)):
            dx = xy[i][0] - xy[i - 1][0]
            dy = xy[i][1] - xy[i - 1][1]
            length += float(np.hypot(dx, dy))
        with self._lock:
            self._path_xy = xy
            self._path_len_m = length

    def _odom_cb(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        # yaw from quaternion (Z up)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = float(np.arctan2(siny_cosp, cosy_cosp))
        with self._lock:
            self._cur_pose = (msg.pose.pose.position.x,
                              msg.pose.pose.position.y,
                              yaw)

    # ------------------- helpers -------------------

    def _to_gray(self, msg, compressed: bool) -> np.ndarray:
        if compressed:
            img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        else:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _load_mask(self, path: str, target_shape: tuple) -> Optional[np.ndarray]:
        if not path:
            return None
        raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            rospy.logerr(
                "[viz_fisheye] mask file does NOT exist or could not be "
                "decoded: %r — running WITHOUT mask. Check the path.", path)
            return None
        h, w = target_shape[:2]
        mh, mw = raw.shape
        target_aspect = w / float(h)
        mask_aspect   = mw / float(mh)
        if abs(target_aspect - mask_aspect) < 0.01:
            out = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            scale = w / float(mw)
            new_w = w
            new_h = int(round(mh * scale))
            scaled = cv2.resize(raw, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            out = np.zeros((h, w), dtype=np.uint8)
            if new_h <= h:
                top = (h - new_h) // 2
                out[top:top + new_h] = scaled
            else:
                top = (new_h - h) // 2
                out = scaled[top:top + h]
        _, out = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)
        rospy.loginfo("[viz_fisheye] mask %s loaded -> %dx%d, valid=%.1f%%",
                      path, out.shape[1], out.shape[0],
                      100.0 * float(np.count_nonzero(out)) / out.size)
        return out

    # ------------------- image callback -------------------

    def _image_cb(self, msg_l, msg_r) -> None:
        try:
            gray_l = self._to_gray(msg_l, self.left_compressed)
            gray_r = self._to_gray(msg_r, self.right_compressed)
        except Exception as e:
            rospy.logerr_throttle(5.0, f"[viz_fisheye] decode failed: {e}")
            return

        # Lazy-load masks once we know the runtime image size
        if not self._mask_left_loaded:
            self.tracker_l.set_mask(self._load_mask(self.mask_left_path, gray_l.shape))
            self._mask_left_loaded = True
        if not self._mask_right_loaded:
            self.tracker_r.set_mask(self._load_mask(self.mask_right_path, gray_r.shape))
            self._mask_right_loaded = True

        self.tracker_l.step(gray_l)
        self.tracker_r.step(gray_r)

        # Resize cam panels to display width FIRST (perf: avoid drawing
        # hundreds of circles/lines on a 1920x1920 canvas).
        target_w = self.show_width
        per_cam_w = target_w // 2
        viz_l = draw_camera_panel(gray_l, self.tracker_l, "LEFT",
                                  target_w=per_cam_w)
        viz_r = draw_camera_panel(gray_r, self.tracker_r, "RIGHT",
                                  target_w=per_cam_w)

        # FPS
        now = time.monotonic()
        self._frame_times.append(now)
        if len(self._frame_times) >= 2:
            fps = (len(self._frame_times) - 1) / max(
                self._frame_times[-1] - self._frame_times[0], 1e-6)
        else:
            fps = 0.0

        with self._lock:
            path_xy = list(self._path_xy)
            cur_pose = self._cur_pose
            path_len = self._path_len_m

        cam_h = viz_l.shape[0]
        sep = np.full((cam_h, 2, 3), 80, dtype=np.uint8)
        cam_row = np.hstack([viz_l, sep, viz_r])
        if cam_row.shape[1] != target_w:
            cam_row = cv2.resize(cam_row, (target_w, cam_h),
                                 interpolation=cv2.INTER_AREA)

        # Trajectory panel
        nl = 0 if self.tracker_l.cur_pts is None else len(self.tracker_l.cur_pts)
        nr = 0 if self.tracker_r.cur_pts is None else len(self.tracker_r.cur_pts)
        if cur_pose is not None:
            pose_str = f"x={cur_pose[0]:+6.2f}  y={cur_pose[1]:+6.2f}  yaw={np.degrees(cur_pose[2]):+6.1f}°"
        else:
            pose_str = ""
        status_line = f"path samples = {len(path_xy):d}   travelled = {path_len:.2f} m"

        traj = draw_trajectory_panel(target_w, self.traj_height,
                                     path_xy, cur_pose, status_line)

        # Header
        header = draw_header(target_w, self.header_height, {
            "left_pts": nl, "right_pts": nr, "fps": fps, "pose_str": pose_str,
        })

        # Compose final image
        final = np.vstack([header, cam_row, traj])

        # Publish (always)
        try:
            out_msg = self.bridge.cv2_to_imgmsg(final, encoding="bgr8")
            out_msg.header = msg_l.header
            self.pub.publish(out_msg)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[viz_fisheye] publish failed: {e}")

        # Hand off the frame to the main-thread render loop.
        # DO NOT call cv2.imshow() here — it must run on the main thread.
        if self.display:
            with self._frame_lock:
                self._latest_frame = final
            self._got_frame_event.set()

    # ------------------- main-thread render loop -------------------

    def render_loop(self) -> None:
        """Run on the main thread. Pulls the latest composed frame from the
        callback's slot at ~60 Hz and pumps the GUI event loop, which is
        what makes the window smooth instead of flickering."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        rate = rospy.Rate(60)
        while not rospy.is_shutdown():
            with self._frame_lock:
                frame = self._latest_frame
            if frame is not None:
                cv2.imshow(self.window_name, frame)
            # waitKey pumps Qt/GTK events; 1 ms is enough.
            cv2.waitKey(1)
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break
        cv2.destroyAllWindows()


def main() -> int:
    rospy.init_node("viz_fisheye_tracking", anonymous=True)
    viz = StereoFisheyeVisualizer()
    if viz.display:
        # Run ROS callbacks on a background thread so the main thread is
        # free for cv2.imshow / cv2.waitKey (mandatory on Linux for a
        # smooth, non-flickering window).
        spin_t = threading.Thread(target=rospy.spin, daemon=True)
        spin_t.start()
        try:
            viz.render_loop()
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
    else:
        try:
            rospy.spin()
        finally:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
