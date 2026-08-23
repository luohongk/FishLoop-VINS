/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include <eigen3/Eigen/Dense>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

extern camodocal::CameraPtr m_camera;
extern std::vector<camodocal::CameraPtr> m_cameras;
extern Eigen::Vector3d tic;
extern Eigen::Matrix3d qic;
extern std::vector<Eigen::Vector3d> loop_tics;
extern std::vector<Eigen::Matrix3d> loop_qics;
extern std::vector<std::vector<std::pair<cv::Mat, cv::Mat>>> LOOP_VIEW_MAPS;
extern std::vector<Eigen::Matrix3d> LOOP_VIEW_ROTATIONS;
extern int LOOP_VIEW_SIZE;
extern double LOOP_VIEW_FOCAL;
extern ros::Publisher pub_match_img;
extern int VISUALIZATION_SHIFT_X;
extern int VISUALIZATION_SHIFT_Y;
extern std::string BRIEF_PATTERN_FILE;
extern std::string POSE_GRAPH_SAVE_PATH;
extern int ROW;
extern int COL;
extern std::string VINS_RESULT_PATH;
extern int DEBUG_IMAGE;

// Effective focal length (pixels) of the camera that loop_fusion runs on,
// derived from m_camera at startup. Used to scale the normalized coordinates
// back to pixels for the fundamental/PnP RANSAC reprojection thresholds. For
// the VINS-Fisheye virtual "top" pinhole this is ~156, NOT the hard-coded 460
// that stock VINS-Fusion assumed for its 460-focal cameras.
extern double LOOP_FOCAL_LENGTH;
// Loop-closure tuning knobs (set from the config / defaults in pose_graph_node).
extern int LOOP_MIN_LOOP_NUM;      // min PnP inliers to accept a loop
extern int LOOP_FAST_TH;           // FAST corner threshold for old-frame BRIEF
extern int LOOP_BRIEF_DIST_TH;     // max Hamming distance for a descriptor match
extern double LOOP_BRIEF_RATIO_TH; // max best/second-best Hamming ratio
extern int LOOP_MIN_QUERY_GAP;     // only query keyframes older than this gap
extern int LOOP_MIN_DETECT_INDEX;  // do not detect loops before this keyframe
extern int LOOP_DBOW_MAX_RESULTS;  // number of DBoW candidates to retrieve
extern double LOOP_DBOW_MIN_NEIGHBOR_SCORE;   // min best DBoW score to search
extern double LOOP_DBOW_MIN_CANDIDATE_SCORE;  // min selected candidate score
extern double LOOP_MAX_YAW_DEG;           // max relative yaw accepted
extern double LOOP_MAX_TRANSLATION;       // max relative translation accepted
extern int LOOP_VIEW_FEATURES;            // descriptor cap per camera/view
extern double LOOP_EUCM_MIN_Z;            // min bearing z accepted by central PnP
extern int LOOP_RETRIEVAL_CAMERA_COUNT;   // cameras used by DBoW retrieval
extern int LOOP_RETRIEVAL_VIEW_COUNT;     // perspective views used per camera
extern int LOOP_RETRIEVAL_USE_VIO_FEATURES; // use window/VIO BRIEF for DBoW
extern int LOOP_ORB_GEOMETRY;              // use rotation-invariant ORB for PnP matching
extern int LOOP_USE_GPU;                   // use CUDA for supported loop-closure stages
extern int LOOP_ORB_FEATURES;              // dense ORB features retained in each old frame
extern int LOOP_ORB_FAST_TH;               // ORB FAST threshold
extern int LOOP_ORB_DIST_TH;               // max ORB Hamming distance
extern double LOOP_ORB_RATIO_TH;            // max ORB best/second-best ratio
extern int LOOP_ORB_EXACT_DENSE_DIST_TH;    // exact-current to dense-old Hamming threshold
extern double LOOP_ORB_EXACT_DENSE_RATIO_TH; // exact-current to dense-old ratio threshold
extern int LOOP_ORB_EXACT_DENSE_MIN_INLIERS; // final PnP inliers for ordinary hybrid mode
extern int LOOP_ORB_DENSE_DIST_TH;         // dense-query Hamming threshold
extern double LOOP_ORB_DENSE_RATIO_TH;      // dense-query ratio threshold
extern int LOOP_ORB_DENSE_MIN_GEOM;        // min dense homography inliers
extern int LOOP_ORB_DENSE_MIN_INLIERS;     // min final PnP inliers for dense mode
extern double LOOP_ORB_DENSE_H_THRESHOLD;  // dense homography RANSAC threshold in pixels
extern double LOOP_ORB_DENSE_MAX_YAW_DEG;  // max yaw after strong dense verification
extern double LOOP_ORB_ASSOC_RADIUS;        // max pixels from a VIO point to dense ORB
extern int LOOP_FORCE_FIRST_KEYFRAMES;     // append the first N frames as diagnostic candidates
extern int LOOP_FORCE_FIRST_INTERVAL;      // only force early candidates every N frames
extern int LOOP_TEMPORAL_VOTE_FRAMES;      // consecutive query frames retained for voting
extern int LOOP_TEMPORAL_VOTE_MIN_HITS;    // frames that must vote for the same old neighborhood
extern int LOOP_TEMPORAL_CANDIDATE_WINDOW; // old-keyframe index tolerance for one place
extern int LOOP_PER_CAMERA_MIN_INLIERS;     // minimum PnP inliers for one camera estimate
extern int LOOP_SINGLE_CAMERA_MIN_INLIERS;  // stronger absolute gate when only one camera solves
extern double LOOP_SINGLE_CAMERA_MIN_INLIER_RATIO;
extern double LOOP_CAMERA_CONSISTENCY_ROTATION_DEG;
extern double LOOP_CAMERA_CONSISTENCY_TRANSLATION;
extern int LOOP_PNP_RANSAC_ITERATIONS;      // hypotheses tested by PnP RANSAC
extern double LOOP_PNP_REPROJECTION_ERROR_PX; // canonical-view px, converted by LOOP_VIEW_FOCAL
extern double LOOP_PNP_CONFIDENCE;
