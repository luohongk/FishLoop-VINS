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

#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Empty.h>
#include <cv_bridge/cv_bridge.h>
#include <fishloop_vins/VIOKeyframe.h>
#include <fstream>
#include <iostream>
#include <ros/package.h>
#include <mutex>
#include <queue>
#include <thread>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "keyframe.h"
#include "utility/tic_toc.h"
#include "pose_graph.h"
#include "utility/CameraPoseVisualization.h"
#include "parameters.h"
using namespace std;

queue<sensor_msgs::ImageConstPtr> image0_buf;
queue<sensor_msgs::ImageConstPtr> image1_buf;
queue<fishloop_vins::VIOKeyframeConstPtr> keyframe_buf;
struct SynchronizedKeyframe
{
    sensor_msgs::ImageConstPtr image0;
    sensor_msgs::ImageConstPtr image1;
    fishloop_vins::VIOKeyframeConstPtr keyframe;
};
queue<SynchronizedKeyframe> synchronized_keyframe_buf;
queue<Eigen::Vector3d> odometry_buf;
std::mutex m_buf;
std::mutex m_process;
int frame_index  = 0;
int sequence = 1;
PoseGraph posegraph;
int skip_first_cnt = 0;
int SKIP_CNT;
int skip_cnt = 0;
bool load_flag = 0;
bool start_flag = 0;
double SKIP_DIS = 0;

int VISUALIZATION_SHIFT_X;
int VISUALIZATION_SHIFT_Y;
int ROW;
int COL;
int DEBUG_IMAGE;

double LOOP_FOCAL_LENGTH = 460.0;
int LOOP_PRESERVE_ORIGINAL_FLOW = 1;
int LOOP_MIN_LOOP_NUM = 25;
int LOOP_FAST_TH = 20;
int LOOP_BRIEF_DIST_TH = 80;
double LOOP_BRIEF_RATIO_TH = 0.80;
int LOOP_MIN_QUERY_GAP = 50;
int LOOP_MIN_DETECT_INDEX = 50;
int LOOP_SKIP_FIRST_KEYFRAMES = 0;
int LOOP_DBOW_MAX_RESULTS = 4;
int LOOP_CANDIDATE_NEIGHBOR_WINDOW = 0;
int LOOP_CANDIDATE_NEIGHBOR_STRIDE = 5;
double LOOP_DBOW_MIN_NEIGHBOR_SCORE = 0.05;
double LOOP_DBOW_MIN_CANDIDATE_SCORE = 0.015;
double LOOP_MAX_YAW_DEG = 30.0;
double LOOP_MAX_TRANSLATION = 20.0;
int LOOP_VIEW_FEATURES = 80;
double LOOP_EUCM_MIN_Z = 0.10;
int LOOP_RETRIEVAL_CAMERA_COUNT = 1;
int LOOP_RETRIEVAL_VIEW_COUNT = 1;
int LOOP_RETRIEVAL_USE_VIO_FEATURES = 1;
int LOOP_ORB_GEOMETRY = 1;
int LOOP_USE_GPU = 0;
int LOOP_ORB_FEATURES = 2500;
int LOOP_ORB_FAST_TH = 12;
int LOOP_ORB_DIST_TH = 64;
double LOOP_ORB_RATIO_TH = 0.85;
int LOOP_ORB_EXACT_DENSE_DIST_TH = 72;
double LOOP_ORB_EXACT_DENSE_RATIO_TH = 0.90;
int LOOP_ORB_EXACT_DENSE_MIN_INLIERS = 14;
int LOOP_ORB_DENSE_DIST_TH = 72;
double LOOP_ORB_DENSE_RATIO_TH = 0.90;
int LOOP_ORB_DENSE_MIN_GEOM = 40;
int LOOP_ORB_DENSE_MIN_INLIERS = 10;
double LOOP_ORB_DENSE_H_THRESHOLD = 5.0;
double LOOP_ORB_DENSE_MAX_YAW_DEG = 150.0;
double LOOP_ORB_ASSOC_RADIUS = 10.0;
int LOOP_FORCE_FIRST_KEYFRAMES = 0;
int LOOP_FORCE_FIRST_INTERVAL = 5;
int LOOP_TEMPORAL_VOTE_FRAMES = 3;
int LOOP_TEMPORAL_VOTE_MIN_HITS = 2;
int LOOP_TEMPORAL_CANDIDATE_WINDOW = 5;
int LOOP_PER_CAMERA_MIN_INLIERS = 6;
int LOOP_SINGLE_CAMERA_MIN_INLIERS = 12;
double LOOP_SINGLE_CAMERA_MIN_INLIER_RATIO = 0.65;
double LOOP_CAMERA_CONSISTENCY_ROTATION_DEG = 15.0;
double LOOP_CAMERA_CONSISTENCY_TRANSLATION = 1.0;
int LOOP_EPIPOLAR_MIN_MATCHES = 8;
double LOOP_EPIPOLAR_REPROJECTION_ERROR_PX = 3.0;
double LOOP_EPIPOLAR_CONFIDENCE = 0.999;
int LOOP_PNP_RANSAC_ITERATIONS = 300;
double LOOP_PNP_REPROJECTION_ERROR_PX = 10.0;
double LOOP_PNP_CONFIDENCE = 0.999;

camodocal::CameraPtr m_camera;
std::vector<camodocal::CameraPtr> m_cameras;
Eigen::Vector3d tic;
Eigen::Matrix3d qic;
std::vector<Eigen::Vector3d> loop_tics(2, Eigen::Vector3d::Zero());
std::vector<Eigen::Matrix3d> loop_qics(2, Eigen::Matrix3d::Identity());
std::vector<std::vector<std::pair<cv::Mat, cv::Mat>>> LOOP_VIEW_MAPS;
std::vector<Eigen::Matrix3d> LOOP_VIEW_ROTATIONS;
int LOOP_VIEW_SIZE = 320;
double LOOP_VIEW_FOCAL = 160.0;
ros::Publisher pub_match_img;
ros::Publisher pub_camera_pose_visual;
ros::Publisher pub_odometry_rect;
ros::Publisher pub_pose_graph_path_live;

std::deque<geometry_msgs::PoseStamped> live_vio_tail;
nav_msgs::Path latest_vio_path;
std::mutex m_live_vio_tail;
static const size_t MAX_LIVE_VIO_TAIL_SIZE = 2000;

std::string BRIEF_PATTERN_FILE;
std::string POSE_GRAPH_SAVE_PATH;
std::string VINS_RESULT_PATH;
CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
Eigen::Vector3d last_t(-100, -100, -100);
double last_image_time = -1;

// Retain enough stereo frames to absorb short processing bursts without
// discarding keyframes from the stable published trajectory.
// Cover the estimator's longest (12 s) validation interval with margin at
// 15 Hz, allowing delayed validated keyframes to find their original images.
static const size_t MAX_IMAGE_BUFFER_SIZE = 240;
static const size_t MAX_KEYFRAME_BUFFER_SIZE = 200;
static const size_t MAX_SYNCHRONIZED_KEYFRAME_BUFFER_SIZE = 300;

ros::Publisher pub_point_cloud, pub_margin_cloud;

void new_sequence()
{
    printf("new sequence\n");
	const int max_sequence = 9;
	if (sequence < max_sequence)
		sequence++;
	else
		ROS_WARN_THROTTLE(5.0,
			"reached maximum visualized sequence count (%d); reusing sequence %d",
			max_sequence, sequence);
    printf("sequence cnt %d \n", sequence);
    posegraph.posegraph_visualization->reset();
    posegraph.publish();
    skip_first_cnt = 0;
    skip_cnt = 0;
    m_buf.lock();
    while(!image0_buf.empty())
        image0_buf.pop();
    while(!image1_buf.empty())
        image1_buf.pop();
    while(!keyframe_buf.empty())
        keyframe_buf.pop();
    while(!synchronized_keyframe_buf.empty())
        synchronized_keyframe_buf.pop();
    while(!odometry_buf.empty())
        odometry_buf.pop();
    m_buf.unlock();
    {
        std::lock_guard<std::mutex> lock(m_live_vio_tail);
        live_vio_tail.clear();
        latest_vio_path.poses.clear();
    }
}

void estimator_reset_callback(const std_msgs::EmptyConstPtr &)
{
    ROS_WARN("estimator reset received; starting a new loop-fusion sequence");
    std::lock_guard<std::mutex> process_lock(m_process);
    new_sequence();

    // RViz subscribes to the live path rather than PoseGraph::publish(). Clear
    // it immediately while the estimator establishes its new local frame.
    nav_msgs::Path empty_path;
    empty_path.header.frame_id = "world";
    empty_path.header.stamp = ros::Time::now();
    pub_pose_graph_path_live.publish(empty_path);
}

void image0_callback(const sensor_msgs::ImageConstPtr &image_msg)
{
    //ROS_INFO("image_callback!");
    m_buf.lock();
    image0_buf.push(image_msg);
	while (image0_buf.size() > MAX_IMAGE_BUFFER_SIZE)
		image0_buf.pop();
    m_buf.unlock();
    //printf(" image time %f \n", image_msg->header.stamp.toSec());

    // detect unstable camera stream
    if (last_image_time == -1)
        last_image_time = image_msg->header.stamp.toSec();
	else if (image_msg->header.stamp.toSec() - last_image_time > 1.0 || image_msg->header.stamp.toSec() < last_image_time)
    {
		ROS_WARN("image discontinue %.6f -> %.6f (dt=%.6f); detect a new sequence",
			last_image_time, image_msg->header.stamp.toSec(),
			image_msg->header.stamp.toSec() - last_image_time);
        new_sequence();
    }
    last_image_time = image_msg->header.stamp.toSec();
}

void image1_callback(const sensor_msgs::ImageConstPtr &image_msg)
{
    std::lock_guard<std::mutex> lock(m_buf);
    image1_buf.push(image_msg);
	while (image1_buf.size() > MAX_IMAGE_BUFFER_SIZE)
		image1_buf.pop();
}

void point_callback(const sensor_msgs::PointCloudConstPtr &point_msg)
{
    // for visualization
    sensor_msgs::PointCloud point_cloud;
    point_cloud.header = point_msg->header;
    for (unsigned int i = 0; i < point_msg->points.size(); i++)
    {
        cv::Point3f p_3d;
        p_3d.x = point_msg->points[i].x;
        p_3d.y = point_msg->points[i].y;
        p_3d.z = point_msg->points[i].z;
        Eigen::Vector3d tmp = posegraph.r_drift * Eigen::Vector3d(p_3d.x, p_3d.y, p_3d.z) + posegraph.t_drift;
        geometry_msgs::Point32 p;
        p.x = tmp(0);
        p.y = tmp(1);
        p.z = tmp(2);
        point_cloud.points.push_back(p);
    }
    pub_point_cloud.publish(point_cloud);
}

// only for visualization
void margin_point_callback(const sensor_msgs::PointCloudConstPtr &point_msg)
{
    sensor_msgs::PointCloud point_cloud;
    point_cloud.header = point_msg->header;
    for (unsigned int i = 0; i < point_msg->points.size(); i++)
    {
        cv::Point3f p_3d;
        p_3d.x = point_msg->points[i].x;
        p_3d.y = point_msg->points[i].y;
        p_3d.z = point_msg->points[i].z;
        Eigen::Vector3d tmp = posegraph.r_drift * Eigen::Vector3d(p_3d.x, p_3d.y, p_3d.z) + posegraph.t_drift;
        geometry_msgs::Point32 p;
        p.x = tmp(0);
        p.y = tmp(1);
        p.z = tmp(2);
        point_cloud.points.push_back(p);
    }
    pub_margin_cloud.publish(point_cloud);
}

void keyframe_callback(const fishloop_vins::VIOKeyframeConstPtr &keyframe_msg)
{
    std::lock_guard<std::mutex> lock(m_buf);
    keyframe_buf.push(keyframe_msg);
	while (keyframe_buf.size() > MAX_KEYFRAME_BUFFER_SIZE)
		keyframe_buf.pop();
}

void vio_path_callback(const nav_msgs::Path::ConstPtr &path_msg)
{
    std::lock_guard<std::mutex> lock(m_live_vio_tail);
    latest_vio_path = *path_msg;
}

void vio_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    //ROS_INFO("vio_callback!");
    geometry_msgs::PoseStamped raw_pose;
    raw_pose.header = pose_msg->header;
    raw_pose.header.frame_id = "world";
    raw_pose.pose = pose_msg->pose.pose;

    // Use one transform snapshot for the corrected odometry and the complete
    // live tail, avoiding a mixed path if graph optimization updates drift
    // during this callback.
    const Matrix3d w_r_vio = posegraph.w_r_vio;
    const Vector3d w_t_vio = posegraph.w_t_vio;
    const Matrix3d r_drift = posegraph.r_drift;
    const Vector3d t_drift = posegraph.t_drift;
    auto correct_pose = [&](const geometry_msgs::PoseStamped &input) {
        geometry_msgs::PoseStamped output = input;
        Vector3d position(input.pose.position.x,
                          input.pose.position.y,
                          input.pose.position.z);
        Quaterniond orientation(input.pose.orientation.w,
                                input.pose.orientation.x,
                                input.pose.orientation.y,
                                input.pose.orientation.z);
        position = w_r_vio * position + w_t_vio;
        orientation = w_r_vio * orientation;
        position = r_drift * position + t_drift;
        orientation = r_drift * orientation;
        orientation.normalize();
        output.pose.position.x = position.x();
        output.pose.position.y = position.y();
        output.pose.position.z = position.z();
        output.pose.orientation.x = orientation.x();
        output.pose.orientation.y = orientation.y();
        output.pose.orientation.z = orientation.z();
        output.pose.orientation.w = orientation.w();
        return output;
    };

    const geometry_msgs::PoseStamped corrected_pose = correct_pose(raw_pose);
    Vector3d vio_t(corrected_pose.pose.position.x,
                   corrected_pose.pose.position.y,
                   corrected_pose.pose.position.z);
    Quaterniond vio_q(corrected_pose.pose.orientation.w,
                      corrected_pose.pose.orientation.x,
                      corrected_pose.pose.orientation.y,
                      corrected_pose.pose.orientation.z);

    nav_msgs::Odometry odometry;
    odometry.header = pose_msg->header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = vio_t.x();
    odometry.pose.pose.position.y = vio_t.y();
    odometry.pose.pose.position.z = vio_t.z();
    odometry.pose.pose.orientation.x = vio_q.x();
    odometry.pose.pose.orientation.y = vio_q.y();
    odometry.pose.pose.orientation.z = vio_q.z();
    odometry.pose.pose.orientation.w = vio_q.w();
    pub_odometry_rect.publish(odometry);

    Vector3d vio_t_cam;
    Quaterniond vio_q_cam;
    vio_t_cam = vio_t + vio_q * tic;
    vio_q_cam = vio_q * qic;        

    cameraposevisual.reset();
    cameraposevisual.add_pose(vio_t_cam, vio_q_cam);
    cameraposevisual.publish_by(pub_camera_pose_visual, pose_msg->header);

    // The graph topic contains only fully processed keyframes. During startup
    // replay it can be seconds behind the estimator, so join the optimized
    // history with all corrected VIO samples newer than the last graph pose.
    nav_msgs::Path live_path = posegraph.pathSnapshot(sequence);
    const double graph_stamp = live_path.poses.empty()
        ? -1.0 : live_path.poses.back().header.stamp.toSec();
    size_t tail_size = 0;
    {
        std::lock_guard<std::mutex> lock(m_live_vio_tail);
        // Keep buffered startup keyframes available for retrieval, but align
        // the displayed red path with the first pose visible in the green VIO
        // path so the two trajectories have the same visual start time.
        if (!latest_vio_path.poses.empty())
        {
            const ros::Time display_start =
                latest_vio_path.poses.front().header.stamp;
            live_path.poses.erase(std::remove_if(live_path.poses.begin(),
                live_path.poses.end(), [&](const geometry_msgs::PoseStamped &pose) {
                    return pose.header.stamp < display_start;
                }), live_path.poses.end());
        }
        live_vio_tail.push_back(raw_pose);
        while (!live_vio_tail.empty() && graph_stamp >= 0.0 &&
               live_vio_tail.front().header.stamp.toSec() <= graph_stamp + 1e-6)
            live_vio_tail.pop_front();
        while (live_vio_tail.size() > MAX_LIVE_VIO_TAIL_SIZE)
            live_vio_tail.pop_front();
        std::vector<geometry_msgs::PoseStamped> raw_tail;
        if (!latest_vio_path.poses.empty())
        {
            const size_t begin = latest_vio_path.poses.size() > MAX_LIVE_VIO_TAIL_SIZE
                ? latest_vio_path.poses.size() - MAX_LIVE_VIO_TAIL_SIZE : 0;
            for (size_t i = begin; i < latest_vio_path.poses.size(); ++i)
            {
                const geometry_msgs::PoseStamped &candidate = latest_vio_path.poses[i];
                const double stamp = candidate.header.stamp.toSec();
                if (stamp > graph_stamp + 1e-6 &&
                    stamp <= raw_pose.header.stamp.toSec() + 1e-6)
                    raw_tail.push_back(candidate);
            }
        }
        else
        {
            raw_tail.assign(live_vio_tail.begin(), live_vio_tail.end());
        }
        if (raw_tail.empty() ||
            raw_tail.back().header.stamp < raw_pose.header.stamp)
            raw_tail.push_back(raw_pose);
        for (const geometry_msgs::PoseStamped &tail_pose : raw_tail)
            live_path.poses.push_back(correct_pose(tail_pose));
        tail_size = raw_tail.size();
    }
    live_path.header = corrected_pose.header;
    live_path.header.frame_id = "world";
    pub_pose_graph_path_live.publish(live_path);
    ROS_INFO_THROTTLE(2.0,
        "[loop_live_path] graph_lag=%.1fms graph_poses=%zu live_tail=%zu",
        graph_stamp < 0.0 ? 0.0 :
            (corrected_pose.header.stamp.toSec() - graph_stamp) * 1000.0,
        live_path.poses.size() - tail_size, tail_size);
}

void extrinsic_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    m_process.lock();
    tic = Vector3d(pose_msg->pose.pose.position.x,
                   pose_msg->pose.pose.position.y,
                   pose_msg->pose.pose.position.z);
    qic = Quaterniond(pose_msg->pose.pose.orientation.w,
                      pose_msg->pose.pose.orientation.x,
                      pose_msg->pose.pose.orientation.y,
                      pose_msg->pose.pose.orientation.z).toRotationMatrix();
    loop_tics[0] = tic;
    loop_qics[0] = qic;
    m_process.unlock();
}

void process()
{
    while (true)
    {
        sensor_msgs::ImageConstPtr image0_msg = NULL;
        sensor_msgs::ImageConstPtr image1_msg = NULL;
        fishloop_vins::VIOKeyframeConstPtr keyframe_msg = NULL;

        // Synchronize every currently available keyframe before running the
        // expensive DBoW/ORB/PnP stage. This keeps graph backlog in a sparse
        // queue of exact stereo/keyframe triples instead of retaining every
        // incoming 15 Hz raw image until graph processing catches up.
        m_buf.lock();
        while(!image0_buf.empty() && !image1_buf.empty() && !keyframe_buf.empty())
        {
            const double target = keyframe_buf.front()->header.stamp.toSec();
            const double image_tolerance = 0.02;
            while (!image0_buf.empty() && image0_buf.front()->header.stamp.toSec() < target - image_tolerance)
                image0_buf.pop();
            while (!image1_buf.empty() && image1_buf.front()->header.stamp.toSec() < target - image_tolerance)
                image1_buf.pop();

            if (image0_buf.empty() || image1_buf.empty())
            {
                // Wait for the corresponding raw stereo images.
                break;
            }
            else if (image0_buf.front()->header.stamp.toSec() > target + image_tolerance ||
                     image1_buf.front()->header.stamp.toSec() > target + image_tolerance)
            {
                keyframe_buf.pop();
                ROS_WARN("throw VIO keyframe %.6f without synchronized raw stereo images", target);
            }
            else
            {
                SynchronizedKeyframe synchronized;
                synchronized.keyframe = keyframe_buf.front();
                keyframe_buf.pop();
                synchronized.image0 = image0_buf.front();
                image0_buf.pop();
                synchronized.image1 = image1_buf.front();
                image1_buf.pop();
                synchronized_keyframe_buf.push(synchronized);
                if (synchronized_keyframe_buf.size() >
                    MAX_SYNCHRONIZED_KEYFRAME_BUFFER_SIZE)
                {
                    const double dropped_stamp = synchronized_keyframe_buf.front()
                        .keyframe->header.stamp.toSec();
                    synchronized_keyframe_buf.pop();
                    ROS_WARN("drop synchronized VIO keyframe %.6f: processing queue exceeds %zu",
                        dropped_stamp, MAX_SYNCHRONIZED_KEYFRAME_BUFFER_SIZE);
                }
            }
        }
        if (!synchronized_keyframe_buf.empty())
        {
            const SynchronizedKeyframe synchronized = synchronized_keyframe_buf.front();
            synchronized_keyframe_buf.pop();
            keyframe_msg = synchronized.keyframe;
            image0_msg = synchronized.image0;
            image1_msg = synchronized.image1;
        }
        ROS_INFO_THROTTLE(2.0,
            "[loop_sync] raw=(%zu,%zu) pending_keyframes=%zu synchronized=%zu",
            image0_buf.size(), image1_buf.size(), keyframe_buf.size(),
            synchronized_keyframe_buf.size());
        m_buf.unlock();

        if (keyframe_msg != NULL)
        {
            //printf(" pose time %f \n", pose_msg->header.stamp.toSec());
            //printf(" point time %f \n", point_msg->header.stamp.toSec());
            //printf(" image time %f \n", image_msg->header.stamp.toSec());
            // Optional compatibility knob. Fisheye bags often revisit the
            // exact startup scene, so the default must preserve every stable
            // keyframe supplied by the estimator.
            if (skip_first_cnt < LOOP_SKIP_FIRST_KEYFRAMES)
            {
                skip_first_cnt++;
                continue;
            }

            if (skip_cnt < SKIP_CNT)
            {
                skip_cnt++;
                continue;
            }
            else
            {
                skip_cnt = 0;
            }

            cv_bridge::CvImageConstPtr ptr0 = cv_bridge::toCvCopy(image0_msg, sensor_msgs::image_encodings::MONO8);
            cv_bridge::CvImageConstPtr ptr1 = cv_bridge::toCvCopy(image1_msg, sensor_msgs::image_encodings::MONO8);
            cv::Mat image = ptr0->image;
            cv::Mat image_right = ptr1->image;
            // build keyframe
            Vector3d T = Vector3d(keyframe_msg->pose_drone.position.x,
                                  keyframe_msg->pose_drone.position.y,
                                  keyframe_msg->pose_drone.position.z);
            Matrix3d R = Quaterniond(keyframe_msg->pose_drone.orientation.w,
                                     keyframe_msg->pose_drone.orientation.x,
                                     keyframe_msg->pose_drone.orientation.y,
                                     keyframe_msg->pose_drone.orientation.z).toRotationMatrix();
            if((T - last_t).norm() > SKIP_DIS)
            {
                vector<cv::Point3f> point_3d;
                vector<cv::Point2f> point_2d_uv;
                vector<cv::Point2f> point_2d_normal;
                vector<Eigen::Vector3d> point_bearing;
                vector<int> point_camera_id;
                vector<double> point_id;

                if (!keyframe_msg->loop_features.empty())
                {
                    for (const fishloop_vins::LoopFeature &feature : keyframe_msg->loop_features)
                    {
                        point_3d.emplace_back(feature.point_w.x, feature.point_w.y, feature.point_w.z);
                        point_2d_uv.emplace_back(feature.raw_uv.x, feature.raw_uv.y);
                        const double safe_z = std::abs(feature.bearing.z) > 1e-8
                            ? feature.bearing.z : (feature.bearing.z >= 0.0 ? 1e-8 : -1e-8);
                        point_2d_normal.emplace_back(feature.bearing.x / safe_z,
                            feature.bearing.y / safe_z);
                        point_bearing.emplace_back(feature.bearing.x, feature.bearing.y,
                            feature.bearing.z);
                        point_camera_id.push_back(feature.camera_id);
                        point_id.push_back(feature.feature_id);
                    }
                }
                else
                {
                    const size_t count = std::min(keyframe_msg->feature_points_3d.size(),
                        std::min(keyframe_msg->feature_points_2d_uv.size(),
                            std::min(keyframe_msg->feature_points_2d_norm.size(),
                                keyframe_msg->feature_points_id.size())));
                    for (size_t i = 0; i < count; ++i)
                    {
                        const geometry_msgs::Point32 &p3 = keyframe_msg->feature_points_3d[i];
                        const geometry_msgs::Point32 &uv = keyframe_msg->feature_points_2d_uv[i];
                        const geometry_msgs::Point32 &norm = keyframe_msg->feature_points_2d_norm[i];
                        point_3d.emplace_back(p3.x, p3.y, p3.z);
                        point_2d_uv.emplace_back(uv.x, uv.y);
                        point_2d_normal.emplace_back(norm.x, norm.y);
                        point_bearing.emplace_back(norm.x, norm.y,
                            std::abs(norm.z) > 1e-8 ? norm.z : 1.0);
                        point_camera_id.push_back((int)std::lround(uv.z));
                        point_id.push_back(keyframe_msg->feature_points_id[i]);
                    }
                }

                KeyFrame* keyframe = new KeyFrame(keyframe_msg->header.stamp.toSec(), frame_index, T, R,
                                   image, image_right,
                                   point_3d, point_2d_uv, point_2d_normal, point_bearing,
                                   point_camera_id, point_id, sequence);
                if (frame_index < 5)
                {
                    ROS_INFO("[loop_fusion][startup-kf] index=%d time=%.6f observations=%zu",
                        frame_index, keyframe_msg->header.stamp.toSec(), point_3d.size());
                }
                m_process.lock();
                start_flag = 1;
                posegraph.addKeyFrame(keyframe, 1);
                m_process.unlock();
                frame_index++;
                last_t = T;
            }
        }
        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}

void command()
{
    while(1)
    {
        char c = getchar();
        if (c == 's')
        {
            m_process.lock();
            posegraph.savePoseGraph();
            m_process.unlock();
            printf("save pose graph finish\nyou can set 'load_previous_pose_graph' to 1 in the config file to reuse it next time\n");
            printf("program shutting down...\n");
            ros::shutdown();
        }
        if (c == 'n')
            new_sequence();

        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "loop_fusion");
    ros::NodeHandle n("~");
    posegraph.registerPub(n);
    
    VISUALIZATION_SHIFT_X = 0;
    VISUALIZATION_SHIFT_Y = 0;
    SKIP_CNT = 0;
    SKIP_DIS = 0;

    if(argc != 2)
    {
        printf("please intput: rosrun fishloop loop_fusion_node [config file] \n"
               "for example: rosrun fishloop loop_fusion_node "
               "/home/tony-ws1/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
        return 0;
    }
    
    string config_file = argv[1];
    printf("config_file: %s\n", argv[1]);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings: " << config_file << std::endl;
        return 1;
    }

    cameraposevisual.setScale(0.1);
    cameraposevisual.setLineWidth(0.01);

    std::string IMAGE_TOPIC0;
    std::string IMAGE_TOPIC1;
    int LOAD_PREVIOUS_POSE_GRAPH;

    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    const std::string pkg_path = ros::package::getPath("fishloop");
    if (pkg_path.empty())
    {
        ROS_FATAL("Failed to resolve ROS package 'fishloop'. Source the rebuilt catkin workspace before launching.");
        return 1;
    }

    const std::string vocabulary_file = pkg_path + "/brief/brief_k10L6.bin";
    BRIEF_PATTERN_FILE = pkg_path + "/brief/brief_pattern.yml";

    std::ifstream vocabulary_stream(vocabulary_file, std::ios::binary);
    if (!vocabulary_stream.good())
    {
        ROS_FATAL_STREAM("Failed to open loop vocabulary: " << vocabulary_file);
        return 1;
    }
    std::ifstream brief_pattern_stream(BRIEF_PATTERN_FILE);
    if (!brief_pattern_stream.good())
    {
        ROS_FATAL_STREAM("Failed to open BRIEF pattern: " << BRIEF_PATTERN_FILE);
        return 1;
    }

    cout << "vocabulary_file: " << vocabulary_file << endl;
    cout << "BRIEF_PATTERN_FILE: " << BRIEF_PATTERN_FILE << endl;
    posegraph.loadVocabulary(vocabulary_file);

    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);
    std::string camera_calib[2];
    fsSettings["cam0_calib"] >> camera_calib[0];
    fsSettings["cam1_calib"] >> camera_calib[1];
    m_cameras.resize(2);
    for (int camera_id = 0; camera_id < 2; ++camera_id)
    {
        const std::string camera_path = configPath + "/" + camera_calib[camera_id];
        printf("EUCM cam%d calib path: %s\n", camera_id, camera_path.c_str());
        m_cameras[camera_id] = camodocal::CameraFactory::instance()
            ->generateCameraFromYamlFile(camera_path.c_str());
        if (!m_cameras[camera_id])
        {
            ROS_ERROR_STREAM("Failed to load loop_fusion cam" << camera_id << "_calib '"
                << camera_calib[camera_id] << "' resolved as '" << camera_path << "'.");
            return 1;
        }
    }
    m_camera = m_cameras[0];

    std::string vio_config_file;
    fsSettings["vio_config"] >> vio_config_file;
    const std::string vio_config_path = configPath + "/" + vio_config_file;
    cv::FileStorage fsVio(vio_config_path, cv::FileStorage::READ);
    if (!fsVio.isOpened())
    {
        ROS_ERROR_STREAM("Failed to load VIO extrinsics from " << vio_config_path);
        return 1;
    }
    for (int camera_id = 0; camera_id < 2; ++camera_id)
    {
        cv::Mat body_T_camera;
        fsVio[std::string("body_T_cam") + std::to_string(camera_id)] >> body_T_camera;
        if (body_T_camera.rows != 4 || body_T_camera.cols != 4)
        {
            ROS_ERROR_STREAM("Missing body_T_cam" << camera_id << " in " << vio_config_path);
            return 1;
        }
        cv::cv2eigen(body_T_camera(cv::Rect(0, 0, 3, 3)), loop_qics[camera_id]);
        cv::Mat translation = body_T_camera(cv::Rect(3, 0, 1, 3));
        cv::cv2eigen(translation, loop_tics[camera_id]);
    }
    fsVio.release();
    qic = loop_qics[0];
    tic = loop_tics[0];

    int view_size = 320;
    double view_fov_deg = 100.0;
    if (!fsSettings["loop_view_size"].empty())
        view_size = (int)fsSettings["loop_view_size"];
    if (!fsSettings["loop_view_fov_deg"].empty())
        view_fov_deg = (double)fsSettings["loop_view_fov_deg"];
    if (!fsSettings["loop_view_features"].empty())
        LOOP_VIEW_FEATURES = (int)fsSettings["loop_view_features"];
    if (!fsSettings["loop_eucm_min_z"].empty())
        LOOP_EUCM_MIN_Z = (double)fsSettings["loop_eucm_min_z"];
    if (!fsSettings["loop_retrieval_camera_count"].empty())
        LOOP_RETRIEVAL_CAMERA_COUNT = (int)fsSettings["loop_retrieval_camera_count"];
    if (!fsSettings["loop_retrieval_view_count"].empty())
        LOOP_RETRIEVAL_VIEW_COUNT = (int)fsSettings["loop_retrieval_view_count"];
    if (!fsSettings["loop_retrieval_use_vio_features"].empty())
        LOOP_RETRIEVAL_USE_VIO_FEATURES = (int)fsSettings["loop_retrieval_use_vio_features"];
    if (!fsSettings["loop_preserve_original_flow"].empty())
        LOOP_PRESERVE_ORIGINAL_FLOW =
            (int)fsSettings["loop_preserve_original_flow"] != 0;
    if (!fsSettings["loop_orb_geometry"].empty())
        LOOP_ORB_GEOMETRY = (int)fsSettings["loop_orb_geometry"];
    if (!fsSettings["loop_use_gpu"].empty())
        LOOP_USE_GPU = (int)fsSettings["loop_use_gpu"];
    if (!fsSettings["loop_orb_features"].empty())
        LOOP_ORB_FEATURES = std::max(100, (int)fsSettings["loop_orb_features"]);
    if (!fsSettings["loop_orb_fast_threshold"].empty())
        LOOP_ORB_FAST_TH = std::max(1, (int)fsSettings["loop_orb_fast_threshold"]);
    if (!fsSettings["loop_orb_dist_threshold"].empty())
        LOOP_ORB_DIST_TH = std::max(1, (int)fsSettings["loop_orb_dist_threshold"]);
    if (!fsSettings["loop_orb_ratio_threshold"].empty())
        LOOP_ORB_RATIO_TH = (double)fsSettings["loop_orb_ratio_threshold"];
    if (!(LOOP_ORB_RATIO_TH > 0.0 && LOOP_ORB_RATIO_TH < 1.0))
    {
        ROS_WARN("invalid loop_orb_ratio_threshold=%.3f; using 0.85", LOOP_ORB_RATIO_TH);
        LOOP_ORB_RATIO_TH = 0.85;
    }
    if (!fsSettings["loop_orb_exact_dense_dist_threshold"].empty())
        LOOP_ORB_EXACT_DENSE_DIST_TH = std::max(1,
            (int)fsSettings["loop_orb_exact_dense_dist_threshold"]);
    if (!fsSettings["loop_orb_exact_dense_ratio_threshold"].empty())
        LOOP_ORB_EXACT_DENSE_RATIO_TH =
            (double)fsSettings["loop_orb_exact_dense_ratio_threshold"];
    if (!(LOOP_ORB_EXACT_DENSE_RATIO_TH > 0.0 &&
          LOOP_ORB_EXACT_DENSE_RATIO_TH < 1.0))
    {
        ROS_WARN("invalid loop_orb_exact_dense_ratio_threshold=%.3f; using 0.90",
            LOOP_ORB_EXACT_DENSE_RATIO_TH);
        LOOP_ORB_EXACT_DENSE_RATIO_TH = 0.90;
    }
    if (!fsSettings["loop_orb_exact_dense_min_pnp_inliers"].empty())
        LOOP_ORB_EXACT_DENSE_MIN_INLIERS = std::max(8,
            (int)fsSettings["loop_orb_exact_dense_min_pnp_inliers"]);
    if (!fsSettings["loop_orb_dense_dist_threshold"].empty())
        LOOP_ORB_DENSE_DIST_TH = std::max(1, (int)fsSettings["loop_orb_dense_dist_threshold"]);
    if (!fsSettings["loop_orb_dense_ratio_threshold"].empty())
        LOOP_ORB_DENSE_RATIO_TH = (double)fsSettings["loop_orb_dense_ratio_threshold"];
    if (!(LOOP_ORB_DENSE_RATIO_TH > 0.0 && LOOP_ORB_DENSE_RATIO_TH < 1.0))
    {
        ROS_WARN("invalid loop_orb_dense_ratio_threshold=%.3f; using 0.90",
            LOOP_ORB_DENSE_RATIO_TH);
        LOOP_ORB_DENSE_RATIO_TH = 0.90;
    }
    if (!fsSettings["loop_orb_dense_min_geometric_inliers"].empty())
        LOOP_ORB_DENSE_MIN_GEOM = std::max(8,
            (int)fsSettings["loop_orb_dense_min_geometric_inliers"]);
    if (!fsSettings["loop_orb_dense_min_pnp_inliers"].empty())
        LOOP_ORB_DENSE_MIN_INLIERS = std::max(6,
            (int)fsSettings["loop_orb_dense_min_pnp_inliers"]);
    if (!fsSettings["loop_orb_dense_homography_threshold"].empty())
        LOOP_ORB_DENSE_H_THRESHOLD = std::max(1.0,
            (double)fsSettings["loop_orb_dense_homography_threshold"]);
    if (!fsSettings["loop_orb_dense_max_yaw_deg"].empty())
        LOOP_ORB_DENSE_MAX_YAW_DEG = std::max(0.0,
            (double)fsSettings["loop_orb_dense_max_yaw_deg"]);
    if (!fsSettings["loop_orb_association_radius"].empty())
        LOOP_ORB_ASSOC_RADIUS = std::max(1.0, (double)fsSettings["loop_orb_association_radius"]);
    if (!fsSettings["loop_force_first_keyframes"].empty())
        LOOP_FORCE_FIRST_KEYFRAMES = std::max(0, (int)fsSettings["loop_force_first_keyframes"]);
    if (!fsSettings["loop_force_first_interval"].empty())
        LOOP_FORCE_FIRST_INTERVAL = std::max(1, (int)fsSettings["loop_force_first_interval"]);
    if (!fsSettings["loop_temporal_vote_frames"].empty())
        LOOP_TEMPORAL_VOTE_FRAMES = std::max(1, (int)fsSettings["loop_temporal_vote_frames"]);
    if (!fsSettings["loop_temporal_vote_min_hits"].empty())
        LOOP_TEMPORAL_VOTE_MIN_HITS = std::max(1, (int)fsSettings["loop_temporal_vote_min_hits"]);
    if (!fsSettings["loop_temporal_candidate_window"].empty())
        LOOP_TEMPORAL_CANDIDATE_WINDOW = std::max(0,
            (int)fsSettings["loop_temporal_candidate_window"]);
    if (!fsSettings["loop_per_camera_min_inliers"].empty())
        LOOP_PER_CAMERA_MIN_INLIERS = std::max(4,
            (int)fsSettings["loop_per_camera_min_inliers"]);
    if (!fsSettings["loop_single_camera_min_inliers"].empty())
        LOOP_SINGLE_CAMERA_MIN_INLIERS = std::max(6,
            (int)fsSettings["loop_single_camera_min_inliers"]);
    if (!fsSettings["loop_single_camera_min_inlier_ratio"].empty())
        LOOP_SINGLE_CAMERA_MIN_INLIER_RATIO =
            (double)fsSettings["loop_single_camera_min_inlier_ratio"];
    if (!(LOOP_SINGLE_CAMERA_MIN_INLIER_RATIO > 0.0 &&
          LOOP_SINGLE_CAMERA_MIN_INLIER_RATIO <= 1.0))
    {
        ROS_WARN("invalid loop_single_camera_min_inlier_ratio=%.3f; using 0.65",
            LOOP_SINGLE_CAMERA_MIN_INLIER_RATIO);
        LOOP_SINGLE_CAMERA_MIN_INLIER_RATIO = 0.65;
    }
    LOOP_SINGLE_CAMERA_MIN_INLIERS = std::max(
        LOOP_SINGLE_CAMERA_MIN_INLIERS, LOOP_PER_CAMERA_MIN_INLIERS);
    if (!fsSettings["loop_camera_consistency_rotation_deg"].empty())
        LOOP_CAMERA_CONSISTENCY_ROTATION_DEG = std::max(0.0,
            (double)fsSettings["loop_camera_consistency_rotation_deg"]);
    if (!fsSettings["loop_camera_consistency_translation"].empty())
        LOOP_CAMERA_CONSISTENCY_TRANSLATION = std::max(0.0,
            (double)fsSettings["loop_camera_consistency_translation"]);
    if (!fsSettings["loop_epipolar_min_matches"].empty())
        LOOP_EPIPOLAR_MIN_MATCHES = std::max(8,
            (int)fsSettings["loop_epipolar_min_matches"]);
    if (!fsSettings["loop_epipolar_reprojection_error_px"].empty())
        LOOP_EPIPOLAR_REPROJECTION_ERROR_PX = std::max(0.5,
            (double)fsSettings["loop_epipolar_reprojection_error_px"]);
    if (!fsSettings["loop_epipolar_confidence"].empty())
        LOOP_EPIPOLAR_CONFIDENCE = (double)fsSettings["loop_epipolar_confidence"];
    if (!(LOOP_EPIPOLAR_CONFIDENCE > 0.0 && LOOP_EPIPOLAR_CONFIDENCE < 1.0))
    {
        ROS_WARN("invalid loop_epipolar_confidence=%.6f; using 0.999",
            LOOP_EPIPOLAR_CONFIDENCE);
        LOOP_EPIPOLAR_CONFIDENCE = 0.999;
    }
    if (!fsSettings["loop_pnp_ransac_iterations"].empty())
        LOOP_PNP_RANSAC_ITERATIONS = std::max(50,
            (int)fsSettings["loop_pnp_ransac_iterations"]);
    if (!fsSettings["loop_pnp_reprojection_error_px"].empty())
        LOOP_PNP_REPROJECTION_ERROR_PX = std::max(1.0,
            (double)fsSettings["loop_pnp_reprojection_error_px"]);
    if (!fsSettings["loop_pnp_confidence"].empty())
        LOOP_PNP_CONFIDENCE = (double)fsSettings["loop_pnp_confidence"];
    if (!(LOOP_PNP_CONFIDENCE > 0.0 && LOOP_PNP_CONFIDENCE < 1.0))
    {
        ROS_WARN("invalid loop_pnp_confidence=%.6f; using 0.999",
            LOOP_PNP_CONFIDENCE);
        LOOP_PNP_CONFIDENCE = 0.999;
    }
    LOOP_TEMPORAL_VOTE_MIN_HITS = std::min(LOOP_TEMPORAL_VOTE_MIN_HITS,
        LOOP_TEMPORAL_VOTE_FRAMES);
    LOOP_RETRIEVAL_CAMERA_COUNT = std::max(1, std::min(2, LOOP_RETRIEVAL_CAMERA_COUNT));
    LOOP_RETRIEVAL_VIEW_COUNT = std::max(1, std::min(5, LOOP_RETRIEVAL_VIEW_COUNT));
    LOOP_VIEW_SIZE = view_size;
    LOOP_VIEW_FOCAL = view_size / (2.0 * tan(view_fov_deg * M_PI / 360.0));
    const float view_focal = (float)LOOP_VIEW_FOCAL;
    LOOP_VIEW_ROTATIONS.clear();
    LOOP_VIEW_ROTATIONS.push_back(Eigen::Matrix3d::Identity());
    LOOP_VIEW_ROTATIONS.push_back(Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitY()).toRotationMatrix());
    LOOP_VIEW_ROTATIONS.push_back(Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitY()).toRotationMatrix());
    LOOP_VIEW_ROTATIONS.push_back(Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitX()).toRotationMatrix());
    LOOP_VIEW_ROTATIONS.push_back(Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitX()).toRotationMatrix());
    LOOP_VIEW_MAPS.resize(2);
    for (int camera_id = 0; camera_id < 2; ++camera_id)
    {
        for (const Eigen::Matrix3d &view_rotation : LOOP_VIEW_ROTATIONS)
        {
            cv::Mat rotation_cv;
            const Eigen::Matrix3d rectification_rotation = view_rotation.transpose();
            cv::eigen2cv(rectification_rotation, rotation_cv);
            rotation_cv.convertTo(rotation_cv, CV_32F);
            cv::Mat map1, map2;
            m_cameras[camera_id]->initUndistortRectifyMap(
                map1, map2, view_focal, view_focal, cv::Size(view_size, view_size),
                view_size / 2.0f, view_size / 2.0f, rotation_cv);
            LOOP_VIEW_MAPS[camera_id].push_back(std::make_pair(map1, map2));
        }
    }
    ROS_INFO("[loop_fusion] retrieval source=%s; fallback maps available=2 cameras x %zu views; active=%d cameras x %d views, %dx%d, fov=%.1f, cap=%d/view",
        LOOP_RETRIEVAL_USE_VIO_FEATURES ? "VIO-window" : "perspective-FAST",
        LOOP_VIEW_ROTATIONS.size(), LOOP_RETRIEVAL_CAMERA_COUNT, LOOP_RETRIEVAL_VIEW_COUNT,
        view_size, view_size, view_fov_deg, LOOP_VIEW_FEATURES);

    // Derive the effective focal length from the loaded camera so the RANSAC
    // reprojection thresholds are scaled correctly (stock code hard-codes 460,
    // which is wrong for the VINS-Fisheye virtual top pinhole, f ~= 156).
    {
        std::vector<double> cam_params;
        m_camera->writeParameters(cam_params);
        double fx = 0.0;
        if (m_camera->modelType() == camodocal::Camera::PINHOLE && cam_params.size() >= 5)
            fx = cam_params[4];                 // [k1,k2,p1,p2,fx,fy,cx,cy]
        else if (m_camera->modelType() == camodocal::Camera::MEI && cam_params.size() >= 6)
            fx = cam_params[5];                 // [xi,k1,k2,p1,p2,gamma1,gamma2,u0,v0]
        else if (m_camera->modelType() == camodocal::Camera::EUCM && cam_params.size() >= 4)
            fx = cam_params[2];                 // [alpha,beta,fx,fy,cx,cy]
        else if (cam_params.size() >= 5)
            fx = cam_params[4];
        if (fx > 1.0)
            LOOP_FOCAL_LENGTH = fx;
        printf("[loop_fusion] effective focal length for RANSAC = %.2f px\n", LOOP_FOCAL_LENGTH);
    }

    // Optional loop-closure tuning knobs (fall back to sensible defaults).
    if (!fsSettings["loop_min_inliers"].empty())
        LOOP_MIN_LOOP_NUM = (int)fsSettings["loop_min_inliers"];
    if (!fsSettings["loop_fast_threshold"].empty())
        LOOP_FAST_TH = (int)fsSettings["loop_fast_threshold"];
    if (!fsSettings["loop_brief_dist_threshold"].empty())
        LOOP_BRIEF_DIST_TH = (int)fsSettings["loop_brief_dist_threshold"];
    if (!fsSettings["loop_brief_ratio_threshold"].empty())
        LOOP_BRIEF_RATIO_TH = (double)fsSettings["loop_brief_ratio_threshold"];
    if (!(LOOP_BRIEF_RATIO_TH > 0.0 && LOOP_BRIEF_RATIO_TH < 1.0))
    {
        ROS_WARN("invalid loop_brief_ratio_threshold=%.3f; using 0.80", LOOP_BRIEF_RATIO_TH);
        LOOP_BRIEF_RATIO_TH = 0.80;
    }
    if (!fsSettings["loop_min_query_gap"].empty())
        LOOP_MIN_QUERY_GAP = (int)fsSettings["loop_min_query_gap"];
    if (!fsSettings["loop_min_detect_index"].empty())
        LOOP_MIN_DETECT_INDEX = (int)fsSettings["loop_min_detect_index"];
    if (!fsSettings["loop_skip_first_keyframes"].empty())
        LOOP_SKIP_FIRST_KEYFRAMES = std::max(0, (int)fsSettings["loop_skip_first_keyframes"]);
    if (!fsSettings["loop_keyframe_skip"].empty())
        SKIP_CNT = std::max(0, (int)fsSettings["loop_keyframe_skip"]);
    if (!fsSettings["loop_dbow_max_results"].empty())
        LOOP_DBOW_MAX_RESULTS = (int)fsSettings["loop_dbow_max_results"];
    if (!fsSettings["loop_candidate_neighbor_window"].empty())
        LOOP_CANDIDATE_NEIGHBOR_WINDOW = std::max(0,
            (int)fsSettings["loop_candidate_neighbor_window"]);
    if (!fsSettings["loop_candidate_neighbor_stride"].empty())
        LOOP_CANDIDATE_NEIGHBOR_STRIDE = std::max(1,
            (int)fsSettings["loop_candidate_neighbor_stride"]);
    if (!fsSettings["loop_dbow_min_neighbor_score"].empty())
        LOOP_DBOW_MIN_NEIGHBOR_SCORE = (double)fsSettings["loop_dbow_min_neighbor_score"];
    if (!fsSettings["loop_dbow_min_candidate_score"].empty())
        LOOP_DBOW_MIN_CANDIDATE_SCORE = (double)fsSettings["loop_dbow_min_candidate_score"];
    if (!fsSettings["loop_max_yaw_deg"].empty())
        LOOP_MAX_YAW_DEG = (double)fsSettings["loop_max_yaw_deg"];
    if (!fsSettings["loop_max_translation"].empty())
        LOOP_MAX_TRANSLATION = (double)fsSettings["loop_max_translation"];
    printf("[loop_fusion] original_flow=%d min_inliers=%d fast_th=%d brief_dist_th=%d brief_ratio=%.2f "
           "dbow_top=%d neighbor=%d/%d query_gap=%d min_idx=%d skip_first=%d keyframe_skip=%d "
           "dbow_best=%.3f dbow_candidate=%.3f max_yaw=%.1f max_t=%.1f "
	           "orb=%d gpu=%d orb_features=%d orb_fast=%d orb_dist=%d orb_ratio=%.2f "
           "exact_dense=%d/%.2f/%d "
           "dense_dist=%d dense_ratio=%.2f dense_geom=%d dense_pnp=%d dense_h=%.1f dense_yaw=%.1f "
           "orb_radius=%.1f "
           "force_first=%d/%d temporal=%d/%d window=%d per_cam=%d single=%d/%.2f "
           "consistency=%.1fdeg/%.2fm epipolar=%d/%.1fpx/%.4f "
           "pnp=%diter/%.1fpx/%.4f\n",
           LOOP_PRESERVE_ORIGINAL_FLOW,
           LOOP_MIN_LOOP_NUM, LOOP_FAST_TH, LOOP_BRIEF_DIST_TH, LOOP_BRIEF_RATIO_TH,
           LOOP_DBOW_MAX_RESULTS, LOOP_CANDIDATE_NEIGHBOR_WINDOW,
           LOOP_CANDIDATE_NEIGHBOR_STRIDE, LOOP_MIN_QUERY_GAP, LOOP_MIN_DETECT_INDEX,
           LOOP_SKIP_FIRST_KEYFRAMES, SKIP_CNT,
           LOOP_DBOW_MIN_NEIGHBOR_SCORE, LOOP_DBOW_MIN_CANDIDATE_SCORE,
           LOOP_MAX_YAW_DEG, LOOP_MAX_TRANSLATION,
           LOOP_ORB_GEOMETRY, LOOP_USE_GPU, LOOP_ORB_FEATURES, LOOP_ORB_FAST_TH,
           LOOP_ORB_DIST_TH, LOOP_ORB_RATIO_TH,
           LOOP_ORB_EXACT_DENSE_DIST_TH, LOOP_ORB_EXACT_DENSE_RATIO_TH,
           LOOP_ORB_EXACT_DENSE_MIN_INLIERS,
           LOOP_ORB_DENSE_DIST_TH, LOOP_ORB_DENSE_RATIO_TH,
           LOOP_ORB_DENSE_MIN_GEOM, LOOP_ORB_DENSE_MIN_INLIERS,
           LOOP_ORB_DENSE_H_THRESHOLD, LOOP_ORB_DENSE_MAX_YAW_DEG,
           LOOP_ORB_ASSOC_RADIUS,
           LOOP_FORCE_FIRST_KEYFRAMES,
           LOOP_FORCE_FIRST_INTERVAL,
           LOOP_TEMPORAL_VOTE_MIN_HITS, LOOP_TEMPORAL_VOTE_FRAMES,
           LOOP_TEMPORAL_CANDIDATE_WINDOW, LOOP_PER_CAMERA_MIN_INLIERS,
           LOOP_SINGLE_CAMERA_MIN_INLIERS,
           LOOP_SINGLE_CAMERA_MIN_INLIER_RATIO,
           LOOP_CAMERA_CONSISTENCY_ROTATION_DEG,
           LOOP_CAMERA_CONSISTENCY_TRANSLATION,
           LOOP_EPIPOLAR_MIN_MATCHES, LOOP_EPIPOLAR_REPROJECTION_ERROR_PX,
           LOOP_EPIPOLAR_CONFIDENCE,
           LOOP_PNP_RANSAC_ITERATIONS, LOOP_PNP_REPROJECTION_ERROR_PX,
           LOOP_PNP_CONFIDENCE);

    fsSettings["image0_topic"] >> IMAGE_TOPIC0;
    fsSettings["image1_topic"] >> IMAGE_TOPIC1;
    fsSettings["pose_graph_save_path"] >> POSE_GRAPH_SAVE_PATH;
    fsSettings["output_path"] >> VINS_RESULT_PATH;
    fsSettings["save_image"] >> DEBUG_IMAGE;

    LOAD_PREVIOUS_POSE_GRAPH = fsSettings["load_previous_pose_graph"];
    VINS_RESULT_PATH = VINS_RESULT_PATH + "/vio_loop.csv";
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    int USE_IMU = fsSettings["imu"];
    posegraph.setIMUFlag(USE_IMU);
    fsSettings.release();

    if (LOAD_PREVIOUS_POSE_GRAPH)
    {
        printf("load pose graph\n");
        m_process.lock();
        posegraph.loadPoseGraph();
        m_process.unlock();
        printf("load pose graph finish\n");
        load_flag = 1;
    }
    else
    {
        printf("no previous pose graph\n");
        load_flag = 1;
    }

    ros::Subscriber sub_vio = n.subscribe("/vins_estimator/odometry", 2000, vio_callback);
	ros::Subscriber sub_vio_path = n.subscribe("/vins_estimator/path", 10, vio_path_callback);
	ros::Subscriber sub_estimator_reset = n.subscribe(
		"/vins_estimator/reset", 10, estimator_reset_callback);
	ros::Subscriber sub_image0 = n.subscribe(IMAGE_TOPIC0, 30, image0_callback);
	ros::Subscriber sub_image1 = n.subscribe(IMAGE_TOPIC1, 30, image1_callback);
	ros::Subscriber sub_keyframe = n.subscribe("/vins_estimator/viokeyframe", 100, keyframe_callback);
    ros::Subscriber sub_extrinsic = n.subscribe("/vins_estimator/extrinsic", 2000, extrinsic_callback);
    ros::Subscriber sub_point = n.subscribe("/vins_estimator/keyframe_point", 2000, point_callback);
    ros::Subscriber sub_margin_point = n.subscribe("/vins_estimator/margin_cloud", 2000, margin_point_callback);

	pub_match_img = n.advertise<sensor_msgs::Image>("match_image", 1);
    pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual", 1000);
    pub_point_cloud = n.advertise<sensor_msgs::PointCloud>("point_cloud_loop_rect", 1000);
    pub_margin_cloud = n.advertise<sensor_msgs::PointCloud>("margin_cloud_loop_rect", 1000);
    pub_odometry_rect = n.advertise<nav_msgs::Odometry>("odometry_rect", 1000);
    pub_pose_graph_path_live = n.advertise<nav_msgs::Path>("pose_graph_path_live", 1000);

    std::thread measurement_process;
    std::thread keyboard_command_process;

    measurement_process = std::thread(process);
    keyboard_command_process = std::thread(command);
    
    ros::spin();

    // The worker loops are process-lifetime threads in the upstream node.
    // Detach them before returning so Ctrl-C does not invoke std::terminate.
    if (measurement_process.joinable())
        measurement_process.detach();
    if (keyboard_command_process.joinable())
        keyboard_command_process.detach();

    return 0;
}
