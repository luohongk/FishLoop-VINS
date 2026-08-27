/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "visualization.h"
#include <fishloop_vins/VIOKeyframe.h>
#include <sensor_msgs/PointCloud.h>
#include <fishloop_vins/FlattenImages.h>
#include "cv_bridge/cv_bridge.h"
#include <camodocal/camera_models/CameraFactory.h>
#include "../utility/ros_utility.h"
#include <algorithm>
#include <cmath>
#include <deque>
#include <mutex>
#include <std_msgs/Empty.h>
#include <utility>

using namespace ros;
using namespace Eigen;
ros::Publisher pub_odometry, pub_latest_odometry;
ros::Publisher pub_path;
ros::Publisher pub_point_cloud, pub_margin_cloud;
ros::Publisher pub_key_poses;
ros::Publisher pub_camera_pose;
ros::Publisher pub_camera_pose_right;
ros::Publisher pub_rectify_pose_left;
ros::Publisher pub_rectify_pose_right;
ros::Publisher pub_camera_pose_visual;
nav_msgs::Path path;
ros::Publisher pub_flatten_images;
ros::Publisher pub_keyframe_pose;
ros::Publisher pub_keyframe_point;
ros::Publisher pub_keyframe_image;
ros::Publisher pub_extrinsic;
ros::Publisher pub_viokeyframe;
ros::Publisher pub_viononkeyframe;
ros::Publisher pub_bias;
ros::Publisher pub_estimator_reset;

CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
static double sum_of_path = 0;
static Vector3d last_path(0.0, 0.0, 0.0);

size_t pub_counter = 0;

namespace
{
struct KeyframePublication
{
    nav_msgs::Odometry pose;
    sensor_msgs::PointCloud points;
    fishloop_vins::VIOKeyframe vio_keyframe;
    sensor_msgs::Image image;
    bool has_image = false;
};

std::deque<KeyframePublication> warmup_keyframes;
std::mutex warmup_keyframes_mutex;
// At the observed keyframe rate a full 12-second initial validation interval
// can contain more than 100 keyframes. Keep the complete attempt so a healthy
// first initialization also retains its trajectory beginning.
constexpr size_t kMaxWarmupKeyframes = 192;
std::mutex vio_path_mutex;
}

static camodocal::CameraPtr loopFusionCamera(int camera_id)
{
    static std::vector<camodocal::CameraPtr> cameras;
    if (cameras.size() < CAM_NAMES.size())
        cameras.resize(CAM_NAMES.size());
    if (camera_id < 0 || camera_id >= (int)CAM_NAMES.size())
        return camodocal::CameraPtr();
    if (!cameras[camera_id])
    {
        cameras[camera_id] = camodocal::CameraFactory::instance()
            ->generateCameraFromYamlFile(CAM_NAMES[camera_id]);
        if (cameras[camera_id])
            ROS_INFO("[loop_fusion_image_model] raw EUCM cam%d %dx%d", camera_id,
                cameras[camera_id]->imageWidth(), cameras[camera_id]->imageHeight());
        else
            ROS_ERROR_STREAM("Failed to load raw loop camera " << CAM_NAMES[camera_id]);
    }
    return cameras[camera_id];
}

void registerPub(ros::NodeHandle &n)
{
    pub_latest_odometry = n.advertise<nav_msgs::Odometry>("imu_propagate", 1000);
    pub_path = n.advertise<nav_msgs::Path>("path", 1000);
    pub_odometry = n.advertise<nav_msgs::Odometry>("odometry", 1000);
    pub_point_cloud = n.advertise<sensor_msgs::PointCloud>("point_cloud", 1000);
    pub_margin_cloud = n.advertise<sensor_msgs::PointCloud>("margin_cloud", 1000);
    pub_key_poses = n.advertise<visualization_msgs::Marker>("key_poses", 1000);
    pub_camera_pose = n.advertise<nav_msgs::Odometry>("camera_pose", 1000);
    pub_camera_pose_right = n.advertise<nav_msgs::Odometry>("camera_pose_right", 1000);
    pub_rectify_pose_left = n.advertise<geometry_msgs::PoseStamped>("rectify_pose_left", 1000);
    pub_rectify_pose_right = n.advertise<geometry_msgs::PoseStamped>("rectify_pose_right", 1000);
    pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual", 1000);
    pub_keyframe_pose = n.advertise<nav_msgs::Odometry>("keyframe_pose", 1000);
    pub_keyframe_point = n.advertise<sensor_msgs::PointCloud>("keyframe_point", 1000);
    pub_keyframe_image = n.advertise<sensor_msgs::Image>("keyframe_image", 1000);
    pub_extrinsic = n.advertise<nav_msgs::Odometry>("extrinsic", 1000);
    pub_viokeyframe = n.advertise<fishloop_vins::VIOKeyframe>("viokeyframe", 1000);
    pub_viononkeyframe = n.advertise<fishloop_vins::VIOKeyframe>("viononkeyframe", 1000);
    pub_flatten_images = n.advertise<fishloop_vins::FlattenImages>("flatten_images", 1000);
    pub_bias = n.advertise<sensor_msgs::Imu>("imu_bias", 1000);
    pub_estimator_reset = n.advertise<std_msgs::Empty>("reset", 10);

    cameraposevisual.setScale(0.1);
    cameraposevisual.setLineWidth(0.01);
}


void pubIMUBias(const Eigen::Vector3d &Ba, const Eigen::Vector3d Bg, const std_msgs::Header &header) {
    sensor_msgs::Imu bias;
    bias.header = header;
    bias.linear_acceleration.x = Ba.x();
    bias.linear_acceleration.y = Ba.y();
    bias.linear_acceleration.z = Ba.z();

    bias.angular_velocity.x = Ba.x();
    bias.angular_velocity.y = Ba.y();
    bias.angular_velocity.z = Ba.z();

    pub_bias.publish(bias);
}

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, double t)
{
    nav_msgs::Odometry odometry;
    odometry.header.stamp = ros::Time(t);
    odometry.header.frame_id = "world";
    odometry.child_frame_id = "odometry";
    odometry.pose.pose.position.x = P.x();
    odometry.pose.pose.position.y = P.y();
    odometry.pose.pose.position.z = P.z();
    odometry.pose.pose.orientation.x = Q.x();
    odometry.pose.pose.orientation.y = Q.y();
    odometry.pose.pose.orientation.z = Q.z();
    odometry.pose.pose.orientation.w = Q.w();
    odometry.twist.twist.linear.x = V.x();
    odometry.twist.twist.linear.y = V.y();
    odometry.twist.twist.linear.z = V.z();
    pub_latest_odometry.publish(odometry);

}

void printStatistics(const Estimator &estimator, double t)
{
    if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    //printf("position: %f, %f, %f\r", estimator.Ps[WINDOW_SIZE].x(), estimator.Ps[WINDOW_SIZE].y(), estimator.Ps[WINDOW_SIZE].z());
    ROS_DEBUG_STREAM("position: " << estimator.Ps[WINDOW_SIZE].transpose());
    ROS_DEBUG_STREAM("orientation: " << estimator.Vs[WINDOW_SIZE].transpose());
    if (ESTIMATE_EXTRINSIC)
    {
        cv::FileStorage fs(EX_CALIB_RESULT_PATH, cv::FileStorage::WRITE);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            //ROS_DEBUG("calibration result for camera %d", i);
            ROS_DEBUG_STREAM("extirnsic tic: " << estimator.tic[i].transpose());
            ROS_DEBUG_STREAM("extrinsic ric: " << Utility::R2ypr(estimator.ric[i]).transpose());

            Eigen::Matrix4d eigen_T = Eigen::Matrix4d::Identity();
            eigen_T.block<3, 3>(0, 0) = estimator.ric[i];
            eigen_T.block<3, 1>(0, 3) = estimator.tic[i];
            cv::Mat cv_T;
            cv::eigen2cv(eigen_T, cv_T);
            if(i == 0)
                fs << "body_T_cam0" << cv_T ;
            else
                fs << "body_T_cam1" << cv_T ;
        }
        fs.release();
    }

    static double sum_of_time = 0;
    static int sum_of_calculation = 0;
    sum_of_time += t;
    sum_of_calculation++;
    ROS_DEBUG("vo solver costs: %f ms", t);
    ROS_DEBUG("average of time %f ms", sum_of_time / sum_of_calculation);

    sum_of_path += (estimator.Ps[WINDOW_SIZE] - last_path).norm();
    last_path = estimator.Ps[WINDOW_SIZE];
    ROS_DEBUG("sum of path %f", sum_of_path);
}

namespace
{
struct WarmupOdometryPublication
{
    nav_msgs::Odometry odometry;
    double td_ms = 0.0;
};

std::deque<WarmupOdometryPublication> warmup_odometry;
std::mutex warmup_odometry_mutex;
constexpr size_t kMaxWarmupOdometry = 256;

nav_msgs::Odometry buildOdometryMessage(const Estimator &estimator,
                                        const std_msgs::Header &header)
{
    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.child_frame_id = "odometry";
    Quaterniond orientation(estimator.Rs[WINDOW_SIZE]);
    orientation.normalize();
    odometry.pose.pose.position.x = estimator.Ps[WINDOW_SIZE].x();
    odometry.pose.pose.position.y = estimator.Ps[WINDOW_SIZE].y();
    odometry.pose.pose.position.z = estimator.Ps[WINDOW_SIZE].z();
    odometry.pose.pose.orientation.x = orientation.x();
    odometry.pose.pose.orientation.y = orientation.y();
    odometry.pose.pose.orientation.z = orientation.z();
    odometry.pose.pose.orientation.w = orientation.w();
    odometry.twist.twist.linear.x = estimator.Vs[WINDOW_SIZE].x();
    odometry.twist.twist.linear.y = estimator.Vs[WINDOW_SIZE].y();
    odometry.twist.twist.linear.z = estimator.Vs[WINDOW_SIZE].z();
    return odometry;
}

void publishOdometryMessage(const nav_msgs::Odometry &odometry, double td_ms)
{
    pub_odometry.publish(odometry);

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = odometry.header;
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose = odometry.pose.pose;
    {
        std::lock_guard<std::mutex> lock(vio_path_mutex);
        path.header = odometry.header;
        path.header.frame_id = "world";
        path.poses.push_back(pose_stamped);
        pub_path.publish(path);
    }

    ofstream foutC(VINS_RESULT_PATH, ios::app);
    foutC.setf(ios::fixed, ios::floatfield);
    foutC.precision(0);
    foutC << odometry.header.stamp.toSec() * 1e9 << ",";
    foutC.precision(5);
    foutC << odometry.pose.pose.position.x << ","
          << odometry.pose.pose.position.y << ","
          << odometry.pose.pose.position.z << ","
          << odometry.pose.pose.orientation.w << ","
          << odometry.pose.pose.orientation.x << ","
          << odometry.pose.pose.orientation.y << ","
          << odometry.pose.pose.orientation.z << ","
          << odometry.twist.twist.linear.x << ","
          << odometry.twist.twist.linear.y << ","
          << odometry.twist.twist.linear.z << "," << endl;

    printf("time: %f, t: %5.3f %5.3f %5.3f q: %4.2f %4.2f %4.2f %4.2f td: %3.1fms\n",
           odometry.header.stamp.toSec(),
           odometry.pose.pose.position.x,
           odometry.pose.pose.position.y,
           odometry.pose.pose.position.z,
           odometry.pose.pose.orientation.w,
           odometry.pose.pose.orientation.x,
           odometry.pose.pose.orientation.y,
           odometry.pose.pose.orientation.z,
           td_ms);
}
}

void pubOdometry(const Estimator &estimator, const std_msgs::Header &header)
{
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        nav_msgs::Odometry odometry = buildOdometryMessage(estimator, header);
        publishOdometryMessage(odometry, estimator.td * 1000.0);

        fishloop_vins::VIOKeyframe vkf;
        vkf.header = header;
        int i = WINDOW_SIZE;
        Vector3d P = estimator.Ps[i];
        Quaterniond R = Quaterniond(estimator.Rs[i]);
        Vector3d P_r = P + R * estimator.tic[0];
        Quaterniond R_r = Quaterniond(R * estimator.ric[0]);
        vkf.pose_cam.position.x = P_r.x();
        vkf.pose_cam.position.y = P_r.y();
        vkf.pose_cam.position.z = P_r.z();
        vkf.pose_cam.orientation.x = R_r.x();
        vkf.pose_cam.orientation.y = R_r.y();
        vkf.pose_cam.orientation.z = R_r.z();
        vkf.pose_cam.orientation.w = R_r.w();

        vkf.camera_extrisinc.position.x = estimator.tic[0].x();
        vkf.camera_extrisinc.position.y = estimator.tic[0].y();
        vkf.camera_extrisinc.position.z = estimator.tic[0].z();

        Quaterniond ric = Quaterniond(estimator.ric[0]);
        ric.normalize();

        vkf.camera_extrisinc.orientation.x = ric.x();
        vkf.camera_extrisinc.orientation.y = ric.y();
        vkf.camera_extrisinc.orientation.z = ric.z();
        vkf.camera_extrisinc.orientation.w = ric.w();

        vkf.pose_drone = odometry.pose.pose;
        
        vkf.header.stamp = odometry.header.stamp;

        for (auto &_it : estimator.f_manager.feature)
        {
            auto & it_per_id = _it.second;
            int frame_size = it_per_id.feature_per_frame.size();
            // ROS_INFO("START FRAME %d FRAME_SIZE %d WIN SIZE %d solve flag %d", it_per_id.start_frame, frame_size, WINDOW_SIZE, it_per_id.solve_flag);
            if(it_per_id.start_frame < WINDOW_SIZE && it_per_id.start_frame + frame_size >= WINDOW_SIZE&& it_per_id.solve_flag < 2)
            {
                geometry_msgs::Point32 fp2d_uv;
                geometry_msgs::Point32 fp2d_norm;
                int imu_j = frame_size - 1;

                fp2d_uv.x = it_per_id.feature_per_frame[imu_j].uv.x();
                fp2d_uv.y = it_per_id.feature_per_frame[imu_j].uv.y();
                fp2d_uv.z = 0;

                fp2d_norm.x = it_per_id.feature_per_frame[imu_j].point.x();
                fp2d_norm.y = it_per_id.feature_per_frame[imu_j].point.y();
                fp2d_norm.z = 0;

                vkf.feature_points_id.push_back(it_per_id.feature_id);
                vkf.feature_points_2d_uv.push_back(fp2d_uv);
                vkf.feature_points_2d_norm.push_back(fp2d_norm);

                Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
                Vector3d w_pts_i = estimator.Rs[imu_j] * (estimator.ric[it_per_id.main_cam] * pts_i + estimator.tic[it_per_id.main_cam])
                                    + estimator.Ps[imu_j];

                geometry_msgs::Point32 p;
                p.x = w_pts_i(0);
                p.y = w_pts_i(1);
                p.z = w_pts_i(2);

                vkf.feature_points_3d.push_back(p);
                vkf.feature_points_flag.push_back(it_per_id.solve_flag);
            }

        }
        pub_viononkeyframe.publish(vkf);
    }
    
   
}

void bufferWarmupOdometry(const Estimator &estimator,
                          const std_msgs::Header &header)
{
    if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;

    WarmupOdometryPublication publication;
    publication.odometry = buildOdometryMessage(estimator, header);
    publication.td_ms = estimator.td * 1000.0;
    std::lock_guard<std::mutex> lock(warmup_odometry_mutex);
    const double stamp = publication.odometry.header.stamp.toSec();
    if (!warmup_odometry.empty() &&
        fabs(warmup_odometry.back().odometry.header.stamp.toSec() - stamp) < 1e-6)
        return;
    warmup_odometry.push_back(std::move(publication));
    while (warmup_odometry.size() > kMaxWarmupOdometry)
        warmup_odometry.pop_front();
    ROS_INFO_THROTTLE(1.0,
        "[output_warmup] buffered odometry=%zu %.3f -> %.3f",
        warmup_odometry.size(),
        warmup_odometry.front().odometry.header.stamp.toSec(),
        warmup_odometry.back().odometry.header.stamp.toSec());
}

double flushWarmupOdometry()
{
    std::deque<WarmupOdometryPublication> publications;
    {
        std::lock_guard<std::mutex> lock(warmup_odometry_mutex);
        publications.swap(warmup_odometry);
    }
    if (publications.empty())
        return -1.0;

    std::stable_sort(publications.begin(), publications.end(),
        [](const WarmupOdometryPublication &a,
           const WarmupOdometryPublication &b) {
            return a.odometry.header.stamp < b.odometry.header.stamp;
        });

    // A metric initialization can make one final gauge/scale adjustment in
    // the first nonlinear frames without subsequently failing. Retain the
    // complete stable suffix after the last implausible frame-to-frame jump,
    // rather than hiding a fixed multi-second prefix.
    constexpr double kWarmupMaxPoseStep = 0.25;
    size_t stable_begin = 0;
    for (size_t i = 1; i < publications.size(); ++i)
    {
        const geometry_msgs::Point &a =
            publications[i - 1].odometry.pose.pose.position;
        const geometry_msgs::Point &b =
            publications[i].odometry.pose.pose.position;
        const double dx = b.x - a.x;
        const double dy = b.y - a.y;
        const double dz = b.z - a.z;
        const double step = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (!std::isfinite(step) || step > kWarmupMaxPoseStep)
            stable_begin = i;
    }
    if (stable_begin > 0)
    {
        ROS_WARN("[output_warmup] trimming %zu pose(s) before last startup jump; retained start %.3f",
            stable_begin,
            publications[stable_begin].odometry.header.stamp.toSec());
        publications.erase(publications.begin(),
            publications.begin() + stable_begin);
    }
    ROS_INFO("[output_warmup] publishing %zu buffered odometry poses %.3f -> %.3f",
        publications.size(),
        publications.front().odometry.header.stamp.toSec(),
        publications.back().odometry.header.stamp.toSec());
    for (const WarmupOdometryPublication &publication : publications)
        publishOdometryMessage(publication.odometry, publication.td_ms);
    return publications.front().odometry.header.stamp.toSec();
}

void clearWarmupOdometry()
{
    std::lock_guard<std::mutex> lock(warmup_odometry_mutex);
    if (!warmup_odometry.empty())
        ROS_WARN("[output_warmup] discarded %zu buffered odometry poses after reset/anomaly",
            warmup_odometry.size());
    warmup_odometry.clear();
}

void pubKeyPoses(const Estimator &estimator, const std_msgs::Header &header)
{
    if (estimator.key_poses.size() == 0)
        return;
    visualization_msgs::Marker key_poses;
    key_poses.header = header;
    key_poses.header.frame_id = "world";
    key_poses.ns = "key_poses";
    key_poses.type = visualization_msgs::Marker::SPHERE_LIST;
    key_poses.action = visualization_msgs::Marker::ADD;
    key_poses.pose.orientation.w = 1.0;
    key_poses.lifetime = ros::Duration();

    //static int key_poses_id = 0;
    key_poses.id = 0; //key_poses_id++;
    key_poses.scale.x = 0.05;
    key_poses.scale.y = 0.05;
    key_poses.scale.z = 0.05;
    key_poses.color.r = 1.0;
    key_poses.color.a = 1.0;

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        geometry_msgs::Point pose_marker;
        Vector3d correct_pose;
        correct_pose = estimator.key_poses[i];
        pose_marker.x = correct_pose.x();
        pose_marker.y = correct_pose.y();
        pose_marker.z = correct_pose.z();
        key_poses.points.push_back(pose_marker);
    }
    pub_key_poses.publish(key_poses);
}

void pubCameraPose(const Estimator &estimator, const std_msgs::Header &header)
{
    int idx2 = WINDOW_SIZE - 1;

    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        int i = idx2;
        Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
        Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[0]);

        nav_msgs::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();

        if(STEREO)
        {
            Vector3d P_r = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[1];
            Quaterniond R_r = Quaterniond(estimator.Rs[i] * estimator.ric[1]);

            nav_msgs::Odometry odometry_r;
            odometry_r.header = header;
            odometry_r.header.frame_id = "world";
            odometry_r.pose.pose.position.x = P_r.x();
            odometry_r.pose.pose.position.y = P_r.y();
            odometry_r.pose.pose.position.z = P_r.z();
            odometry_r.pose.pose.orientation.x = R_r.x();
            odometry_r.pose.pose.orientation.y = R_r.y();
            odometry_r.pose.pose.orientation.z = R_r.z();
            odometry_r.pose.pose.orientation.w = R_r.w();
            pub_camera_pose_right.publish(odometry_r);
            if(PUB_RECTIFY)
            {
                Vector3d R_P_l = P;
                Vector3d R_P_r = P_r;
                Quaterniond R_R_l = Quaterniond(estimator.Rs[i] * estimator.ric[0] * rectify_R_left.inverse());
                Quaterniond R_R_r = Quaterniond(estimator.Rs[i] * estimator.ric[1] * rectify_R_right.inverse());
                geometry_msgs::PoseStamped R_pose_l, R_pose_r;
                R_pose_l.header = header;
                R_pose_r.header = header;
                R_pose_l.header.frame_id = "world";
                R_pose_r.header.frame_id = "world";
                R_pose_l.pose.position.x = R_P_l.x();
                R_pose_l.pose.position.y = R_P_l.y();
                R_pose_l.pose.position.z = R_P_l.z();
                R_pose_l.pose.orientation.x = R_R_l.x();
                R_pose_l.pose.orientation.y = R_R_l.y();
                R_pose_l.pose.orientation.z = R_R_l.z();
                R_pose_l.pose.orientation.w = R_R_l.w();

                R_pose_r.pose.position.x = R_P_r.x();
                R_pose_r.pose.position.y = R_P_r.y();
                R_pose_r.pose.position.z = R_P_r.z();
                R_pose_r.pose.orientation.x = R_R_r.x();
                R_pose_r.pose.orientation.y = R_R_r.y();
                R_pose_r.pose.orientation.z = R_R_r.z();
                R_pose_r.pose.orientation.w = R_R_r.w();

                pub_rectify_pose_left.publish(R_pose_l);
                pub_rectify_pose_right.publish(R_pose_r);

            }
        }

        pub_camera_pose.publish(odometry);

        cameraposevisual.reset();
        cameraposevisual.add_pose(P, R);
        if(STEREO)
        {
            Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[1];
            Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[1]);
            cameraposevisual.add_pose(P, R);
        }
        cameraposevisual.publish_by(pub_camera_pose_visual, odometry.header);
    }
}


void pubPointCloud(const Estimator &estimator, const std_msgs::Header &header)
{
    sensor_msgs::PointCloud point_cloud, loop_point_cloud;
    point_cloud.header = header;
    loop_point_cloud.header = header;


    for (auto _it : estimator.f_manager.feature)
    {
        auto it_per_id = _it.second;
        int used_num;
        used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        if (it_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id.solve_flag != 1)
            continue;
        int imu_i = it_per_id.start_frame;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
        Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[it_per_id.main_cam] * pts_i + estimator.tic[it_per_id.main_cam]) + estimator.Ps[imu_i];

        geometry_msgs::Point32 p;
        p.x = w_pts_i(0);
        p.y = w_pts_i(1);
        p.z = w_pts_i(2);
        point_cloud.points.push_back(p);
    }
    pub_point_cloud.publish(point_cloud);


    // pub margined potin
    sensor_msgs::PointCloud margin_cloud;
    margin_cloud.header = header;

    for (auto &_it : estimator.f_manager.feature)
    {
        auto & it_per_id = _it.second;
        int used_num;
        used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        //if (it_per_id->start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id->solve_flag != 1)
        //        continue;

        if (it_per_id.start_frame == 0 && it_per_id.feature_per_frame.size() <= 2 
            && it_per_id.solve_flag == 1 )
        {
            int imu_i = it_per_id.start_frame;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
            Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[it_per_id.main_cam] * pts_i + estimator.tic[it_per_id.main_cam]) + estimator.Ps[imu_i];

            geometry_msgs::Point32 p;
            p.x = w_pts_i(0);
            p.y = w_pts_i(1);
            p.z = w_pts_i(2);
            margin_cloud.points.push_back(p);
        }
    }
    pub_margin_cloud.publish(margin_cloud);
}


void pubTF(const Estimator &estimator, const std_msgs::Header &header)
{
    if( estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    // body frame
    Vector3d correct_t;
    Quaterniond correct_q;
    correct_t = estimator.Ps[WINDOW_SIZE];
    correct_q = estimator.Rs[WINDOW_SIZE];

    transform.setOrigin(tf::Vector3(correct_t(0),
                                    correct_t(1),
                                    correct_t(2)));
    q.setW(correct_q.w());
    q.setX(correct_q.x());
    q.setY(correct_q.y());
    q.setZ(correct_q.z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "world", "body"));

    // camera frame
    transform.setOrigin(tf::Vector3(estimator.tic[0].x(),
                                    estimator.tic[0].y(),
                                    estimator.tic[0].z()));
    q.setW(Quaterniond(estimator.ric[0]).w());
    q.setX(Quaterniond(estimator.ric[0]).x());
    q.setY(Quaterniond(estimator.ric[0]).y());
    q.setZ(Quaterniond(estimator.ric[0]).z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "body", "camera"));

    
    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = estimator.tic[0].x();
    odometry.pose.pose.position.y = estimator.tic[0].y();
    odometry.pose.pose.position.z = estimator.tic[0].z();
    Quaterniond tmp_q{estimator.ric[0]};
    odometry.pose.pose.orientation.x = tmp_q.x();
    odometry.pose.pose.orientation.y = tmp_q.y();
    odometry.pose.pose.orientation.z = tmp_q.z();
    odometry.pose.pose.orientation.w = tmp_q.w();
    pub_extrinsic.publish(odometry);

}

namespace
{
bool buildKeyframePublication(const Estimator &estimator, KeyframePublication &publication)
{
    if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR ||
        estimator.marginalization_flag != 0)
        return false;

    const int keyframe_index = WINDOW_SIZE - 2;
    const Vector3d P = estimator.Ps[keyframe_index];
    const Quaterniond R(estimator.Rs[keyframe_index]);
    nav_msgs::Odometry &odometry = publication.pose;
    sensor_msgs::PointCloud &point_cloud = publication.points;
    fishloop_vins::VIOKeyframe &vkf = publication.vio_keyframe;

    odometry.header.stamp = ros::Time(estimator.Headers[keyframe_index]);
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = P.x();
    odometry.pose.pose.position.y = P.y();
    odometry.pose.pose.position.z = P.z();
    odometry.pose.pose.orientation.x = R.x();
    odometry.pose.pose.orientation.y = R.y();
    odometry.pose.pose.orientation.z = R.z();
    odometry.pose.pose.orientation.w = R.w();

    const Vector3d P_r = P + R * estimator.tic[0];
    Quaterniond R_r(R * estimator.ric[0]);
    R_r.normalize();
    vkf.pose_cam.position.x = P_r.x();
    vkf.pose_cam.position.y = P_r.y();
    vkf.pose_cam.position.z = P_r.z();
    vkf.pose_cam.orientation.x = R_r.x();
    vkf.pose_cam.orientation.y = R_r.y();
    vkf.pose_cam.orientation.z = R_r.z();
    vkf.pose_cam.orientation.w = R_r.w();

    vkf.camera_extrisinc.position.x = estimator.tic[0].x();
    vkf.camera_extrisinc.position.y = estimator.tic[0].y();
    vkf.camera_extrisinc.position.z = estimator.tic[0].z();
    Quaterniond ric(estimator.ric[0]);
    ric.normalize();
    vkf.camera_extrisinc.orientation.x = ric.x();
    vkf.camera_extrisinc.orientation.y = ric.y();
    vkf.camera_extrisinc.orientation.z = ric.z();
    vkf.camera_extrisinc.orientation.w = ric.w();
    vkf.pose_drone = odometry.pose.pose;
    vkf.header.stamp = odometry.header.stamp;

    point_cloud.header.stamp = odometry.header.stamp;
    point_cloud.header.frame_id = "world";
    for (const auto &_it : estimator.f_manager.feature)
    {
        const auto &it_per_id = _it.second;
        const int frame_size = it_per_id.feature_per_frame.size();
        if (it_per_id.start_frame >= keyframe_index ||
            it_per_id.start_frame + frame_size - 1 < keyframe_index ||
            it_per_id.solve_flag >= 2)
            continue;

        const int imu_j = keyframe_index - it_per_id.start_frame;
        const int imu_i = it_per_id.start_frame;
        const Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
        const Vector3d w_pts_i =
            estimator.Rs[imu_i] *
                (estimator.ric[it_per_id.main_cam] * pts_i + estimator.tic[it_per_id.main_cam]) +
            estimator.Ps[imu_i];
        const FeaturePerFrame &frame_observation = it_per_id.feature_per_frame[imu_j];
        auto append_observation = [&](int camera_id, const Vector3d &bearing)
        {
            const camodocal::CameraPtr loop_camera = loopFusionCamera(camera_id);
            if (!loop_camera || !bearing.allFinite() || bearing.norm() < 1e-8)
                return;

            Eigen::Vector2d raw_pixel;
            loop_camera->spaceToPlane(bearing, raw_pixel);
            if (!raw_pixel.allFinite() || raw_pixel.x() < 0.0 ||
                raw_pixel.x() >= loop_camera->imageWidth() || raw_pixel.y() < 0.0 ||
                raw_pixel.y() >= loop_camera->imageHeight())
                return;

            const double safe_z = fabs(bearing.z()) > 1e-8 ? bearing.z() :
                (bearing.z() >= 0.0 ? 1e-8 : -1e-8);
            geometry_msgs::Point32 p;
            p.x = w_pts_i.x();
            p.y = w_pts_i.y();
            p.z = w_pts_i.z();
            point_cloud.points.push_back(p);
            vkf.feature_points_3d.push_back(p);

            sensor_msgs::ChannelFloat32 p_2d;
            p_2d.values.push_back(bearing.x() / safe_z);
            p_2d.values.push_back(bearing.y() / safe_z);
            p_2d.values.push_back(raw_pixel.x());
            p_2d.values.push_back(raw_pixel.y());
            p_2d.values.push_back(it_per_id.feature_id);
            p_2d.values.push_back(camera_id);
            p_2d.values.push_back(bearing.x());
            p_2d.values.push_back(bearing.y());
            p_2d.values.push_back(bearing.z());
            point_cloud.channels.push_back(p_2d);

            geometry_msgs::Point32 fp2d_uv;
            fp2d_uv.x = raw_pixel.x();
            fp2d_uv.y = raw_pixel.y();
            fp2d_uv.z = camera_id;
            geometry_msgs::Point32 fp2d_norm;
            fp2d_norm.x = bearing.x() / safe_z;
            fp2d_norm.y = bearing.y() / safe_z;
            fp2d_norm.z = bearing.z();
            vkf.feature_points_id.push_back(it_per_id.feature_id);
            vkf.feature_points_2d_uv.push_back(fp2d_uv);
            vkf.feature_points_2d_norm.push_back(fp2d_norm);
            vkf.feature_points_flag.push_back(it_per_id.solve_flag);

            fishloop_vins::LoopFeature loop_feature;
            loop_feature.feature_id = it_per_id.feature_id;
            loop_feature.camera_id = camera_id;
            loop_feature.view_id = -1;
            loop_feature.point_w = p;
            loop_feature.bearing.x = bearing.x();
            loop_feature.bearing.y = bearing.y();
            loop_feature.bearing.z = bearing.z();
            loop_feature.raw_uv = fp2d_uv;
            loop_feature.virtual_uv.x = -1.0f;
            loop_feature.virtual_uv.y = -1.0f;
            loop_feature.virtual_uv.z = 0.0f;
            vkf.loop_features.push_back(loop_feature);
        };

        append_observation(it_per_id.main_cam, frame_observation.point);
        if (frame_observation.is_stereo && it_per_id.main_cam != 1)
            append_observation(1, frame_observation.pointRight);
    }

    if (pub_keyframe_image.getNumSubscribers() > 0)
    {
        cv::Mat kf_img;
        if (estimator.getKeyframeImage(estimator.Headers[keyframe_index], kf_img) && !kf_img.empty())
        {
            std_msgs::Header img_header;
            img_header.stamp = odometry.header.stamp;
            img_header.frame_id = "world";
            sensor_msgs::ImagePtr img_msg =
                cv_bridge::CvImage(img_header, "mono8", kf_img).toImageMsg();
            publication.image = *img_msg;
            publication.has_image = true;
        }
    }
    return true;
}

void publishKeyframePublication(const KeyframePublication &publication)
{
    pub_keyframe_pose.publish(publication.pose);
    pub_keyframe_point.publish(publication.points);
    pub_viokeyframe.publish(publication.vio_keyframe);
    if (publication.has_image)
        pub_keyframe_image.publish(publication.image);
}
}

void bufferWarmupKeyframe(const Estimator &estimator)
{
    KeyframePublication publication;
    if (!buildKeyframePublication(estimator, publication))
        return;

    std::lock_guard<std::mutex> lock(warmup_keyframes_mutex);
    const double stamp = publication.pose.header.stamp.toSec();
    if (!warmup_keyframes.empty() &&
        fabs(warmup_keyframes.back().pose.header.stamp.toSec() - stamp) < 1e-6)
        return;
    warmup_keyframes.push_back(std::move(publication));
    while (warmup_keyframes.size() > kMaxWarmupKeyframes)
        warmup_keyframes.pop_front();
    ROS_INFO_THROTTLE(1.0,
        "[loop_warmup] buffered=%zu oldest=%.3f newest=%.3f newest_pts=%zu",
        warmup_keyframes.size(),
        warmup_keyframes.front().pose.header.stamp.toSec(),
        warmup_keyframes.back().pose.header.stamp.toSec(),
        warmup_keyframes.back().points.points.size());
}

void flushWarmupKeyframes(double min_stamp)
{
    std::deque<KeyframePublication> publications;
    {
        std::lock_guard<std::mutex> lock(warmup_keyframes_mutex);
        publications.swap(warmup_keyframes);
    }
    if (publications.empty())
        return;

    std::stable_sort(publications.begin(), publications.end(),
        [](const KeyframePublication &a, const KeyframePublication &b) {
            return a.pose.header.stamp < b.pose.header.stamp;
        });

    if (min_stamp >= 0.0)
        publications.erase(std::remove_if(publications.begin(), publications.end(),
            [min_stamp](const KeyframePublication &publication) {
                return publication.pose.header.stamp.toSec() + 1e-6 < min_stamp;
            }), publications.end());
    if (publications.empty())
        return;

    ROS_INFO("[loop_warmup] publishing %zu buffered keyframes %.3f -> %.3f",
        publications.size(),
        publications.front().pose.header.stamp.toSec(),
        publications.back().pose.header.stamp.toSec());
    for (const KeyframePublication &publication : publications)
        publishKeyframePublication(publication);
}

void clearWarmupKeyframes()
{
    std::lock_guard<std::mutex> lock(warmup_keyframes_mutex);
    if (!warmup_keyframes.empty())
        ROS_WARN("[loop_warmup] discarded %zu buffered keyframes after estimator reset/anomaly",
            warmup_keyframes.size());
    warmup_keyframes.clear();
}

void resetVioPath()
{
    std::lock_guard<std::mutex> lock(vio_path_mutex);
    path = nav_msgs::Path();
    pub_path.publish(path);
    pub_estimator_reset.publish(std_msgs::Empty());
    sum_of_path = 0.0;
    last_path.setZero();
    ROS_INFO("[vio_path] cleared after estimator reset");
}

void pubKeyframe(const Estimator &estimator)
{
    KeyframePublication publication;
    if (buildKeyframePublication(estimator, publication))
        publishKeyframePublication(publication);
}
