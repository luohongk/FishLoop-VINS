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

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "parameters.h"
#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"

#define MIN_LOOP_NUM 25

using namespace Eigen;
using namespace std;
using namespace DVision;


class BriefExtractor
{
public:
  virtual void operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const;
  BriefExtractor(const std::string &pattern_file);

  DVision::BRIEF m_brief;
};

struct BriefMatchStats
{
	int query_count = 0;
	int train_count = 0;
	int distance_pass = 0;
	int ratio_pass = 0;
	int unique_train_after_ratio = 0;
	int mutual_unique_pass = 0;
	double mean_best_distance = 0.0;
};

struct OrbMatchStats
{
	int query_count = 0;
	int train_count = 0;
	int distance_pass = 0;
	int ratio_pass = 0;
	int geometric_pass = 0;
	int mutual_unique_pass = 0;
	double mean_best_distance = 0.0;
};

struct LoopFeature
{
	int feature_id = -1;
	int camera_id = -1;
	int view_id = -1;
	cv::Point3f point_w;
	Eigen::Vector3d bearing = Eigen::Vector3d::Zero();
	cv::Point2f raw_uv;
	cv::Point2f virtual_uv;
	cv::Point2f normalized_uv;
	BRIEF::bitset brief_descriptor;
	cv::Mat orb_descriptor;
};

class KeyFrame
{
public:
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i,
			 cv::Mat &_image, cv::Mat &_image_right,
			 vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv,
			 vector<cv::Point2f> &_point_2d_normal, vector<Eigen::Vector3d> &_point_bearing,
			 vector<int> &_point_camera_id, vector<double> &_point_id, int _sequence);
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
			 cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1 > &_loop_info,
			 vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors);
	bool findConnection(KeyFrame* old_kf);
	void computeWindowBRIEFPoint();
	void computeBRIEFPoint();
	void computeRetrievalBRIEFPoint();
	void computeORBPoint();
	void buildCanonicalViews();
	bool projectToCanonicalView(int camera_id, const Eigen::Vector3d &bearing,
		int &view_id, cv::Point2f &virtual_uv) const;
	//void extractBrief();
	int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b);
	BriefMatchStats searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
									 std::vector<cv::Point2f> &matched_2d_old_norm,
									 std::vector<uchar> &status,
									 std::vector<int> &matched_camera_old,
									 std::vector<int> &matched_feature_old,
									 const std::vector<BRIEF::bitset> &descriptors_old,
									 const std::vector<cv::KeyPoint> &keypoints_old,
									 const std::vector<cv::KeyPoint> &keypoints_old_norm,
									 const std::vector<int> &camera_ids_old,
									 const std::vector<int> &feature_indices_old);
	OrbMatchStats searchByORBDes(std::vector<cv::Point2f> &matched_2d_old,
								 std::vector<cv::Point2f> &matched_2d_old_norm,
								 std::vector<uchar> &status,
								 std::vector<int> &matched_camera_old,
								 std::vector<int> &matched_feature_old,
								 const KeyFrame *old_kf,
								 bool dense_old,
								 int distance_threshold,
								 double ratio_threshold) const;
	void FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                const std::vector<cv::Point2f> &matched_2d_old_norm,
                                vector<uchar> &status);
	bool PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
	               const std::vector<cv::Point3f> &matched_3d,
	               std::vector<uchar> &status,
	               Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old,
	               int camera_id);
	void getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info);

	Eigen::Vector3d getLoopRelativeT();
	double getLoopRelativeYaw();
	Eigen::Quaterniond getLoopRelativeQ();



	double time_stamp; 
	int index;
	int local_index;
	Eigen::Vector3d vio_T_w_i; 
	Eigen::Matrix3d vio_R_w_i; 
	Eigen::Vector3d T_w_i;
	Eigen::Matrix3d R_w_i;
	Eigen::Vector3d origin_vio_T;		
	Eigen::Matrix3d origin_vio_R;
	cv::Mat image;
	cv::Mat image_right;
	cv::Mat thumbnail;
	vector<cv::Point3f> point_3d; 
	vector<cv::Point2f> point_2d_uv;
	vector<cv::Point2f> point_2d_norm;
	vector<double> point_id;
	vector<LoopFeature> loop_features;
	vector<vector<int>> camera_feature_indices;
	vector<vector<int>> view_feature_indices;
	vector<vector<cv::Mat>> canonical_views;
	vector<cv::KeyPoint> keypoints;
	vector<cv::KeyPoint> keypoints_norm;
	vector<cv::KeyPoint> window_keypoints;
	vector<cv::KeyPoint> window_keypoints_norm;
	vector<BRIEF::bitset> brief_descriptors;
	vector<vector<cv::KeyPoint>> camera_keypoints;
	vector<vector<cv::KeyPoint>> camera_keypoints_norm;
	vector<vector<BRIEF::bitset>> camera_brief_descriptors;
	vector<BRIEF::bitset> retrieval_brief_descriptors;
	vector<vector<BRIEF::bitset>> retrieval_view_descriptors;
	vector<BRIEF::bitset> window_brief_descriptors;
	vector<cv::KeyPoint> orb_keypoints;
	vector<cv::KeyPoint> orb_keypoints_raw;
	vector<cv::KeyPoint> orb_keypoints_norm;
	vector<int> orb_keypoint_camera_ids;
	vector<int> orb_keypoint_view_ids;
	cv::Mat orb_descriptors;
	vector<cv::KeyPoint> window_orb_keypoints;
	cv::Mat window_orb_descriptors;
	vector<int> window_orb_point_indices;
	bool has_fast_point;
	int sequence;

	bool has_loop;
	int loop_index;
	Eigen::Matrix<double, 8, 1 > loop_info;
};
