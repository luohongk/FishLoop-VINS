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

#include "keyframe.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <opencv2/opencv_modules.hpp>

#ifdef HAVE_OPENCV_CUDAFEATURES2D
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#endif

namespace {

bool cudaOrbMatcherAvailable()
{
#ifdef HAVE_OPENCV_CUDAFEATURES2D
	if (!LOOP_USE_GPU)
		return false;
	static const bool available = []()
	{
		try
		{
			if (cv::cuda::getCudaEnabledDeviceCount() <= 0)
				return false;
			cv::cuda::DeviceInfo device;
			if (!device.isCompatible())
			{
				ROS_WARN("[loop_fusion][CUDA] device %s compute=%d.%d is not "
					"supported by this OpenCV CUDA build; using CPU matcher",
					device.name(), device.majorVersion(), device.minorVersion());
				return false;
			}
			return true;
		}
		catch (const cv::Exception &error)
		{
			ROS_WARN("[loop_fusion][CUDA] availability check failed; using CPU "
				"matcher: %s", error.what());
			return false;
		}
	}();
	return available;
#else
	return false;
#endif
}

bool cudaOrbKnnMatch(const cv::Mat &query_descriptors,
	const cv::Mat &train_descriptors,
	std::vector<std::vector<cv::DMatch>> &forward_matches,
	std::vector<std::vector<cv::DMatch>> &reverse_matches)
{
#ifdef HAVE_OPENCV_CUDAFEATURES2D
	if (!cudaOrbMatcherAvailable())
		return false;
	try
	{
		cv::cuda::GpuMat query_gpu;
		cv::cuda::GpuMat train_gpu;
		query_gpu.upload(query_descriptors);
		train_gpu.upload(train_descriptors);
		static thread_local cv::Ptr<cv::cuda::DescriptorMatcher> matcher =
			cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
		matcher->knnMatch(query_gpu, train_gpu, forward_matches, 2);
		matcher->knnMatch(train_gpu, query_gpu, reverse_matches, 1);
		return true;
	}
	catch (const cv::Exception &error)
	{
		ROS_WARN_THROTTLE(5.0,
			"[loop_fusion][CUDA] ORB matcher failed; falling back to CPU: %s",
			error.what());
	}
#endif
	return false;
}

} // namespace

template <typename Derived>
static void reduceVector(vector<Derived> &v, const vector<uchar> &status)
{
    int j = 0;
	const int count = std::min<int>(v.size(), status.size());
	for (int i = 0; i < count; i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

static float canonicalPatchAngle(const cv::Mat &image, const cv::Point2f &point)
{
	const int radius = 15;
	const int cx = cvRound(point.x);
	const int cy = cvRound(point.y);
	if (image.empty() || cx - radius < 0 || cy - radius < 0 ||
		cx + radius >= image.cols || cy + radius >= image.rows)
		return 0.f;
	long m10 = 0;
	long m01 = 0;
	for (int y = -radius; y <= radius; ++y)
	{
		const int half_width = cvFloor(std::sqrt((double)radius * radius - y * y));
		const uchar *row = image.ptr<uchar>(cy + y);
		for (int x = -half_width; x <= half_width; ++x)
		{
			const int intensity = row[cx + x];
			m10 += x * intensity;
			m01 += y * intensity;
		}
	}
	return cv::fastAtan2((float)m01, (float)m10);
}

static cv::Mat loopDebugBgr(const cv::Mat &image)
{
	if (image.empty())
		return cv::Mat();
	cv::Mat image_8u;
	if (image.depth() == CV_8U)
		image_8u = image;
	else
		image.convertTo(image_8u, CV_8U);
	cv::Mat bgr;
	if (image_8u.channels() == 1)
		cv::cvtColor(image_8u, bgr, cv::COLOR_GRAY2BGR);
	else if (image_8u.channels() == 3)
		bgr = image_8u.clone();
	else if (image_8u.channels() == 4)
		cv::cvtColor(image_8u, bgr, cv::COLOR_BGRA2BGR);
	return bgr;
}

static bool encodeLoopDebugImage(const cv::Mat &image, vector<uchar> &encoded,
	int keyframe_index, int camera_id)
{
	encoded.clear();
	if (image.empty())
		return false;
	try
	{
		const vector<int> parameters = {cv::IMWRITE_JPEG_QUALITY, 80};
		if (cv::imencode(".jpg", image, encoded, parameters))
			return true;
	}
	catch (const cv::Exception &error)
	{
		ROS_WARN("[loop_fusion] failed to compress debug image kf=%d cam=%d: %s",
			keyframe_index, camera_id, error.what());
	}
	encoded.clear();
	return false;
}

static void publishLoopMatchImage(
	const KeyFrame &current_kf, const KeyFrame &old_kf,
	const vector<cv::Point2f> &matched_2d_cur,
	const vector<cv::Point2f> &matched_2d_old,
	const vector<int> &matched_camera_cur,
	const vector<int> &matched_camera_old,
	const vector<int> &inlier_match_indices)
{
	if (!DEBUG_IMAGE || inlier_match_indices.empty())
		return;
	if (pub_match_img.getNumSubscribers() == 0)
		return;

	const int camera_gap = 8;
	const int match_gap = 16;
	const int header_height = 48;
	vector<cv::Mat> current_images = {
		loopDebugBgr(current_kf.getDebugImage(0)),
		loopDebugBgr(current_kf.getDebugImage(1))};
	vector<cv::Mat> old_images = {
		loopDebugBgr(old_kf.getDebugImage(0)),
		loopDebugBgr(old_kf.getDebugImage(1))};
	vector<cv::Point> current_origins(2, cv::Point(-1, -1));
	vector<cv::Point> old_origins(2, cv::Point(-1, -1));

	auto sideSize = [camera_gap](const vector<cv::Mat> &images)
	{
		cv::Size size(0, 0);
		for (const cv::Mat &image : images)
		{
			if (image.empty())
				continue;
			size.width = std::max(size.width, image.cols);
			size.height += image.rows;
		}
		if (!images[0].empty() && !images[1].empty())
			size.height += camera_gap;
		return size;
	};
	const cv::Size current_size = sideSize(current_images);
	const cv::Size old_size = sideSize(old_images);
	if (current_size.width == 0 || old_size.width == 0)
	{
		ROS_WARN_THROTTLE(5.0,
			"[loop_fusion] cannot publish match image: retained keyframe image is empty");
		return;
	}

	const int canvas_height = header_height +
		std::max(current_size.height, old_size.height);
	const int old_side_x = current_size.width + match_gap;
	cv::Mat canvas(canvas_height, current_size.width + match_gap + old_size.width,
		CV_8UC3, cv::Scalar(32, 32, 32));
	auto copySide = [header_height, camera_gap](const vector<cv::Mat> &images,
		int side_x, vector<cv::Point> &origins, cv::Mat &canvas_image)
	{
		int y = header_height;
		for (int camera_id = 0; camera_id < (int)images.size(); ++camera_id)
		{
			if (images[camera_id].empty())
				continue;
			origins[camera_id] = cv::Point(side_x, y);
			images[camera_id].copyTo(canvas_image(
				cv::Rect(side_x, y, images[camera_id].cols, images[camera_id].rows)));
			cv::putText(canvas_image, "cam" + to_string(camera_id),
				cv::Point(side_x + 8, y + 24), cv::FONT_HERSHEY_SIMPLEX,
				0.65, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
			y += images[camera_id].rows + camera_gap;
		}
	};
	copySide(current_images, 0, current_origins, canvas);
	copySide(old_images, old_side_x, old_origins, canvas);

	cv::putText(canvas,
		"current: " + to_string(current_kf.index) + "  seq: " +
			to_string(current_kf.sequence),
		cv::Point(12, 31), cv::FONT_HERSHEY_SIMPLEX, 0.75,
		cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
	cv::putText(canvas,
		"previous: " + to_string(old_kf.index) + "  seq: " +
			to_string(old_kf.sequence),
		cv::Point(old_side_x + 12, 31), cv::FONT_HERSHEY_SIMPLEX, 0.75,
		cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

	int drawn_inliers = 0;
	for (int match_index : inlier_match_indices)
	{
		if (match_index < 0 || match_index >= (int)matched_2d_cur.size() ||
			match_index >= (int)matched_2d_old.size() ||
			match_index >= (int)matched_camera_cur.size() ||
			match_index >= (int)matched_camera_old.size())
			continue;
		const int current_camera = matched_camera_cur[match_index];
		const int old_camera = matched_camera_old[match_index];
		if (current_camera < 0 || current_camera >= (int)current_origins.size() ||
			old_camera < 0 || old_camera >= (int)old_origins.size() ||
			current_origins[current_camera].x < 0 || old_origins[old_camera].x < 0)
			continue;
		const cv::Point2f current_point = matched_2d_cur[match_index] +
			cv::Point2f(current_origins[current_camera]);
		const cv::Point2f old_point = matched_2d_old[match_index] +
			cv::Point2f(old_origins[old_camera]);
		cv::line(canvas, current_point, old_point, cv::Scalar(0, 220, 0),
			1, cv::LINE_AA);
		cv::circle(canvas, current_point, 4, cv::Scalar(0, 255, 0), 2,
			cv::LINE_AA);
		cv::circle(canvas, old_point, 4, cv::Scalar(0, 255, 0), 2,
			cv::LINE_AA);
		drawn_inliers++;
	}
	if (drawn_inliers == 0)
		return;

	const double max_display_width = 1600.0;
	if (canvas.cols > max_display_width)
	{
		const double scale = max_display_width / canvas.cols;
		cv::resize(canvas, canvas, cv::Size(), scale, scale, cv::INTER_AREA);
	}
	sensor_msgs::ImagePtr msg = cv_bridge::CvImage(
		std_msgs::Header(), "bgr8", canvas).toImageMsg();
	msg->header.stamp = ros::Time(current_kf.time_stamp);
	pub_match_img.publish(msg);
	ROS_INFO("[loop_fusion] published accepted match image cur=%d old=%d inliers=%d",
		current_kf.index, old_kf.index, drawn_inliers);
}

// create keyframe online
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i,
			   cv::Mat &_image, cv::Mat &_image_right,
		           vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv,
		           vector<cv::Point2f> &_point_2d_norm, vector<Eigen::Vector3d> &_point_bearing,
		           vector<int> &_point_camera_id, vector<double> &_point_id, int _sequence)
{
	time_stamp = _time_stamp;
	index = _index;
	vio_T_w_i = _vio_T_w_i;
	vio_R_w_i = _vio_R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
	origin_vio_T = vio_T_w_i;		
	origin_vio_R = vio_R_w_i;
	image = _image.clone();
	image_right = _image_right.clone();
	cv::resize(image, thumbnail, cv::Size(80, 60));
	point_3d = _point_3d;
	point_2d_uv = _point_2d_uv;
	point_2d_norm = _point_2d_norm;
	point_id = _point_id;
	const int feature_count = std::min<int>(_point_3d.size(),
		std::min<int>(_point_2d_uv.size(), std::min<int>(_point_bearing.size(),
			std::min<int>(_point_camera_id.size(), _point_id.size()))));
	loop_features.reserve(feature_count);
	for (int i = 0; i < feature_count; ++i)
	{
		LoopFeature feature;
		feature.feature_id = static_cast<int>(_point_id[i]);
		feature.camera_id = _point_camera_id[i];
		feature.point_w = _point_3d[i];
		feature.bearing = _point_bearing[i].normalized();
		feature.raw_uv = _point_2d_uv[i];
		feature.normalized_uv = _point_2d_norm[i];
		loop_features.push_back(feature);
	}
	has_loop = false;
	loop_index = -1;
	has_fast_point = false;
	loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
	sequence = _sequence;
	TicToc build_stage_timer;
	buildCanonicalViews();
	const double canonical_ms = build_stage_timer.toc();
	build_stage_timer.tic();
	computeWindowBRIEFPoint();
	const double window_brief_ms = build_stage_timer.toc();
	build_stage_timer.tic();
	computeBRIEFPoint();
	const double raw_brief_ms = build_stage_timer.toc();
	build_stage_timer.tic();
	computeRetrievalBRIEFPoint();
	const double retrieval_brief_ms = build_stage_timer.toc();
	build_stage_timer.tic();
	computeORBPoint();
	const double orb_ms = build_stage_timer.toc();
	if (index % 20 == 0)
	{
		ROS_INFO("[loop_fusion][perf-build] frame=%d canonical=%.2fms "
			"window_brief=%.2fms raw_brief=%.2fms retrieval_brief=%.2fms "
			"orb=%.2fms total=%.2fms",
			index, canonical_ms, window_brief_ms, raw_brief_ms,
			retrieval_brief_ms, orb_ms,
			canonical_ms + window_brief_ms + raw_brief_ms +
				retrieval_brief_ms + orb_ms);
	}
	if (index % 50 == 0)
	{
		vector<cv::Point2f> self_matched_uv;
		vector<cv::Point2f> self_matched_norm;
		vector<uchar> self_status;
		vector<int> self_camera_ids;
		vector<int> self_feature_indices;
		vector<int> feature_cameras;
		vector<int> feature_indices;
		for (int i = 0; i < (int)loop_features.size(); ++i)
		{
			feature_cameras.push_back(loop_features[i].camera_id);
			feature_indices.push_back(i);
		}
		const BriefMatchStats self_stats = searchByBRIEFDes(
			self_matched_uv, self_matched_norm, self_status, self_camera_ids,
			self_feature_indices, window_brief_descriptors, window_keypoints,
			window_keypoints_norm, feature_cameras, feature_indices);
		ROS_INFO("[loop_fusion][BRIEF-self] frame=%d query=%d train=%d dist=%d ratio=%d "
			"ratio_unique=%d mutual_unique=%d mean_best=%.1f",
			index, self_stats.query_count, self_stats.train_count,
			self_stats.distance_pass, self_stats.ratio_pass,
			self_stats.unique_train_after_ratio, self_stats.mutual_unique_pass,
			self_stats.mean_best_distance);
		if (!window_orb_descriptors.empty())
		{
			vector<cv::Point2f> orb_self_uv;
			vector<cv::Point2f> orb_self_norm;
			vector<uchar> orb_self_status;
			vector<int> orb_self_cameras;
			vector<int> orb_self_features;
			const OrbMatchStats orb_self_stats = searchByORBDes(
				orb_self_uv, orb_self_norm, orb_self_status, orb_self_cameras,
				orb_self_features, this, false, LOOP_ORB_DIST_TH,
				LOOP_ORB_RATIO_TH);
			ROS_INFO("[loop_fusion][ORB-self] frame=%d query=%d train=%d dist=%d "
				"ratio=%d mutual_unique=%d mean_best=%.1f",
				index, orb_self_stats.query_count, orb_self_stats.train_count,
				orb_self_stats.distance_pass, orb_self_stats.ratio_pass,
				orb_self_stats.mutual_unique_pass,
				orb_self_stats.mean_best_distance);
			if (!orb_descriptors.empty())
			{
				const OrbMatchStats dense_self_stats = searchByORBDes(
					orb_self_uv, orb_self_norm, orb_self_status, orb_self_cameras,
					orb_self_features, this, true,
					LOOP_ORB_EXACT_DENSE_DIST_TH,
					LOOP_ORB_EXACT_DENSE_RATIO_TH);
				ROS_INFO("[loop_fusion][ORB-self-dense] frame=%d query=%d train=%d "
					"dist=%d ratio=%d mutual_unique=%d mean_best=%.1f",
					index, dense_self_stats.query_count, dense_self_stats.train_count,
					dense_self_stats.distance_pass, dense_self_stats.ratio_pass,
					dense_self_stats.mutual_unique_pass,
					dense_self_stats.mean_best_distance);
				for (int camera_id = 0; camera_id < (int)m_cameras.size(); ++camera_id)
				{
					vector<cv::Point2f> self_2d;
					vector<cv::Point3f> self_3d;
					const int count = std::min<int>(point_3d.size(),
						std::min<int>(orb_self_status.size(), orb_self_cameras.size()));
					for (int i = 0; i < count; ++i)
					{
						if (!orb_self_status[i] || orb_self_cameras[i] != camera_id ||
							!std::isfinite(orb_self_norm[i].x) ||
							!std::isfinite(orb_self_norm[i].y) ||
							!std::isfinite(point_3d[i].x) ||
							!std::isfinite(point_3d[i].y) ||
							!std::isfinite(point_3d[i].z))
							continue;
						self_2d.push_back(orb_self_norm[i]);
						self_3d.push_back(point_3d[i]);
					}
					vector<uchar> self_pnp_status;
					Eigen::Vector3d self_T;
					Eigen::Matrix3d self_R;
					const bool solved = PnPRANSAC(self_2d, self_3d, self_pnp_status,
						self_T, self_R, camera_id);
					const int inliers = std::count(self_pnp_status.begin(),
						self_pnp_status.end(), (uchar)1);
					const double rotation_error = solved
						? Eigen::AngleAxisd(origin_vio_R.transpose() * self_R).angle() *
							180.0 / M_PI
						: -1.0;
					const double translation_error = solved
						? (origin_vio_T - self_T).norm() : -1.0;
					ROS_INFO("[loop_fusion][PnP-self-dense] frame=%d cam=%d matches=%zu "
						"solved=%d inliers=%d dR=%.3fdeg dT=%.4fm",
						index, camera_id, self_3d.size(), solved ? 1 : 0, inliers,
						rotation_error, translation_error);
				}
			}
		}

		// This bypasses descriptors completely. If it fails, the published
		// world point, bearing, camera extrinsic, or PnP conversion is wrong.
		for (int camera_id = 0; camera_id < (int)m_cameras.size(); ++camera_id)
		{
			vector<cv::Point2f> self_2d;
			vector<cv::Point3f> self_3d;
			for (const LoopFeature &feature : loop_features)
			{
				if (feature.camera_id != camera_id || !feature.bearing.allFinite() ||
					feature.bearing.z() <= LOOP_EUCM_MIN_Z ||
					!std::isfinite(feature.point_w.x) ||
					!std::isfinite(feature.point_w.y) ||
					!std::isfinite(feature.point_w.z))
					continue;
				self_2d.emplace_back(feature.bearing.x() / feature.bearing.z(),
					feature.bearing.y() / feature.bearing.z());
				self_3d.push_back(feature.point_w);
			}
			vector<uchar> self_pnp_status;
			Eigen::Vector3d self_T;
			Eigen::Matrix3d self_R;
			const bool solved = PnPRANSAC(self_2d, self_3d, self_pnp_status,
				self_T, self_R, camera_id);
			const int inliers = std::count(self_pnp_status.begin(),
				self_pnp_status.end(), (uchar)1);
			const double rotation_error = solved
				? Eigen::AngleAxisd(origin_vio_R.transpose() * self_R).angle() *
					180.0 / M_PI
				: -1.0;
			const double translation_error = solved
				? (origin_vio_T - self_T).norm() : -1.0;
			ROS_INFO("[loop_fusion][PnP-self-exact] frame=%d cam=%d matches=%zu "
				"solved=%d inliers=%d dR=%.3fdeg dT=%.4fm",
				index, camera_id, self_3d.size(), solved ? 1 : 0, inliers,
				rotation_error, translation_error);
		}
	}
	if (DEBUG_IMAGE)
	{
		encodeLoopDebugImage(image, debug_image_encoded, index, 0);
		encodeLoopDebugImage(image_right, debug_image_right_encoded, index, 1);
	}
	// The raw images and the ten canonical views are only needed while building
	// this keyframe's descriptors. Keep compressed debug copies instead of
	// retaining roughly 5.6 MB of image Mats for every keyframe.
	if (!DEBUG_IMAGE || !debug_image_encoded.empty())
		image.release();
	if (!DEBUG_IMAGE || !debug_image_right_encoded.empty())
		image_right.release();
	canonical_views.clear();
}

// load previous keyframe
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
					cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1 > &_loop_info,
					vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors)
{
	time_stamp = _time_stamp;
	index = _index;
	//vio_T_w_i = _vio_T_w_i;
	//vio_R_w_i = _vio_R_w_i;
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = _T_w_i;
	R_w_i = _R_w_i;
	if (DEBUG_IMAGE)
	{
		image = _image.clone();
		if (!image.empty())
		{
			cv::resize(image, thumbnail, cv::Size(80, 60));
			if (encodeLoopDebugImage(image, debug_image_encoded, index, 0))
				image.release();
		}
	}
	if (_loop_index != -1)
		has_loop = true;
	else
		has_loop = false;
	loop_index = _loop_index;
	loop_info = _loop_info;
	has_fast_point = false;
	sequence = 0;
	keypoints = _keypoints;
	keypoints_norm = _keypoints_norm;
	brief_descriptors = _brief_descriptors;
}

cv::Mat KeyFrame::getDebugImage(int camera_id) const
{
	const vector<uchar> *encoded = NULL;
	const cv::Mat *fallback = NULL;
	if (camera_id == 0)
	{
		encoded = &debug_image_encoded;
		fallback = &image;
	}
	else if (camera_id == 1)
	{
		encoded = &debug_image_right_encoded;
		fallback = &image_right;
	}
	else
		return cv::Mat();

	if (!encoded->empty())
	{
		try
		{
			return cv::imdecode(*encoded, cv::IMREAD_UNCHANGED);
		}
		catch (const cv::Exception &error)
		{
			ROS_WARN("[loop_fusion] failed to decode debug image kf=%d cam=%d: %s",
				index, camera_id, error.what());
		}
	}
	return fallback->empty() ? cv::Mat() : fallback->clone();
}


void KeyFrame::computeWindowBRIEFPoint()
{
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	const int total_views = std::max(1, (int)m_cameras.size() *
		std::max(1, (int)LOOP_VIEW_ROTATIONS.size()));
	view_feature_indices.assign(total_views, vector<int>());
	camera_feature_indices.assign(std::max<int>(2, m_cameras.size()), vector<int>());

	vector<LoopFeature> valid_features;
	valid_features.reserve(loop_features.size());
	for (LoopFeature feature : loop_features)
	{
		if (!projectToCanonicalView(feature.camera_id, feature.bearing,
			feature.view_id, feature.virtual_uv))
			continue;
		const int feature_index = valid_features.size();
		valid_features.push_back(feature);
		camera_feature_indices[feature.camera_id].push_back(feature_index);
		const int global_view = feature.camera_id * LOOP_VIEW_ROTATIONS.size() + feature.view_id;
		if (global_view >= 0 && global_view < (int)view_feature_indices.size())
			view_feature_indices[global_view].push_back(feature_index);
	}
	loop_features.swap(valid_features);

	for (int global_view = 0; global_view < (int)view_feature_indices.size(); ++global_view)
	{
		const int camera_id = global_view / std::max(1, (int)LOOP_VIEW_ROTATIONS.size());
		const int view_id = global_view % std::max(1, (int)LOOP_VIEW_ROTATIONS.size());
		if (camera_id >= (int)canonical_views.size() ||
			view_id >= (int)canonical_views[camera_id].size() ||
			canonical_views[camera_id][view_id].empty())
			continue;
		vector<cv::KeyPoint> keys;
		for (int feature_index : view_feature_indices[global_view])
			keys.emplace_back(loop_features[feature_index].virtual_uv, 31.f);
		vector<BRIEF::bitset> descriptors;
		extractor(canonical_views[camera_id][view_id], keys, descriptors);
		const int count = std::min<int>(descriptors.size(), view_feature_indices[global_view].size());
		for (int i = 0; i < count; ++i)
			loop_features[view_feature_indices[global_view][i]].brief_descriptor = descriptors[i];
	}

	point_3d.clear();
	point_2d_uv.clear();
	point_2d_norm.clear();
	point_id.clear();
	window_keypoints.clear();
	window_keypoints_norm.clear();
	window_brief_descriptors.clear();
	for (const LoopFeature &feature : loop_features)
	{
		point_3d.push_back(feature.point_w);
		point_2d_uv.push_back(feature.raw_uv);
		point_2d_norm.push_back(feature.normalized_uv);
		point_id.push_back(feature.feature_id);
		window_keypoints.emplace_back(feature.virtual_uv, 31.f);
		window_keypoints_norm.emplace_back(feature.normalized_uv, 1.f);
		window_brief_descriptors.push_back(feature.brief_descriptor);
	}
}

void KeyFrame::buildCanonicalViews()
{
	canonical_views.assign(2, vector<cv::Mat>());
	const cv::Mat raw_images[2] = {image, image_right};
	for (int camera_id = 0; camera_id < 2; ++camera_id)
	{
		if (raw_images[camera_id].empty() || camera_id >= (int)LOOP_VIEW_MAPS.size())
			continue;
		canonical_views[camera_id].resize(LOOP_VIEW_MAPS[camera_id].size());
		for (int view_id = 0; view_id < (int)LOOP_VIEW_MAPS[camera_id].size(); ++view_id)
		{
			const auto &maps = LOOP_VIEW_MAPS[camera_id][view_id];
			cv::remap(raw_images[camera_id], canonical_views[camera_id][view_id],
				maps.first, maps.second, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
		}
	}
}

bool KeyFrame::projectToCanonicalView(int camera_id, const Eigen::Vector3d &bearing,
	int &view_id, cv::Point2f &virtual_uv) const
{
	if (camera_id < 0 || camera_id >= (int)canonical_views.size() ||
		!bearing.allFinite() || bearing.norm() < 1e-8)
		return false;
	const int available_views = std::min<int>(LOOP_VIEW_ROTATIONS.size(),
		canonical_views[camera_id].size());
	double best_z = -1.0;
	int best_view = -1;
	cv::Point2f best_uv;
	const double center = 0.5 * LOOP_VIEW_SIZE;
	const double border = 26.0;
	for (int candidate_view = 0; candidate_view < available_views; ++candidate_view)
	{
		const Eigen::Vector3d point_view = LOOP_VIEW_ROTATIONS[candidate_view].transpose() * bearing;
		if (point_view.z() <= 1e-6)
			continue;
		const double u = LOOP_VIEW_FOCAL * point_view.x() / point_view.z() + center;
		const double v = LOOP_VIEW_FOCAL * point_view.y() / point_view.z() + center;
		if (u < border || u >= LOOP_VIEW_SIZE - border ||
			v < border || v >= LOOP_VIEW_SIZE - border)
			continue;
		if (point_view.z() > best_z)
		{
			best_z = point_view.z();
			best_view = candidate_view;
			best_uv = cv::Point2f(u, v);
		}
	}
	if (best_view < 0)
		return false;
	view_id = best_view;
	virtual_uv = best_uv;
	return true;
}

void KeyFrame::computeBRIEFPoint()
{
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	camera_keypoints.resize(2);
	camera_keypoints_norm.resize(2);
	camera_brief_descriptors.resize(2);
	const cv::Mat raw_images[2] = {image, image_right};
	const int camera_count = std::min<int>(2, m_cameras.size());
	for (int camera_id = 0; camera_id < camera_count; ++camera_id)
	{
		if (raw_images[camera_id].empty() || !m_cameras[camera_id])
			continue;
		vector<cv::KeyPoint> detected;
		vector<BRIEF::bitset> detected_descriptors;
		cv::FAST(raw_images[camera_id], detected, LOOP_FAST_TH, true);
		if (detected.size() > 1000)
		{
			std::nth_element(detected.begin(), detected.begin() + 1000, detected.end(),
				[](const cv::KeyPoint &a, const cv::KeyPoint &b) { return a.response > b.response; });
			detected.resize(1000);
		}
		extractor(raw_images[camera_id], detected, detected_descriptors);
		for (size_t i = 0; i < detected.size(); ++i)
		{
			Eigen::Vector3d bearing;
			m_cameras[camera_id]->liftProjective(
				Eigen::Vector2d(detected[i].pt.x, detected[i].pt.y), bearing);
			if (!bearing.allFinite() || bearing.z() <= LOOP_EUCM_MIN_Z)
				continue;
			cv::KeyPoint normalized = detected[i];
			normalized.pt = cv::Point2f(bearing.x() / bearing.z(), bearing.y() / bearing.z());
			camera_keypoints[camera_id].push_back(detected[i]);
			camera_keypoints_norm[camera_id].push_back(normalized);
			camera_brief_descriptors[camera_id].push_back(detected_descriptors[i]);
		}
	}
	if (!camera_keypoints.empty())
	{
		keypoints = camera_keypoints[0];
		keypoints_norm = camera_keypoints_norm[0];
		brief_descriptors = camera_brief_descriptors[0];
	}
}

void KeyFrame::computeRetrievalBRIEFPoint()
{
	// Use exactly the descriptors that later provide the 3D-to-2D geometry.
	// Mixing perspective FAST retrieval with VIO/GFTT geometric features made
	// even the same-frame matching largely random.
	if (LOOP_RETRIEVAL_USE_VIO_FEATURES && !window_brief_descriptors.empty())
	{
		retrieval_brief_descriptors = window_brief_descriptors;
		retrieval_view_descriptors.assign(view_feature_indices.size(), vector<BRIEF::bitset>());
		for (int global_view = 0; global_view < (int)view_feature_indices.size(); ++global_view)
			for (int feature_index : view_feature_indices[global_view])
				retrieval_view_descriptors[global_view].push_back(
					loop_features[feature_index].brief_descriptor);
		// Keep the existing pose-graph save/load format compatible: it persists
		// keypoints/keypoints_norm/brief_descriptors but has no separate window
		// fields. Store the active VIO feature representation in those slots.
		keypoints = window_keypoints;
		keypoints_norm = window_keypoints_norm;
		brief_descriptors = window_brief_descriptors;
		return;
	}

	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	const int camera_count = std::min(2, LOOP_RETRIEVAL_CAMERA_COUNT);
	retrieval_view_descriptors.assign(camera_count * LOOP_RETRIEVAL_VIEW_COUNT,
		vector<BRIEF::bitset>());
	vector<cv::KeyPoint> canonical_keypoints_raw;
	vector<cv::KeyPoint> canonical_keypoints_norm;
	vector<BRIEF::bitset> canonical_descriptors;
	vector<int> canonical_camera_ids;
	for (int camera_id = 0; camera_id < camera_count; ++camera_id)
	{
		if (camera_id >= (int)canonical_views.size())
			continue;
		if (canonical_views[camera_id].empty())
		{
			if (camera_id < (int)camera_brief_descriptors.size())
				retrieval_brief_descriptors.insert(retrieval_brief_descriptors.end(),
					camera_brief_descriptors[camera_id].begin(), camera_brief_descriptors[camera_id].end());
			continue;
		}
		const int view_count = std::min<int>(LOOP_VIEW_MAPS[camera_id].size(),
			LOOP_RETRIEVAL_VIEW_COUNT);
		for (int view_id = 0; view_id < view_count; ++view_id)
		{
			const cv::Mat &view = canonical_views[camera_id][view_id];
			if (view.empty())
				continue;
			vector<cv::KeyPoint> view_keypoints;
			cv::FAST(view, view_keypoints, LOOP_FAST_TH, true);
			view_keypoints.erase(std::remove_if(view_keypoints.begin(), view_keypoints.end(),
				[&view](const cv::KeyPoint &key)
				{
					const float border = 26.f;
					return key.pt.x < border || key.pt.y < border ||
						key.pt.x >= view.cols - border || key.pt.y >= view.rows - border;
				}), view_keypoints.end());
			if ((int)view_keypoints.size() > LOOP_VIEW_FEATURES)
			{
				std::nth_element(view_keypoints.begin(),
					view_keypoints.begin() + LOOP_VIEW_FEATURES, view_keypoints.end(),
					[](const cv::KeyPoint &a, const cv::KeyPoint &b) { return a.response > b.response; });
				view_keypoints.resize(LOOP_VIEW_FEATURES);
			}
			vector<BRIEF::bitset> view_descriptors;
			extractor(view, view_keypoints, view_descriptors);
			retrieval_view_descriptors[camera_id * LOOP_RETRIEVAL_VIEW_COUNT + view_id] =
				view_descriptors;
			retrieval_brief_descriptors.insert(retrieval_brief_descriptors.end(),
				view_descriptors.begin(), view_descriptors.end());

			// Stock VINS-Fusion matches current window descriptors against dense
			// BRIEF features from the old image. For a fisheye camera, the dense
			// image is the set of canonical perspective views; map every detected
			// feature back to its raw EUCM pixel and bearing for geometric checks.
			const int descriptor_count = std::min<int>(
				view_keypoints.size(), view_descriptors.size());
			const cv::Mat &map_x = LOOP_VIEW_MAPS[camera_id][view_id].first;
			const cv::Mat &map_y = LOOP_VIEW_MAPS[camera_id][view_id].second;
			if (map_x.type() != CV_32FC1 || map_y.type() != CV_32FC1)
				continue;
			for (int i = 0; i < descriptor_count; ++i)
			{
				const int x = cvRound(view_keypoints[i].pt.x);
				const int y = cvRound(view_keypoints[i].pt.y);
				if (x < 0 || x >= map_x.cols || y < 0 || y >= map_x.rows)
					continue;
				const float raw_x = map_x.at<float>(y, x);
				const float raw_y = map_y.at<float>(y, x);
				if (!std::isfinite(raw_x) || !std::isfinite(raw_y))
					continue;
				Eigen::Vector3d bearing;
				m_cameras[camera_id]->liftProjective(
					Eigen::Vector2d(raw_x, raw_y), bearing);
				if (!bearing.allFinite() || bearing.z() <= LOOP_EUCM_MIN_Z)
					continue;
				cv::KeyPoint raw_key = view_keypoints[i];
				raw_key.pt = cv::Point2f(raw_x, raw_y);
				cv::KeyPoint norm_key = raw_key;
				norm_key.pt = cv::Point2f(
					bearing.x() / bearing.z(), bearing.y() / bearing.z());
				canonical_keypoints_raw.push_back(raw_key);
				canonical_keypoints_norm.push_back(norm_key);
				canonical_descriptors.push_back(view_descriptors[i]);
				canonical_camera_ids.push_back(camera_id);
			}
		}
	}
	if (retrieval_brief_descriptors.empty())
		retrieval_brief_descriptors = brief_descriptors;
	if (!canonical_descriptors.empty())
	{
		retrieval_brief_descriptors = canonical_descriptors;
		keypoints.swap(canonical_keypoints_raw);
		keypoints_norm.swap(canonical_keypoints_norm);
		brief_descriptors.swap(canonical_descriptors);
		brief_keypoint_camera_ids.swap(canonical_camera_ids);
	}
}

void KeyFrame::computeORBPoint()
{
	if (!LOOP_ORB_GEOMETRY || image.empty())
		return;

	TicToc orb_stage_timer;
	cv::Ptr<cv::ORB> orb = cv::ORB::create(
		LOOP_ORB_FEATURES, 1.2f, 8, 26, 0, 2, cv::ORB::HARRIS_SCORE, 31,
		LOOP_ORB_FAST_TH);

	// Geometry descriptors are evaluated at the exact projected VIO feature
	// pixels in their canonical views.  This preserves the feature's 3D identity
	// and removes the former nearest-dense-ORB association.
	window_orb_keypoints.clear();
	window_orb_point_indices.clear();
	window_orb_descriptors.release();
	int exact_requested_keypoints = 0;
	int exact_returned_keypoints = 0;
	int exact_fallback_bindings = 0;
	for (int global_view = 0; global_view < (int)view_feature_indices.size(); ++global_view)
	{
		const int view_count = std::max(1, (int)LOOP_VIEW_ROTATIONS.size());
		const int camera_id = global_view / view_count;
		const int view_id = global_view % view_count;
		if (camera_id >= (int)canonical_views.size() ||
			view_id >= (int)canonical_views[camera_id].size() ||
			canonical_views[camera_id][view_id].empty())
			continue;
		vector<cv::KeyPoint> keys;
		vector<int> feature_indices;
		for (int feature_index : view_feature_indices[global_view])
		{
			cv::KeyPoint key(loop_features[feature_index].virtual_uv, 31.f);
			key.angle = canonicalPatchAngle(canonical_views[camera_id][view_id], key.pt);
			// ORB::compute() may remove provided keypoints near a pyramid border.
			// Carry the VIO feature index through the mutable keypoint array instead
			// of assuming descriptor row i still belongs to input row i.
			key.class_id = feature_index;
			keys.push_back(key);
			feature_indices.push_back(feature_index);
		}
		exact_requested_keypoints += keys.size();
		cv::Mat descriptors;
		orb->compute(canonical_views[camera_id][view_id], keys, descriptors);
		const int count = std::min<int>(descriptors.rows,
			std::min<int>(keys.size(), feature_indices.size()));
		exact_returned_keypoints += count;
		std::vector<uchar> feature_bound(feature_indices.size(), 0);
		for (int i = 0; i < count; ++i)
		{
			int feature_index = keys[i].class_id;
			if (feature_index < 0 || feature_index >= (int)loop_features.size() ||
				loop_features[feature_index].camera_id != camera_id ||
				loop_features[feature_index].view_id != view_id)
			{
				// Some OpenCV builds do not preserve class_id for supplied
				// keypoints. Recover the binding by the unchanged canonical pixel.
				double best_distance2 = 0.25;
				int best_local_index = -1;
				for (int local_index = 0;
					local_index < (int)feature_indices.size(); ++local_index)
				{
					if (feature_bound[local_index])
						continue;
					const int candidate_index = feature_indices[local_index];
					const cv::Point2f delta =
						loop_features[candidate_index].virtual_uv - keys[i].pt;
					const double distance2 = delta.dot(delta);
					if (distance2 < best_distance2)
					{
						best_distance2 = distance2;
						best_local_index = local_index;
					}
				}
				if (best_local_index < 0)
					continue;
				feature_index = feature_indices[best_local_index];
				exact_fallback_bindings++;
			}
			auto feature_it = std::find(feature_indices.begin(),
				feature_indices.end(), feature_index);
			if (feature_it == feature_indices.end())
				continue;
			const int local_feature_index =
				static_cast<int>(feature_it - feature_indices.begin());
			if (feature_bound[local_feature_index])
				continue;
			feature_bound[local_feature_index] = 1;
			loop_features[feature_index].orb_descriptor = descriptors.row(i).clone();
			window_orb_keypoints.push_back(keys[i]);
			window_orb_point_indices.push_back(feature_index);
			window_orb_descriptors.push_back(descriptors.row(i));
		}
	}
	const double exact_orb_ms = orb_stage_timer.toc();
	orb_stage_timer.tic();

	// Old-frame side: distribute a bounded dense ORB budget over all canonical
	// camera/view images. Each descriptor stores its exact raw EUCM pixel and
	// bearing, so an exact current VIO 3D point can match it directly without
	// any dense-to-nearest-VIO association.
	orb_keypoints.clear();
	orb_keypoints_raw.clear();
	orb_keypoints_norm.clear();
	orb_keypoint_camera_ids.clear();
	orb_keypoint_view_ids.clear();
	orb_descriptors.release();
	int available_dense_views = 0;
	for (const vector<cv::Mat> &camera_views : canonical_views)
		for (const cv::Mat &view : camera_views)
			available_dense_views += view.empty() ? 0 : 1;
	const int features_per_view = std::max(100,
		(LOOP_ORB_FEATURES + std::max(1, available_dense_views) - 1) /
			std::max(1, available_dense_views));
	cv::Ptr<cv::ORB> dense_orb = cv::ORB::create(
		features_per_view, 1.2f, 8, 26, 0, 2, cv::ORB::HARRIS_SCORE, 31,
		LOOP_ORB_FAST_TH);
	const double center = 0.5 * LOOP_VIEW_SIZE;
	const bool collect_roundtrip = index % 50 == 0;
	double remap_error_sum = 0.0;
	double remap_error_max = 0.0;
	double remap_angle_sum_deg = 0.0;
	double remap_angle_max_deg = 0.0;
	int remap_check_count = 0;
	for (int camera_id = 0; camera_id < (int)canonical_views.size(); ++camera_id)
	{
		if (camera_id >= (int)m_cameras.size() || !m_cameras[camera_id])
			continue;
		for (int view_id = 0; view_id < (int)canonical_views[camera_id].size(); ++view_id)
		{
			const cv::Mat &view = canonical_views[camera_id][view_id];
			if (view.empty() || view_id >= (int)LOOP_VIEW_ROTATIONS.size())
				continue;
			vector<cv::KeyPoint> detected_keypoints;
			cv::Mat detected_descriptors;
			dense_orb->detectAndCompute(view, cv::noArray(), detected_keypoints,
				detected_descriptors, false);
			const int count = std::min<int>(detected_descriptors.rows,
				detected_keypoints.size());
			for (int row = 0; row < count; ++row)
			{
				const cv::Point2f &virtual_uv = detected_keypoints[row].pt;
				Eigen::Vector3d point_view(
					(virtual_uv.x - center) / LOOP_VIEW_FOCAL,
					(virtual_uv.y - center) / LOOP_VIEW_FOCAL, 1.0);
				Eigen::Vector3d bearing =
					LOOP_VIEW_ROTATIONS[view_id] * point_view.normalized();
				if (!bearing.allFinite() || bearing.z() <= LOOP_EUCM_MIN_Z)
					continue;

				// Canonical views overlap. Retain the point only in the view where
				// projectToCanonicalView() would place it, preventing duplicate old
				// observations from artificially inflating PnP support.
				int canonical_view_id = -1;
				cv::Point2f canonical_uv;
				if (!projectToCanonicalView(camera_id, bearing,
					canonical_view_id, canonical_uv) || canonical_view_id != view_id)
					continue;
				const cv::Point2f projection_delta = canonical_uv - virtual_uv;
				if (projection_delta.dot(projection_delta) > 1.5 * 1.5)
					continue;

				Eigen::Vector2d raw_uv;
				m_cameras[camera_id]->spaceToPlane(bearing, raw_uv);
				if (!raw_uv.allFinite() || raw_uv.x() < 0.0 || raw_uv.y() < 0.0 ||
					raw_uv.x() >= m_cameras[camera_id]->imageWidth() ||
					raw_uv.y() >= m_cameras[camera_id]->imageHeight())
					continue;

				if (collect_roundtrip && camera_id < (int)LOOP_VIEW_MAPS.size() &&
					view_id < (int)LOOP_VIEW_MAPS[camera_id].size())
				{
					const auto &maps = LOOP_VIEW_MAPS[camera_id][view_id];
					if (!maps.first.empty() && !maps.second.empty() &&
						maps.first.type() == CV_32FC1 && maps.second.type() == CV_32FC1)
					{
						const int sample_x = std::max(0, std::min(maps.first.cols - 1,
							cvRound(virtual_uv.x)));
						const int sample_y = std::max(0, std::min(maps.first.rows - 1,
							cvRound(virtual_uv.y)));
						const Eigen::Vector2d mapped_raw(
							maps.first.at<float>(sample_y, sample_x),
							maps.second.at<float>(sample_y, sample_x));
						Eigen::Vector3d lifted;
						m_cameras[camera_id]->liftProjective(mapped_raw, lifted);
						if (mapped_raw.allFinite() && lifted.allFinite() && lifted.norm() > 1e-8)
						{
							const double raw_error = (mapped_raw - raw_uv).norm();
							const double cosine = std::max(-1.0, std::min(1.0,
								lifted.normalized().dot(bearing.normalized())));
							const double angle_deg = std::acos(cosine) * 180.0 / M_PI;
							remap_error_sum += raw_error;
							remap_error_max = std::max(remap_error_max, raw_error);
							remap_angle_sum_deg += angle_deg;
							remap_angle_max_deg = std::max(remap_angle_max_deg, angle_deg);
							remap_check_count++;
						}
					}
				}

				cv::KeyPoint raw_key = detected_keypoints[row];
				raw_key.pt = cv::Point2f(raw_uv.x(), raw_uv.y());
				cv::KeyPoint normalized_key = detected_keypoints[row];
				normalized_key.pt = cv::Point2f(
					bearing.x() / bearing.z(), bearing.y() / bearing.z());
				orb_keypoints.push_back(detected_keypoints[row]);
				orb_keypoints_raw.push_back(raw_key);
				orb_keypoints_norm.push_back(normalized_key);
				orb_keypoint_camera_ids.push_back(camera_id);
				orb_keypoint_view_ids.push_back(view_id);
				orb_descriptors.push_back(detected_descriptors.row(row));
			}
		}
	}
	const double dense_orb_ms = orb_stage_timer.toc();
	if (index % 20 == 0)
	{
		ROS_INFO("[loop_fusion][perf-orb] frame=%d exact=%.2fms dense=%.2fms "
			"exact_desc=%d dense_desc=%d",
			index, exact_orb_ms, dense_orb_ms,
			window_orb_descriptors.rows, orb_descriptors.rows);
	}

	if (index % 50 == 0)
	{
		const int dense_cam0 = std::count(orb_keypoint_camera_ids.begin(),
			orb_keypoint_camera_ids.end(), 0);
		const int dense_cam1 = std::count(orb_keypoint_camera_ids.begin(),
			orb_keypoint_camera_ids.end(), 1);
		ROS_INFO("[loop_fusion][ORB-extract] frame=%d dense_canonical=%zu "
			"cams=%d/%d exact_vio=%zu/%zu requested/returned=%d/%d "
			"fallback_bindings=%d views=%d budget=%d/view",
			index, orb_keypoints.size(), dense_cam0, dense_cam1,
			window_orb_point_indices.size(), point_2d_uv.size(),
			exact_requested_keypoints, exact_returned_keypoints,
			exact_fallback_bindings,
			available_dense_views, features_per_view);
		ROS_INFO("[loop_fusion][canonical-roundtrip] frame=%d samples=%d "
			"raw_px_mean/max=%.3f/%.3f bearing_deg_mean/max=%.5f/%.5f",
			index, remap_check_count,
			remap_check_count > 0 ? remap_error_sum / remap_check_count : -1.0,
			remap_error_max,
			remap_check_count > 0 ? remap_angle_sum_deg / remap_check_count : -1.0,
			remap_angle_max_deg);
	}
	if ((int)window_orb_point_indices.size() <= LOOP_PER_CAMERA_MIN_INLIERS)
	{
		ROS_WARN_THROTTLE(2.0,
			"[loop_fusion][ORB-extract] only %zu VIO points associated; need >%d before PnP",
			window_orb_point_indices.size(), LOOP_PER_CAMERA_MIN_INLIERS);
	}
}

void BriefExtractor::operator() (const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const
{
  m_brief.compute(im, keys, descriptors);
}


BriefMatchStats KeyFrame::searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
										   std::vector<cv::Point2f> &matched_2d_old_norm,
										   std::vector<uchar> &status,
										   std::vector<int> &matched_camera_old,
										   std::vector<int> &matched_feature_old,
										   const std::vector<BRIEF::bitset> &descriptors_old,
										   const std::vector<cv::KeyPoint> &keypoints_old,
										   const std::vector<cv::KeyPoint> &keypoints_old_norm,
										   const std::vector<int> &camera_ids_old,
										   const std::vector<int> &feature_indices_old)
{
	BriefMatchStats stats;
	const int output_count = static_cast<int>(point_2d_uv.size());
	stats.query_count = std::min<int>(window_brief_descriptors.size(), output_count);
	stats.train_count = std::min<int>(descriptors_old.size(),
		std::min<int>(keypoints_old.size(), keypoints_old_norm.size()));

	// Keep these vectors aligned with the current VIO 3D/2D feature arrays.
	// reduceVector() later applies this exact status mask to every array.
	matched_2d_old.assign(output_count, cv::Point2f(0.f, 0.f));
	matched_2d_old_norm.assign(output_count, cv::Point2f(0.f, 0.f));
	status.assign(output_count, 0);
	matched_camera_old.assign(output_count, -1);
	matched_feature_old.assign(output_count, -1);

	if (stats.query_count == 0 || stats.train_count < 2)
		return stats;

	if (LOOP_PRESERVE_ORIGINAL_FLOW)
	{
		double best_distance_sum = 0.0;
		int best_distance_count = 0;
		for (int query_index = 0; query_index < stats.query_count; ++query_index)
		{
			if (query_index >= (int)loop_features.size())
				continue;
			const int current_camera = loop_features[query_index].camera_id;
			int best_index = -1;
			int best_distance = 128;
			for (int train_index = 0; train_index < stats.train_count; ++train_index)
			{
				if (train_index >= (int)camera_ids_old.size() ||
					camera_ids_old[train_index] != current_camera)
					continue;
				const int distance = HammingDis(
					window_brief_descriptors[query_index],
					descriptors_old[train_index]);
				if (distance < best_distance)
				{
					best_distance = distance;
					best_index = train_index;
				}
			}
			if (best_index < 0)
				continue;
			best_distance_sum += best_distance;
			best_distance_count++;
			if (best_distance >= LOOP_BRIEF_DIST_TH)
				continue;
			status[query_index] = 1;
			matched_2d_old[query_index] = keypoints_old[best_index].pt;
			matched_2d_old_norm[query_index] = keypoints_old_norm[best_index].pt;
			matched_camera_old[query_index] = camera_ids_old[best_index];
			if (best_index < (int)feature_indices_old.size())
				matched_feature_old[query_index] = feature_indices_old[best_index];
			stats.distance_pass++;
			stats.ratio_pass++;
			stats.mutual_unique_pass++;
		}
		if (best_distance_count > 0)
			stats.mean_best_distance = best_distance_sum / best_distance_count;
		stats.unique_train_after_ratio = stats.distance_pass;
		return stats;
	}

	struct Candidate
	{
		int query_index;
		int train_index;
		int best_distance;
		int second_distance;
	};

	std::vector<Candidate> ratio_candidates;
	ratio_candidates.reserve(stats.query_count);
	std::vector<int> reverse_best_query(stats.train_count, -1);
	std::vector<int> reverse_best_distance(stats.train_count, std::numeric_limits<int>::max());
	std::vector<uchar> ratio_train_used(stats.train_count, 0);
	double best_distance_sum = 0.0;
	int best_distance_count = 0;

	for (int query_index = 0; query_index < stats.query_count; ++query_index)
	{
		if (query_index >= (int)loop_features.size())
			continue;
		const int current_camera = loop_features[query_index].camera_id;
		int best_index = -1;
		int best_distance = std::numeric_limits<int>::max();
		int second_distance = std::numeric_limits<int>::max();

		for (int train_index = 0; train_index < stats.train_count; ++train_index)
		{
			if (train_index >= (int)camera_ids_old.size() ||
				camera_ids_old[train_index] != current_camera)
				continue;
			const int distance = HammingDis(window_brief_descriptors[query_index],
				descriptors_old[train_index]);
			if (distance < best_distance)
			{
				second_distance = best_distance;
				best_distance = distance;
				best_index = train_index;
			}
			else if (distance < second_distance)
			{
				second_distance = distance;
			}

			// Reverse nearest neighbour for the mutual cross-check below.
			if (distance < reverse_best_distance[train_index])
			{
				reverse_best_distance[train_index] = distance;
				reverse_best_query[train_index] = query_index;
			}
		}

		if (best_index < 0)
			continue;
		best_distance_sum += best_distance;
		best_distance_count++;
		if (best_distance >= LOOP_BRIEF_DIST_TH)
			continue;
		stats.distance_pass++;

		// Reject tied or ambiguous nearest neighbours. Requiring a strict best
		// also prevents the all-zero/tied descriptors near image borders from
		// being accepted merely because 0 <= ratio * 0.
		if (second_distance == std::numeric_limits<int>::max() ||
			best_distance >= second_distance ||
			static_cast<double>(best_distance) > LOOP_BRIEF_RATIO_TH * second_distance)
			continue;

		stats.ratio_pass++;
		ratio_train_used[best_index] = 1;
		ratio_candidates.push_back({query_index, best_index, best_distance, second_distance});
	}

	if (best_distance_count > 0)
		stats.mean_best_distance = best_distance_sum / best_distance_count;
	for (uchar used : ratio_train_used)
		stats.unique_train_after_ratio += used ? 1 : 0;

	// Prefer the strongest candidates first. Mutual nearest-neighbour checking
	// plus this used-train mask guarantees a one-to-one correspondence set.
	std::sort(ratio_candidates.begin(), ratio_candidates.end(),
		[](const Candidate &a, const Candidate &b)
		{
			if (a.best_distance != b.best_distance)
				return a.best_distance < b.best_distance;
			return a.query_index < b.query_index;
		});
	std::vector<uchar> train_assigned(stats.train_count, 0);
	for (const Candidate &candidate : ratio_candidates)
	{
		if (reverse_best_query[candidate.train_index] != candidate.query_index ||
			train_assigned[candidate.train_index])
			continue;

		status[candidate.query_index] = 1;
		matched_2d_old[candidate.query_index] = keypoints_old[candidate.train_index].pt;
		matched_2d_old_norm[candidate.query_index] = keypoints_old_norm[candidate.train_index].pt;
		if (candidate.train_index < (int)camera_ids_old.size())
			matched_camera_old[candidate.query_index] = camera_ids_old[candidate.train_index];
		if (candidate.train_index < (int)feature_indices_old.size())
			matched_feature_old[candidate.query_index] = feature_indices_old[candidate.train_index];
		train_assigned[candidate.train_index] = 1;
		stats.mutual_unique_pass++;
	}

	return stats;
}

OrbMatchStats KeyFrame::searchByORBDes(std::vector<cv::Point2f> &matched_2d_old,
									   std::vector<cv::Point2f> &matched_2d_old_norm,
									   std::vector<uchar> &status,
									   std::vector<int> &matched_camera_old,
									   std::vector<int> &matched_feature_old,
									   const KeyFrame *old_kf,
									   bool dense_old,
									   int distance_threshold,
									   double ratio_threshold) const
{
	OrbMatchStats stats;
	const int output_count = static_cast<int>(point_2d_uv.size());
	matched_2d_old.assign(output_count, cv::Point2f(0.f, 0.f));
	matched_2d_old_norm.assign(output_count, cv::Point2f(0.f, 0.f));
	status.assign(output_count, 0);
	matched_camera_old.assign(output_count, -1);
	matched_feature_old.assign(output_count, -1);

	// The query side is always the exact current VIO feature set, whose row is
	// permanently bound to a current 3D point. The old side may be either the
	// exact VIO set (fallback/self-test) or the denser canonical-view set.
	const cv::Mat &current_descriptors = window_orb_descriptors;
	stats.query_count = std::min<int>(current_descriptors.rows,
		window_orb_point_indices.size());
	const cv::Mat &old_descriptors = dense_old
		? old_kf->orb_descriptors : old_kf->window_orb_descriptors;
	stats.train_count = dense_old
		? std::min<int>(old_descriptors.rows,
			std::min<int>(old_kf->orb_keypoints_raw.size(),
				std::min<int>(old_kf->orb_keypoints_norm.size(),
					old_kf->orb_keypoint_camera_ids.size())))
		: std::min<int>(old_descriptors.rows, old_kf->window_orb_point_indices.size());
	if (stats.query_count == 0 || stats.train_count < 2)
		return stats;

	double best_distance_sum = 0.0;
	int best_distance_count = 0;
	struct AcceptedMatch
	{
		int point_index;
		int train_index;
		float descriptor_distance;
	};
	std::vector<AcceptedMatch> accepted_matches;
	auto current_camera_for_row = [&](int query_row)
	{
		if (query_row < 0 || query_row >= stats.query_count)
			return -1;
		const int point_index = window_orb_point_indices[query_row];
		if (point_index < 0 || point_index >= (int)loop_features.size())
			return -1;
		return loop_features[point_index].camera_id;
	};
	auto old_camera_for_row = [&](int train_row)
	{
		if (train_row < 0 || train_row >= stats.train_count)
			return -1;
		if (dense_old)
			return old_kf->orb_keypoint_camera_ids[train_row];
		const int old_feature_index = old_kf->window_orb_point_indices[train_row];
		if (old_feature_index < 0 ||
			old_feature_index >= (int)old_kf->loop_features.size())
			return -1;
		return old_kf->loop_features[old_feature_index].camera_id;
	};

	// This is a rigid stereo rig: a temporal loop correspondence must preserve
	// camera identity. Matching a combined descriptor pool allows similar views
	// from cam0/cam1 to swap, which weakens each camera's independent PnP and
	// produces misleading cross-camera debug lines. Run KNN independently for
	// every camera so both the ratio test and mutual check are camera-local.
	int max_camera_id = -1;
	std::vector<int> query_camera_ids(stats.query_count, -1);
	std::vector<int> train_camera_ids(stats.train_count, -1);
	for (int row = 0; row < stats.query_count; ++row)
	{
		query_camera_ids[row] = current_camera_for_row(row);
		max_camera_id = std::max(max_camera_id, query_camera_ids[row]);
	}
	for (int row = 0; row < stats.train_count; ++row)
	{
		train_camera_ids[row] = old_camera_for_row(row);
		max_camera_id = std::max(max_camera_id, train_camera_ids[row]);
	}
	if (max_camera_id < 0)
		return stats;

	std::vector<std::vector<int>> query_rows(max_camera_id + 1);
	std::vector<std::vector<int>> train_rows(max_camera_id + 1);
	for (int row = 0; row < stats.query_count; ++row)
		if (query_camera_ids[row] >= 0)
			query_rows[query_camera_ids[row]].push_back(row);
	for (int row = 0; row < stats.train_count; ++row)
		if (train_camera_ids[row] >= 0)
			train_rows[train_camera_ids[row]].push_back(row);

	bool matcher_attempted = false;
	bool all_matchers_used_gpu = true;
	for (int camera_id = 0; camera_id <= max_camera_id; ++camera_id)
	{
		if (query_rows[camera_id].empty() || train_rows[camera_id].size() < 2)
			continue;

		cv::Mat query_descriptors;
		cv::Mat train_descriptors;
		for (int row : query_rows[camera_id])
			query_descriptors.push_back(current_descriptors.row(row));
		for (int row : train_rows[camera_id])
			train_descriptors.push_back(old_descriptors.row(row));

		std::vector<std::vector<cv::DMatch>> forward_matches;
		std::vector<std::vector<cv::DMatch>> reverse_matches;
		const bool camera_gpu_used = cudaOrbKnnMatch(query_descriptors,
			train_descriptors, forward_matches, reverse_matches);
		matcher_attempted = true;
		all_matchers_used_gpu = all_matchers_used_gpu && camera_gpu_used;
		if (!camera_gpu_used)
		{
			cv::BFMatcher matcher(cv::NORM_HAMMING, false);
			matcher.knnMatch(query_descriptors, train_descriptors, forward_matches, 2);
			matcher.knnMatch(train_descriptors, query_descriptors, reverse_matches, 1);
		}

		for (int local_query_row = 0;
			local_query_row < (int)forward_matches.size(); ++local_query_row)
		{
			if (forward_matches[local_query_row].empty())
				continue;
			const cv::DMatch &best = forward_matches[local_query_row][0];
			best_distance_sum += best.distance;
			best_distance_count++;
			if (best.distance >= distance_threshold)
				continue;
			stats.distance_pass++;
			if (forward_matches[local_query_row].size() < 2)
				continue;
			const cv::DMatch &second = forward_matches[local_query_row][1];
			if (!(best.distance < second.distance) ||
				best.distance > ratio_threshold * second.distance)
				continue;
			stats.ratio_pass++;
			if (best.trainIdx < 0 || best.trainIdx >= (int)reverse_matches.size() ||
				reverse_matches[best.trainIdx].empty() ||
				reverse_matches[best.trainIdx][0].trainIdx != local_query_row)
				continue;

			const int query_row = query_rows[camera_id][local_query_row];
			const int train_row = train_rows[camera_id][best.trainIdx];
			const int point_index = window_orb_point_indices[query_row];
			if (point_index < 0 || point_index >= output_count)
				continue;
			accepted_matches.push_back({point_index, train_row, best.distance});
		}
	}
	stats.gpu_used = matcher_attempted && all_matchers_used_gpu;

	if (best_distance_count > 0)
		stats.mean_best_distance = best_distance_sum / best_distance_count;

	// Retain only the strongest descriptor match for each current 3D point.
	// Mutual nearest-neighbour checking above already makes the old descriptor
	// side one-to-one.
	std::sort(accepted_matches.begin(), accepted_matches.end(),
		[](const AcceptedMatch &a, const AcceptedMatch &b)
		{
			return a.descriptor_distance < b.descriptor_distance;
		});
	std::vector<uchar> point_assigned(output_count, 0);
	for (const AcceptedMatch &match : accepted_matches)
	{
		if (point_assigned[match.point_index])
			continue;
		status[match.point_index] = 1;
		if (dense_old)
		{
			matched_2d_old[match.point_index] =
				old_kf->orb_keypoints_raw[match.train_index].pt;
			matched_2d_old_norm[match.point_index] =
				old_kf->orb_keypoints_norm[match.train_index].pt;
			matched_camera_old[match.point_index] =
				old_kf->orb_keypoint_camera_ids[match.train_index];
		}
		else
		{
			const int old_feature_index = old_kf->window_orb_point_indices[match.train_index];
			if (old_feature_index < 0 || old_feature_index >= (int)old_kf->loop_features.size())
				continue;
			const LoopFeature &old_feature = old_kf->loop_features[old_feature_index];
			matched_2d_old[match.point_index] = old_feature.raw_uv;
			matched_2d_old_norm[match.point_index] = old_feature.normalized_uv;
			matched_camera_old[match.point_index] = old_feature.camera_id;
			matched_feature_old[match.point_index] = old_feature_index;
		}
		point_assigned[match.point_index] = 1;
		stats.mutual_unique_pass++;
	}
	return stats;
}


bool KeyFrame::EpipolarRANSAC(
	const std::vector<cv::Point2f> &matched_2d_cur_norm,
	const std::vector<cv::Point2f> &matched_2d_old_norm,
	vector<uchar> &status) const
{
	const int count = static_cast<int>(matched_2d_cur_norm.size());
	status.assign(count, 1);
	if (matched_2d_old_norm.size() != matched_2d_cur_norm.size() ||
		count < LOOP_EPIPOLAR_MIN_MATCHES)
		return false;

	const double normalized_threshold = LOOP_EPIPOLAR_REPROJECTION_ERROR_PX /
		std::max(1.0, LOOP_VIEW_FOCAL);
	cv::Mat mask;
	try
	{
		const cv::Mat essential = cv::findEssentialMat(
			matched_2d_cur_norm, matched_2d_old_norm, 1.0, cv::Point2d(0.0, 0.0),
			cv::RANSAC, LOOP_EPIPOLAR_CONFIDENCE, normalized_threshold, mask);
		if (essential.empty() || mask.empty() || mask.total() != (size_t)count)
			return false;
	}
	catch (const cv::Exception &exception)
	{
		ROS_WARN_THROTTLE(1.0,
			"[loop_fusion][epipolar] Essential RANSAC exception: %s",
			exception.what());
		return false;
	}

	if (!mask.isContinuous())
		mask = mask.clone();
	const uchar *mask_data = mask.ptr<uchar>(0);
	for (int i = 0; i < count; ++i)
		status[i] = mask_data[i] ? 1 : 0;
	return true;
}

bool KeyFrame::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                         const std::vector<cv::Point3f> &matched_3d,
                         std::vector<uchar> &status,
                         Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old,
                         int camera_id)
{
    status.assign(matched_2d_old_norm.size(), 0);
    PnP_T_old.setZero();
    PnP_R_old.setIdentity();
    if (camera_id < 0 || camera_id >= (int)loop_qics.size() ||
        matched_3d.size() != matched_2d_old_norm.size() || matched_3d.size() < 4)
        return false;

    cv::Mat r, rvec, t, D;
    const cv::Mat K = (cv::Mat_<double>(3, 3) <<
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0);
    const Matrix3d &camera_qic = loop_qics.at(camera_id);
    const Vector3d &camera_tic = loop_tics.at(camera_id);
    cv::Mat inliers;

    double camera_focal = LOOP_FOCAL_LENGTH;
    if (camera_id < (int)m_cameras.size() && m_cameras[camera_id])
    {
        std::vector<double> parameters;
        m_cameras[camera_id]->writeParameters(parameters);
        if (m_cameras[camera_id]->modelType() == camodocal::Camera::EUCM &&
            parameters.size() >= 4)
            camera_focal = 0.5 * (parameters[2] + parameters[3]);
    }
    // Descriptor locations live in the canonical perspective views. Scaling
    // by the raw EUCM fx (~619 px here) made a nominal 10 px threshold about
    // 4.6x tighter than intended because the canonical focal is ~134 px.
    const double threshold_focal = LOOP_VIEW_FOCAL > 1.0
        ? LOOP_VIEW_FOCAL : camera_focal;
    const double normalized_threshold = LOOP_PNP_REPROJECTION_ERROR_PX /
        std::max(1.0, threshold_focal);

    bool solved = false;
    try
    {
#if CV_MAJOR_VERSION < 3
        solved = solvePnPRansac(matched_3d, matched_2d_old_norm, K, D,
            rvec, t, false, LOOP_PNP_RANSAC_ITERATIONS, normalized_threshold,
            4, inliers);
#else
        // A loop candidate can be far from the drifted current VIO pose. Using
        // that pose as an extrinsic guess made RANSAC repeatedly return zero
        // inliers. EPNP now estimates each candidate independently.
        solved = solvePnPRansac(matched_3d, matched_2d_old_norm, K, D,
            rvec, t, false, LOOP_PNP_RANSAC_ITERATIONS, normalized_threshold,
            LOOP_PNP_CONFIDENCE, inliers, cv::SOLVEPNP_EPNP);
#endif
    }
    catch (const cv::Exception &exception)
    {
        ROS_WARN_THROTTLE(1.0,
            "[loop_fusion][EUCM-PnP] solvePnPRansac exception: %s",
            exception.what());
        return false;
    }

    if (!solved || inliers.rows < 4 || rvec.empty() || t.empty())
        return false;

#if CV_MAJOR_VERSION >= 3
    vector<cv::Point3f> inlier_3d;
    vector<cv::Point2f> inlier_2d;
    inlier_3d.reserve(inliers.rows);
    inlier_2d.reserve(inliers.rows);
    for (int i = 0; i < inliers.rows; ++i)
    {
        const int input_index = inliers.at<int>(i);
        if (input_index < 0 || input_index >= (int)matched_3d.size())
            continue;
        inlier_3d.push_back(matched_3d[input_index]);
        inlier_2d.push_back(matched_2d_old_norm[input_index]);
    }
    if (inlier_3d.size() >= 4)
    {
        try
        {
            solvePnP(inlier_3d, inlier_2d, K, D, rvec, t, true,
                cv::SOLVEPNP_ITERATIVE);
        }
        catch (const cv::Exception &exception)
        {
            ROS_WARN_THROTTLE(1.0,
                "[loop_fusion][EUCM-PnP] refinement exception: %s",
                exception.what());
        }
    }
#endif

    for (int i = 0; i < inliers.rows; ++i)
    {
        const int input_index = inliers.at<int>(i);
        if (input_index >= 0 && input_index < (int)status.size())
            status[input_index] = 1;
    }

    cv::Rodrigues(rvec, r);
    Matrix3d R_pnp, R_w_c_old;
    cv::cv2eigen(r, R_pnp);
    R_w_c_old = R_pnp.transpose();
    Vector3d T_pnp, T_w_c_old;
    cv::cv2eigen(t, T_pnp);
    T_w_c_old = R_w_c_old * (-T_pnp);

    PnP_R_old = R_w_c_old * camera_qic.transpose();
    PnP_T_old = T_w_c_old - PnP_R_old * camera_tic;
    return PnP_R_old.allFinite() && PnP_T_old.allFinite();
}


bool KeyFrame::findConnection(KeyFrame* old_kf, bool allow_dense_fallback)
{
	if (old_kf == nullptr)
		return false;
	if (LOOP_PRESERVE_ORIGINAL_FLOW)
		return findConnectionWithORBSource(old_kf, false);
	const bool current_exact_ready =
		window_orb_descriptors.rows > LOOP_PER_CAMERA_MIN_INLIERS &&
		(int)window_orb_point_indices.size() > LOOP_PER_CAMERA_MIN_INLIERS;
	const bool old_exact_ready =
		old_kf->window_orb_descriptors.rows > LOOP_PER_CAMERA_MIN_INLIERS &&
		(int)old_kf->window_orb_point_indices.size() > LOOP_PER_CAMERA_MIN_INLIERS;
	const bool old_dense_ready =
		old_kf->orb_descriptors.rows > LOOP_PER_CAMERA_MIN_INLIERS &&
		(int)old_kf->orb_keypoints_raw.size() > LOOP_PER_CAMERA_MIN_INLIERS &&
		(int)old_kf->orb_keypoints_norm.size() > LOOP_PER_CAMERA_MIN_INLIERS &&
		(int)old_kf->orb_keypoint_camera_ids.size() > LOOP_PER_CAMERA_MIN_INLIERS;

	bool exact_attempted = false;
	if (LOOP_ORB_GEOMETRY && current_exact_ready && old_exact_ready)
	{
		exact_attempted = true;
		if (findConnectionWithORBSource(old_kf, false))
			return true;
	}
	if (LOOP_ORB_GEOMETRY && allow_dense_fallback &&
		current_exact_ready && old_dense_ready)
	{
		ROS_INFO("[loop_fusion][ORB] cur=%d old=%d exact-exact rejected; trying dense fallback",
			index, old_kf->index);
		return findConnectionWithORBSource(old_kf, true);
	}
	if (exact_attempted)
		return false;

	// Preserve BRIEF verification for legacy/loaded keyframes that do not
	// contain the exact VIO ORB descriptor set.
	return findConnectionWithORBSource(old_kf, false);
}

bool KeyFrame::findConnectionWithORBSource(KeyFrame* old_kf, bool dense_old)
{
	TicToc tmp_t;
	//printf("find Connection\n");
	vector<cv::Point2f> matched_2d_cur, matched_2d_old;
	vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
	vector<cv::Point3f> matched_3d;
	vector<double> matched_id;
	vector<uchar> status;
	vector<int> matched_camera_cur;
	vector<int> matched_camera_old;
	vector<int> matched_feature_old;

	matched_3d = point_3d;
	matched_2d_cur = point_2d_uv;
	matched_2d_cur_norm = point_2d_norm;
	matched_id = point_id;
	matched_camera_cur.assign(matched_3d.size(), -1);
	for (int i = 0; i < std::min<int>(matched_camera_cur.size(), loop_features.size()); ++i)
		matched_camera_cur[i] = loop_features[i].camera_id;

	TicToc t_match;
	#if 0
		if (DEBUG_IMAGE)    
	    {
	        cv::Mat gray_img, loop_match_img;
	        cv::Mat old_img = old_kf->image;
	        cv::hconcat(image, old_img, gray_img);
	        cvtColor(gray_img, loop_match_img, cv::COLOR_GRAY2RGB);
	        for(int i = 0; i< (int)point_2d_uv.size(); i++)
	        {
	            cv::Point2f cur_pt = point_2d_uv[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)old_kf->keypoints.size(); i++)
	        {
	            cv::Point2f old_pt = old_kf->keypoints[i].pt;
	            old_pt.x += COL;
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        ostringstream path;
	        path << "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "0raw_point.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
	#endif
	//printf("search by des\n");
	const bool forced_early_candidate = !LOOP_PRESERVE_ORIGINAL_FLOW &&
		LOOP_ORB_GEOMETRY &&
		LOOP_FORCE_FIRST_KEYFRAMES > 0 && old_kf->index >= 0 &&
		old_kf->index < LOOP_FORCE_FIRST_KEYFRAMES;
	const bool current_exact_ready =
		window_orb_descriptors.rows > LOOP_PER_CAMERA_MIN_INLIERS &&
		(int)window_orb_point_indices.size() > LOOP_PER_CAMERA_MIN_INLIERS;
	const bool old_dense_ready =
		old_kf->orb_descriptors.rows > LOOP_PER_CAMERA_MIN_INLIERS &&
		(int)old_kf->orb_keypoints_raw.size() > LOOP_PER_CAMERA_MIN_INLIERS &&
		(int)old_kf->orb_keypoints_norm.size() > LOOP_PER_CAMERA_MIN_INLIERS &&
		(int)old_kf->orb_keypoint_camera_ids.size() > LOOP_PER_CAMERA_MIN_INLIERS;
	const bool old_exact_ready =
		old_kf->window_orb_descriptors.rows > LOOP_PER_CAMERA_MIN_INLIERS &&
		(int)old_kf->window_orb_point_indices.size() > LOOP_PER_CAMERA_MIN_INLIERS;
	const bool use_exact_dense_old = dense_old && current_exact_ready && old_dense_ready;
	const bool use_orb = !LOOP_PRESERVE_ORIGINAL_FLOW &&
		LOOP_ORB_GEOMETRY && current_exact_ready &&
		(use_exact_dense_old || (!dense_old && old_exact_ready));
	const int orb_distance_threshold = use_exact_dense_old
		? (forced_early_candidate ? LOOP_ORB_DENSE_DIST_TH :
			LOOP_ORB_EXACT_DENSE_DIST_TH)
		: LOOP_ORB_DIST_TH;
	const double orb_ratio_threshold = use_exact_dense_old
		? (forced_early_candidate ? LOOP_ORB_DENSE_RATIO_TH :
			LOOP_ORB_EXACT_DENSE_RATIO_TH)
		: LOOP_ORB_RATIO_TH;
	const char *orb_source = use_exact_dense_old
		? (forced_early_candidate ? "exact-VIO-to-dense-canonical-old-forced" :
			"exact-VIO-to-dense-canonical-old")
		: "exact-VIO-to-exact-VIO";
	if (use_orb)
	{
			const OrbMatchStats orb_stats = searchByORBDes(
				matched_2d_old, matched_2d_old_norm, status, matched_camera_old,
				matched_feature_old, old_kf, use_exact_dense_old,
				orb_distance_threshold, orb_ratio_threshold);
		const double descriptor_match_ms = t_match.toc();
		ROS_INFO("[loop_fusion][ORB] cur=%d old=%d query=%d train=%d dist=%d ratio=%d "
			"mutual_unique=%d mean_best=%.1f source=%s "
			"dist_th=%d ratio_th=%.2f backend=%s match=%.2fms",
			index, old_kf->index, orb_stats.query_count, orb_stats.train_count,
			orb_stats.distance_pass, orb_stats.ratio_pass,
			orb_stats.mutual_unique_pass, orb_stats.mean_best_distance,
			orb_source, orb_distance_threshold, orb_ratio_threshold,
			orb_stats.gpu_used ? "CUDA" : "CPU",
			descriptor_match_ms);
	}
	else
	{
		const bool use_old_window = !LOOP_PRESERVE_ORIGINAL_FLOW &&
			!old_kf->window_brief_descriptors.empty() &&
			old_kf->window_brief_descriptors.size() == old_kf->window_keypoints.size() &&
			old_kf->window_keypoints.size() == old_kf->window_keypoints_norm.size();
		const std::vector<BRIEF::bitset> &old_descriptors = use_old_window
			? old_kf->window_brief_descriptors : old_kf->brief_descriptors;
		const std::vector<cv::KeyPoint> &old_keypoints = use_old_window
			? old_kf->window_keypoints : old_kf->keypoints;
			const std::vector<cv::KeyPoint> &old_keypoints_norm = use_old_window
				? old_kf->window_keypoints_norm : old_kf->keypoints_norm;
			std::vector<int> old_camera_ids(old_descriptors.size(), 0);
			std::vector<int> old_feature_indices(old_descriptors.size(), -1);
			if (use_old_window)
			{
				const int count = std::min<int>(old_descriptors.size(), old_kf->loop_features.size());
				for (int i = 0; i < count; ++i)
				{
					old_camera_ids[i] = old_kf->loop_features[i].camera_id;
					old_feature_indices[i] = i;
				}
			}
			else if (old_kf->brief_keypoint_camera_ids.size() ==
				old_descriptors.size())
			{
				old_camera_ids = old_kf->brief_keypoint_camera_ids;
			}
			const BriefMatchStats brief_stats = searchByBRIEFDes(
				matched_2d_old, matched_2d_old_norm, status, matched_camera_old,
				matched_feature_old, old_descriptors, old_keypoints, old_keypoints_norm,
				old_camera_ids, old_feature_indices);
		const double descriptor_match_ms = t_match.toc();
		ROS_INFO("[loop_fusion][BRIEF] cur=%d old=%d query=%d train=%d dist=%d ratio=%d "
			"ratio_unique=%d mutual_unique=%d mean_best=%.1f source=%s dist_th=%d ratio_th=%.2f "
			"match=%.2fms",
			index, old_kf->index, brief_stats.query_count, brief_stats.train_count,
			brief_stats.distance_pass, brief_stats.ratio_pass,
			brief_stats.unique_train_after_ratio, brief_stats.mutual_unique_pass,
			brief_stats.mean_best_distance, use_old_window ? "VIO-window" : "stored-fallback",
			LOOP_BRIEF_DIST_TH, LOOP_BRIEF_RATIO_TH, descriptor_match_ms);
	}
	reduceVector(matched_2d_cur, status);
	reduceVector(matched_2d_old, status);
	reduceVector(matched_2d_cur_norm, status);
	reduceVector(matched_2d_old_norm, status);
	reduceVector(matched_3d, status);
	reduceVector(matched_id, status);
	reduceVector(matched_camera_cur, status);
	reduceVector(matched_camera_old, status);
	reduceVector(matched_feature_old, status);
	const int eucm_descriptor_matches = matched_3d.size();
	int camera_pairs[2][2] = {{0, 0}, {0, 0}};
	for (int i = 0; i < std::min<int>(matched_camera_cur.size(),
		matched_camera_old.size()); ++i)
	{
		const int current_camera = matched_camera_cur[i];
		const int old_camera = matched_camera_old[i];
		if (current_camera >= 0 && current_camera < 2 &&
			old_camera >= 0 && old_camera < 2)
			camera_pairs[current_camera][old_camera]++;
	}
	const bool forced_geometry_active = use_orb && use_exact_dense_old &&
		forced_early_candidate;
	const bool exact_dense_geometry_active = use_orb && use_exact_dense_old;
	const int required_pnp_inliers = forced_geometry_active
		? LOOP_ORB_DENSE_MIN_INLIERS
		: (use_orb ? LOOP_ORB_EXACT_DENSE_MIN_INLIERS :
			LOOP_MIN_LOOP_NUM + 1);
	ROS_INFO("[loop_fusion][EUCM-PnP] cur=%d old=%d cam=multi descriptor_matches=%d "
		"pairs=0->0:%d,0->1:%d,1->0:%d,1->1:%d required=%d source=%s",
		index, old_kf->index, eucm_descriptor_matches,
		camera_pairs[0][0], camera_pairs[0][1],
		camera_pairs[1][0], camera_pairs[1][1],
		required_pnp_inliers,
		use_orb ? (exact_dense_geometry_active ? "ORB-exact-dense" :
			"ORB-exact-exact") : "BRIEF");
	//printf("search by des finish\n");

	#if 0 
		if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, cv::COLOR_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap);
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path, path1, path2;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	        /*
	        path1 <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_1.jpg";
	        cv::imwrite( path1.str().c_str(), image);
	        path2 <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_2.jpg";
	        cv::imwrite( path2.str().c_str(), old_img);	        
	        */
	        
	    }
	#endif
	status.clear();
	#if 0
		if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, cv::COLOR_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap) ;
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "2fundamental_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
	#endif
	#if 0
	Eigen::Vector3d PnP_T_old;
	Eigen::Matrix3d PnP_R_old;
	Eigen::Vector3d relative_t;
	Quaterniond relative_q;
	double relative_yaw;
	if ((int)matched_2d_cur.size() >= required_pnp_inliers)
	{
		status.clear();
	    PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old, 0);
	    reduceVector(matched_2d_cur, status);
	    reduceVector(matched_2d_old, status);
	    reduceVector(matched_2d_cur_norm, status);
	    reduceVector(matched_2d_old_norm, status);
	    reduceVector(matched_3d, status);
	    reduceVector(matched_id, status);
	    ROS_INFO("[loop_fusion][EUCM-PnP] cur=%d old=%d cam=0 inliers=%zu",
	        index, old_kf->index, matched_3d.size());
	    #if 1
	    	if (DEBUG_IMAGE)
	        {
	        	int gap = 10;
	        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
	            cv::Mat gray_img, loop_match_img;
	            cv::Mat old_img = old_kf->image;
	            cv::hconcat(image, gap_image, gap_image);
	            cv::hconcat(gap_image, old_img, gray_img);
	            cvtColor(gray_img, loop_match_img, cv::COLOR_GRAY2RGB);
	            for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	            {
	                cv::Point2f cur_pt = matched_2d_cur[i];
	                cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	            }
	            for(int i = 0; i< (int)matched_2d_old.size(); i++)
	            {
	                cv::Point2f old_pt = matched_2d_old[i];
	                old_pt.x += (COL + gap);
	                cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	            }
	            for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	            {
	                cv::Point2f old_pt = matched_2d_old[i];
	                old_pt.x += (COL + gap) ;
	                cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 2, 8, 0);
	            }
	            cv::Mat notation(50, COL + gap + COL, CV_8UC3, cv::Scalar(255, 255, 255));
	            putText(notation, "current frame: " + to_string(index) + "  sequence: " + to_string(sequence), cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);

	            putText(notation, "previous frame: " + to_string(old_kf->index) + "  sequence: " + to_string(old_kf->sequence), cv::Point2f(20 + COL + gap, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
	            cv::vconcat(notation, loop_match_img, loop_match_img);

	            /*
	            ostringstream path;
	            path <<  "/home/tony-ws1/raw_data/loop_image/"
	                    << index << "-"
	                    << old_kf->index << "-" << "3pnp_match.jpg";
	            cv::imwrite( path.str().c_str(), loop_match_img);
	            */
	            if ((int)matched_2d_cur.size() >= required_pnp_inliers)
	            {
	            	/*
	            	cv::imshow("loop connection",loop_match_img);  
	            	cv::waitKey(10);  
	            	*/
	            	cv::Mat thumbimage;
	            	cv::resize(loop_match_img, thumbimage, cv::Size(loop_match_img.cols / 2, loop_match_img.rows / 2));
	    	    	sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", thumbimage).toImageMsg();
	                msg->header.stamp = ros::Time(time_stamp);
	    	    	pub_match_img.publish(msg);
	            }
	        }
	    #endif
	}

	if ((int)matched_2d_cur.size() >= required_pnp_inliers)
	{
	    relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
	    const Eigen::Matrix3d relative_R = PnP_R_old.transpose() * origin_vio_R;
	    relative_q = Quaterniond(relative_R);
	    relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());
	    //printf("PNP relative\n");
	    //cout << "pnp relative_t " << relative_t.transpose() << endl;
	    //cout << "pnp relative_yaw " << relative_yaw << endl;
	    const double max_loop_yaw = forced_geometry_active
	    	? LOOP_ORB_DENSE_MAX_YAW_DEG : LOOP_MAX_YAW_DEG;
	    const bool translation_ok = LOOP_MAX_TRANSLATION <= 0.0 ||
	        relative_t.norm() < LOOP_MAX_TRANSLATION;
	    if (abs(relative_yaw) < max_loop_yaw && translation_ok)
	    {

	    	has_loop = true;
	    	loop_index = old_kf->index;
	    	loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
	    	             relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
	    	             relative_yaw;
	        ROS_INFO("[loop_fusion] accepted EUCM loop cur=%d old=%d inliers=%zu yaw=%.2f/%.2f t=%.2f",
	            index, old_kf->index, matched_3d.size(), relative_yaw,
	            max_loop_yaw, relative_t.norm());
	    	//cout << "pnp relative_t " << relative_t.transpose() << endl;
	    	//cout << "pnp relative_q " << relative_q.w() << " " << relative_q.vec().transpose() << endl;
	        return true;
	    }
	    ROS_INFO("[loop_fusion][EUCM-PnP] rejected pose cur=%d old=%d inliers=%zu yaw=%.2f/%.2f t=%.2f/%.2f",
	        index, old_kf->index, matched_3d.size(), relative_yaw, max_loop_yaw,
	        relative_t.norm(), LOOP_MAX_TRANSLATION);
	}
	#endif

	struct CameraPnPResult
	{
		int camera_id = -1;
		int descriptor_matches = 0;
		int epipolar_matches = 0;
		int inliers = 0;
		Eigen::Vector3d T = Eigen::Vector3d::Zero();
		Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
		vector<int> inlier_match_indices;
	};
	vector<CameraPnPResult> camera_results;
	const int per_camera_min = forced_geometry_active ? required_pnp_inliers :
		std::min(required_pnp_inliers, LOOP_PER_CAMERA_MIN_INLIERS);
	for (int camera_id = 0; camera_id < (int)m_cameras.size(); ++camera_id)
	{
		vector<cv::Point2f> camera_points_cur;
		vector<cv::Point2f> camera_points_old;
		vector<cv::Point3f> camera_points_3d;
		vector<int> camera_match_indices;
		for (int i = 0; i < (int)matched_3d.size(); ++i)
		{
			if (i >= (int)matched_camera_cur.size() ||
				i >= (int)matched_camera_old.size() ||
				matched_camera_cur[i] != camera_id ||
				matched_camera_old[i] != camera_id)
				continue;
			if (i < (int)matched_feature_old.size() && matched_feature_old[i] >= 0)
			{
				const int old_feature_index = matched_feature_old[i];
				if (old_feature_index >= (int)old_kf->loop_features.size() ||
					old_kf->loop_features[old_feature_index].bearing.z() <= LOOP_EUCM_MIN_Z)
					continue;
			}
			if (!std::isfinite(matched_2d_cur_norm[i].x) ||
				!std::isfinite(matched_2d_cur_norm[i].y) ||
				!std::isfinite(matched_2d_old_norm[i].x) ||
				!std::isfinite(matched_2d_old_norm[i].y) ||
				!std::isfinite(matched_3d[i].x) ||
				!std::isfinite(matched_3d[i].y) ||
				!std::isfinite(matched_3d[i].z))
				continue;
			camera_points_cur.push_back(matched_2d_cur_norm[i]);
			camera_points_old.push_back(matched_2d_old_norm[i]);
			camera_points_3d.push_back(matched_3d[i]);
			camera_match_indices.push_back(i);
		}
		const int descriptor_matches = static_cast<int>(camera_points_3d.size());
		vector<uchar> epipolar_status;
		const bool epipolar_applied = EpipolarRANSAC(
			camera_points_cur, camera_points_old, epipolar_status);
		if (epipolar_applied)
		{
			reduceVector(camera_points_cur, epipolar_status);
			reduceVector(camera_points_old, epipolar_status);
			reduceVector(camera_points_3d, epipolar_status);
			reduceVector(camera_match_indices, epipolar_status);
		}
		ROS_INFO("[loop_fusion][epipolar] cur=%d old=%d cam=%d descriptor=%d "
			"inliers=%zu applied=%d min=%d threshold=%.1fpx",
			index, old_kf->index, camera_id, descriptor_matches,
			camera_points_3d.size(), epipolar_applied ? 1 : 0,
			LOOP_EPIPOLAR_MIN_MATCHES, LOOP_EPIPOLAR_REPROJECTION_ERROR_PX);
		if ((int)camera_points_3d.size() < per_camera_min)
		{
			ROS_INFO("[loop_fusion][EUCM-PnP] cur=%d old=%d cam=%d epipolar=%zu required=%d",
				index, old_kf->index, camera_id, camera_points_3d.size(), per_camera_min);
			continue;
		}
		CameraPnPResult result;
		result.camera_id = camera_id;
		result.descriptor_matches = descriptor_matches;
		result.epipolar_matches = static_cast<int>(camera_points_3d.size());
		vector<uchar> camera_status;
		const bool pnp_solved = PnPRANSAC(camera_points_old, camera_points_3d, camera_status,
			result.T, result.R, camera_id);
		result.inliers = std::count(camera_status.begin(), camera_status.end(), (uchar)1);
		for (int i = 0; i < std::min<int>(camera_status.size(), camera_match_indices.size()); ++i)
			if (camera_status[i])
				result.inlier_match_indices.push_back(camera_match_indices[i]);
		ROS_INFO("[loop_fusion][EUCM-PnP] cur=%d old=%d cam=%d descriptor=%d "
			"epipolar=%d solved=%d inliers=%d",
			index, old_kf->index, camera_id, result.descriptor_matches,
			result.epipolar_matches,
			pnp_solved ? 1 : 0, result.inliers);
		if (result.inliers >= per_camera_min)
			camera_results.push_back(result);
	}
	bool cameras_consistent = true;
	if (camera_results.size() >= 2)
	{
		const Eigen::Matrix3d delta_R = camera_results[0].R.transpose() * camera_results[1].R;
		const double rotation_delta = Eigen::AngleAxisd(delta_R).angle() * 180.0 / M_PI;
		const double translation_delta = (camera_results[0].T - camera_results[1].T).norm();
		cameras_consistent = rotation_delta <= LOOP_CAMERA_CONSISTENCY_ROTATION_DEG &&
			translation_delta <= LOOP_CAMERA_CONSISTENCY_TRANSLATION;
		ROS_INFO("[loop_fusion][EUCM-PnP] cur=%d old=%d camera_consistency=%d dR=%.2f/%.2fdeg dT=%.3f/%.3fm",
			index, old_kf->index, cameras_consistent ? 1 : 0, rotation_delta,
			LOOP_CAMERA_CONSISTENCY_ROTATION_DEG, translation_delta,
			LOOP_CAMERA_CONSISTENCY_TRANSLATION);
	}

	const int total_camera_inliers = std::accumulate(camera_results.begin(), camera_results.end(), 0,
		[](int total, const CameraPnPResult &result) { return total + result.inliers; });
	const CameraPnPResult *best_result = nullptr;
	for (const CameraPnPResult &result : camera_results)
		if (!best_result || result.inliers > best_result->inliers)
			best_result = &result;
	const int single_camera_required = use_orb && !forced_geometry_active
		? LOOP_SINGLE_CAMERA_MIN_INLIERS : required_pnp_inliers;
	// Keep the single-camera ratio conservative: use all descriptor matches as
	// the denominator, not only the epipolar-filtered subset.
	const double best_inlier_ratio = best_result && best_result->descriptor_matches > 0
		? static_cast<double>(best_result->inliers) /
			best_result->descriptor_matches : 0.0;
	const bool single_camera_ratio_ok = !use_orb ||
		forced_geometry_active ||
		best_inlier_ratio >= LOOP_SINGLE_CAMERA_MIN_INLIER_RATIO;
	const bool enough_inliers = best_result &&
		(camera_results.size() >= 2 ? total_camera_inliers >= required_pnp_inliers
			: best_result->inliers >= single_camera_required &&
				single_camera_ratio_ok);

	if (best_result && enough_inliers && cameras_consistent)
	{
		const Eigen::Vector3d relative_t = best_result->R.transpose() *
			(origin_vio_T - best_result->T);
		const Eigen::Matrix3d relative_R =
			best_result->R.transpose() * origin_vio_R;
		const Quaterniond relative_q(relative_R);
		const double relative_yaw = Utility::normalizeAngle(
			Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(best_result->R).x());
		const double max_loop_yaw = forced_geometry_active
			? LOOP_ORB_DENSE_MAX_YAW_DEG : LOOP_MAX_YAW_DEG;
		const double relative_translation = relative_t.norm();
		const bool translation_ok = LOOP_MAX_TRANSLATION <= 0.0 ||
			relative_translation < LOOP_MAX_TRANSLATION;
		if (abs(relative_yaw) < max_loop_yaw && translation_ok)
		{
			has_loop = true;
			loop_index = old_kf->index;
			loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
				relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(), relative_yaw;
			ROS_INFO("[loop_fusion] accepted EUCM loop cur=%d old=%d cameras=%zu "
				"inliers=%d best_ratio=%.2f yaw=%.2f/%.2f t=%.2f",
				index, old_kf->index, camera_results.size(), total_camera_inliers,
				best_inlier_ratio, relative_yaw, max_loop_yaw, relative_translation);
			vector<int> accepted_pnp_inlier_match_indices;
			for (const CameraPnPResult &result : camera_results)
				accepted_pnp_inlier_match_indices.insert(
					accepted_pnp_inlier_match_indices.end(),
					result.inlier_match_indices.begin(),
					result.inlier_match_indices.end());
			std::sort(accepted_pnp_inlier_match_indices.begin(),
				accepted_pnp_inlier_match_indices.end());
			accepted_pnp_inlier_match_indices.erase(
				std::unique(accepted_pnp_inlier_match_indices.begin(),
					accepted_pnp_inlier_match_indices.end()),
				accepted_pnp_inlier_match_indices.end());
			publishLoopMatchImage(*this, *old_kf, matched_2d_cur, matched_2d_old,
				matched_camera_cur, matched_camera_old,
				accepted_pnp_inlier_match_indices);
			return true;
		}
		if (LOOP_MAX_TRANSLATION > 0.0)
			ROS_INFO("[loop_fusion][EUCM-PnP] rejected pose cur=%d old=%d cameras=%zu inliers=%d yaw=%.2f/%.2f t=%.2f/%.2f",
				index, old_kf->index, camera_results.size(), total_camera_inliers,
				relative_yaw, max_loop_yaw, relative_translation, LOOP_MAX_TRANSLATION);
		else
			ROS_INFO("[loop_fusion][EUCM-PnP] rejected pose cur=%d old=%d cameras=%zu inliers=%d yaw=%.2f/%.2f t=%.2f translation_gate=off",
				index, old_kf->index, camera_results.size(), total_camera_inliers,
				relative_yaw, max_loop_yaw, relative_translation);
	}
	else if (!camera_results.empty())
	{
		ROS_INFO("[loop_fusion][EUCM-PnP] rejected geometry cur=%d old=%d cameras=%zu "
			"total_inliers=%d dual_required=%d single_required=%d best_ratio=%.2f/%.2f "
			"consistent=%d",
			index, old_kf->index, camera_results.size(), total_camera_inliers,
			required_pnp_inliers, single_camera_required, best_inlier_ratio,
			LOOP_SINGLE_CAMERA_MIN_INLIER_RATIO, cameras_consistent ? 1 : 0);
	}
	//printf("loop final use num %d %lf--------------- \n", (int)matched_2d_cur.size(), t_match.toc());
	return false;
}


int KeyFrame::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b)
{
    BRIEF::bitset xor_of_bitset = a ^ b;
    int dis = xor_of_bitset.count();
    return dis;
}

void KeyFrame::getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = vio_T_w_i;
    _R_w_i = vio_R_w_i;
}

void KeyFrame::getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = T_w_i;
    _R_w_i = R_w_i;
}

void KeyFrame::updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
    T_w_i = _T_w_i;
    R_w_i = _R_w_i;
}

void KeyFrame::updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
}

Eigen::Vector3d KeyFrame::getLoopRelativeT()
{
    return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}

Eigen::Quaterniond KeyFrame::getLoopRelativeQ()
{
    return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
}

double KeyFrame::getLoopRelativeYaw()
{
    return loop_info(7);
}

void KeyFrame::updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info)
{
	if (abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0)
	{
		//printf("update loop info\n");
		loop_info = _loop_info;
	}
}

BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
  // The DVision::BRIEF extractor computes a random pattern by default when
  // the object is created.
  // We load the pattern that we used to build the vocabulary, to make
  // the descriptors compatible with the predefined vocabulary

  // loads the pattern
  cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
  if(!fs.isOpened()) throw string("Could not open file ") + pattern_file;

  vector<int> x1, y1, x2, y2;
  fs["x1"] >> x1;
  fs["x2"] >> x2;
  fs["y1"] >> y1;
  fs["y2"] >> y2;

  m_brief.importPairs(x1, y1, x2, y2);
}
