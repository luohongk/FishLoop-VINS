#include "fisheyeNode.hpp"
#include "featureTracker/feature_tracker_fisheye.hpp"
#include "featureTracker/fisheye_undist.hpp"
#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "depth_generation/depth_camera_manager.h"
#include <std_msgs/Header.h>

using namespace FeatureTracker;     

namespace
{
constexpr size_t MAX_FISHEYE_BUFFER_SIZE = 5;
constexpr int STEREO_SYNC_QUEUE_SIZE = 800;

template <typename QueueT>
void popIfNotEmpty(QueueT &queue)
{
    if (!queue.empty())
        queue.pop();
}
}

static bool cudaPipelineSmokeTest()
{
#ifdef USE_CUDA
    const char *stage = "device discovery";
    try {
        const int device_count = cv::cuda::getCudaEnabledDeviceCount();
        if (device_count <= 0) {
            ROS_WARN("[cuda] No CUDA-enabled device found.");
            return false;
        }

        cv::cuda::setDevice(0);
        cv::cuda::DeviceInfo device(0);
        ROS_INFO("[cuda] device=%s compute=%d.%d compatible=%d",
            device.name(), device.majorVersion(), device.minorVersion(),
            device.isCompatible() ? 1 : 0);
        if (!device.isCompatible()) {
            ROS_WARN("[cuda] OpenCV CUDA binaries do not support compute %d.%d; rebuild the GPU image with the matching CUDA_ARCH.",
                device.majorVersion(), device.minorVersion());
            return false;
        }

        stage = "host test data";
        cv::Mat host(32, 32, CV_8UC1, cv::Scalar(0));
        cv::Mat map_x(host.size(), CV_32FC1);
        cv::Mat map_y(host.size(), CV_32FC1);
        for (int y = 0; y < host.rows; ++y) {
            for (int x = 0; x < host.cols; ++x) {
                map_x.at<float>(y, x) = (float)x;
                map_y.at<float>(y, x) = (float)y;
            }
        }

        cv::cuda::GpuMat gpu;
        cv::cuda::GpuMat map_x_gpu;
        cv::cuda::GpuMat map_y_gpu;
        cv::cuda::GpuMat remapped;
        cv::cuda::GpuMat pyramid;
        cv::cuda::GpuMat corners;

        stage = "upload";
        gpu.upload(host);
        map_x_gpu.upload(map_x);
        map_y_gpu.upload(map_y);
        stage = "remap";
        cv::cuda::remap(gpu, remapped, map_x_gpu, map_y_gpu, cv::INTER_LINEAR);
        stage = "pyrDown";
        cv::cuda::pyrDown(remapped, pyramid);

        stage = "GFTT";
        cv::Ptr<cv::cuda::CornersDetector> detector =
            cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1, 20, 0.01, 3.0);
        detector->detect(remapped, corners);

        stage = "PyrLK";
        std::vector<cv::cuda::GpuMat> test_pyramid{remapped, pyramid};
        std::vector<cv::Point2f> host_points{cv::Point2f(16.0f, 16.0f)};
        cv::cuda::GpuMat previous_points(host_points);
        cv::cuda::GpuMat current_points(host_points);
        cv::cuda::GpuMat status;
        cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> optical_flow =
            cv::cuda::SparsePyrLKOpticalFlow::create(
                cv::Size(7, 7), 1, 10, true);
        optical_flow->calc(test_pyramid, test_pyramid, previous_points,
            current_points, status, cv::noArray());
        cv::cuda::Stream::Null().waitForCompletion();
        return true;
    } catch (const cv::Exception &e) {
        ROS_WARN_STREAM("[cuda] Smoke test failed at " << stage << ": " << e.what());
        return false;
    }
#else
    return false;
#endif
}

static void publishMatImage(const ros::Publisher &pub, const ros::Time &stamp,
        const std::string &frame_id, const cv::Mat &image)
{
    if (pub.getNumSubscribers() == 0 || image.empty())
        return;

    std_msgs::Header header;
    header.stamp = stamp;
    header.frame_id = frame_id;
    const std::string encoding = image.channels() == 1 ? "mono8" : "bgr8";
    pub.publish(cv_bridge::CvImage(header, encoding, image).toImageMsg());
}

static cv::Mat sideBySideImage(const cv::Mat &left, const cv::Mat &right)
{
    if (left.empty())
        return right.clone();
    if (right.empty())
        return left.clone();

    cv::Mat l = left;
    cv::Mat r = right;
    cv::Mat l_view, r_view;

    if (l.channels() != r.channels()) {
        if (l.channels() == 1)
            cv::cvtColor(l, l_view, cv::COLOR_GRAY2BGR);
        else
            l_view = l;
        if (r.channels() == 1)
            cv::cvtColor(r, r_view, cv::COLOR_GRAY2BGR);
        else
            r_view = r;
    } else {
        l_view = l;
        r_view = r;
    }

    cv::Mat r_resized;
    if (r_view.rows != l_view.rows) {
        const double scale = (double)l_view.rows / (double)r_view.rows;
        cv::resize(r_view, r_resized, cv::Size((int)(r_view.cols * scale), l_view.rows));
    } else {
        r_resized = r_view;
    }

    cv::Mat out;
    cv::hconcat(l_view, r_resized, out);
    return out;
}

static void publishStereoMatImage(const ros::Publisher &pub, const ros::Time &stamp,
        const std::string &frame_id, const cv::Mat &left, const cv::Mat &right)
{
    if (pub.getNumSubscribers() == 0)
        return;
    publishMatImage(pub, stamp, frame_id, sideBySideImage(left, right));
}

FisheyeFlattenHandler::FisheyeFlattenHandler(ros::NodeHandle & n, bool _is_color): mask_up(5, 0), mask_down(5, 0), is_color(_is_color)
{

    readIntrinsicParameter(CAM_NAMES);

    flatten_pub = n.advertise<fishloop_vins::FlattenImages>("/vins_estimator/flattened_raw", 1);
    flatten_gray_pub = n.advertise<fishloop_vins::FlattenImages>("/vins_estimator/flattened_gray", 1);
    raw_left_pub = n.advertise<sensor_msgs::Image>("/vins_estimator/fisheye/left/image_raw", 1);
    raw_right_pub = n.advertise<sensor_msgs::Image>("/vins_estimator/fisheye/right/image_raw", 1);
    raw_stereo_pub = n.advertise<sensor_msgs::Image>("/vins_estimator/fisheye/stereo/image_raw", 1);
    undist_left_pub = n.advertise<sensor_msgs::Image>("/vins_estimator/fisheye/left/image_undistorted", 1);
    undist_right_pub = n.advertise<sensor_msgs::Image>("/vins_estimator/fisheye/right/image_undistorted", 1);
    undist_stereo_pub = n.advertise<sensor_msgs::Image>("/vins_estimator/fisheye/stereo/image_undistorted", 1);

    if (fisheys_undists.size() >= 2) {
        fisheys_undists[0].initFullTopUndistortRectifyMap(full_top_map_l_1, full_top_map_l_2);
        fisheys_undists[1].initFullTopUndistortRectifyMap(full_top_map_r_1, full_top_map_r_2);
        ROS_INFO("[VINS-DBG][full_undist] size=%dx%d f=%.3f cx=%.1f cy=%.1f",
            fisheys_undists[0].rawSize().width,
            fisheys_undists[0].rawSize().height,
            fisheys_undists[0].fullTopFocal(),
            fisheys_undists[0].raw_width / 2.0,
            fisheys_undists[0].raw_height / 2.0);
    }

    if (enable_up_top) {
        mask_up[0] = true;        
    }

    if (enable_down_top) {
        mask_down[0] = true;
    }

    if (enable_up_side) {
        mask_up[1] = true;
        mask_up[2] = true;
        mask_up[3] = true;
    }

    if(enable_rear_side) {
        mask_up[4] = true;
        mask_down[4] = true;
    }

    if (enable_down_side) {
        mask_down[1] = true;
        mask_down[2] = true;
        mask_down[3] = true;
    }
}



int FisheyeFlattenHandler::raw_height() {
    return fisheys_undists[0].raw_height;
}

int FisheyeFlattenHandler::raw_width() {
    return fisheys_undists[0].raw_width;
}


void FisheyeFlattenHandler::imgs_callback(const sensor_msgs::ImageConstPtr &img1_msg, const sensor_msgs::ImageConstPtr &img2_msg)
{
    auto img1 = getImageFromMsg(img1_msg);
    auto img2 = getImageFromMsg(img2_msg);
    stamp = img1_msg->header.stamp;
    imgs_callback(img1_msg->header.stamp.toSec(), img1->image, img2->image);
}

void FisheyeFlattenHandler::imgs_callback(double t, const cv::Mat & img1, const cv::Mat img2, bool is_blank_init) {

    static double flatten_time_sum = 0;
    static double count = 0;

    count += 1;

    if (!is_blank_init) {
        ROS_INFO_THROTTLE(2.0,
            "[VINS-DBG][undist_in] cam0=%dx%d type=%d cam1=%dx%d type=%d t=%.3f",
            img1.cols, img1.rows, img1.type(),
            img2.cols, img2.rows, img2.type(), t);
    }

    TicToc t_f;
    const ros::Time img_stamp(t);
    cv::Mat full_left_undist_gray;

    if (!is_blank_init) {
        publishMatImage(raw_left_pub, img_stamp, "fisheye_left", img1);
        publishMatImage(raw_right_pub, img_stamp, "fisheye_right", img2);
        publishStereoMatImage(raw_stereo_pub, img_stamp, "fisheye_stereo", img1, img2);

        const bool need_full_undist =
            pub_keyframe_image.getNumSubscribers() > 0 ||
            undist_left_pub.getNumSubscribers() > 0 ||
            undist_right_pub.getNumSubscribers() > 0 ||
            undist_stereo_pub.getNumSubscribers() > 0;

        if (need_full_undist && !full_top_map_l_1.empty() && !full_top_map_r_1.empty()) {
            cv::Mat full_left_undist;
            cv::Mat full_right_undist;
            fisheys_undists[0].undistortFullTop(img1, full_left_undist, full_top_map_l_1, full_top_map_l_2);
            fisheys_undists[1].undistortFullTop(img2, full_right_undist, full_top_map_r_1, full_top_map_r_2);

            publishMatImage(undist_left_pub, img_stamp, "fisheye_left_undistorted", full_left_undist);
            publishMatImage(undist_right_pub, img_stamp, "fisheye_right_undistorted", full_right_undist);
            publishStereoMatImage(undist_stereo_pub, img_stamp, "fisheye_stereo_undistorted",
                full_left_undist, full_right_undist);

            if (full_left_undist.channels() == 3) {
                cv::cvtColor(full_left_undist, full_left_undist_gray, cv::COLOR_BGR2GRAY);
            } else {
                full_left_undist_gray = full_left_undist.clone();
            }
        }
    }

    if (USE_GPU) {
        // is_color = true;
        if (is_color) {
            fisheye_up_imgs_cuda = fisheys_undists[0].undist_all_cuda(img1, true, mask_up); 
            fisheye_down_imgs_cuda = fisheys_undists[1].undist_all_cuda(img2, true, mask_down);
            fisheye_up_imgs_cuda_gray.clear();
            fisheye_down_imgs_cuda_gray.clear();


            for (auto & img: fisheye_up_imgs_cuda) {
                cv::cuda::GpuMat gray;
                if(!img.empty()) {
                    cv::cuda::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
                }
                fisheye_up_imgs_cuda_gray.push_back(gray);
            }

            for (auto & img: fisheye_down_imgs_cuda) {
                cv::cuda::GpuMat gray;
                if(!img.empty()) {
                    cv::cuda::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
                }

                fisheye_down_imgs_cuda_gray.push_back(gray);
            }
        } else {
            fisheye_up_imgs_cuda_gray = fisheys_undists[0].undist_all_cuda(img1, false, mask_up); 
            fisheye_down_imgs_cuda_gray = fisheys_undists[1].undist_all_cuda(img2, false, mask_down);
        }

        if (!is_blank_init) {
            buf_lock.lock();
            fisheye_buf_t.push(t);
            fisheye_cuda_buf_up.push(fisheye_up_imgs_cuda_gray);
            fisheye_cuda_buf_down.push(fisheye_down_imgs_cuda_gray);
            if(is_color) {
                fisheye_cuda_buf_up_color.push(fisheye_up_imgs_cuda);
                fisheye_cuda_buf_down_color.push(fisheye_down_imgs_cuda);
            }
            full_left_undist_gray_buf.push(full_left_undist_gray);
            size_t dropped = 0;
            while (fisheye_buf_t.size() > MAX_FISHEYE_BUFFER_SIZE) {
                fisheye_buf_t.pop();
                popIfNotEmpty(fisheye_cuda_buf_up);
                popIfNotEmpty(fisheye_cuda_buf_down);
                popIfNotEmpty(full_left_undist_gray_buf);
                if (is_color) {
                    popIfNotEmpty(fisheye_cuda_buf_up_color);
                    popIfNotEmpty(fisheye_cuda_buf_down_color);
                }
                dropped++;
            }
            if (dropped > 0) {
                ROS_WARN_THROTTLE(2.0, "[fisheye_buffer] dropped %zu stale frames; processing is behind input", dropped);
            }
            buf_lock.unlock();
        }
    } else {
        if (is_color) {
            fisheys_undists[0].stereo_flatten(img1, img2, &fisheys_undists[1], 
                fisheye_up_imgs, fisheye_down_imgs, true, 
                enable_up_top, enable_rear_side, enable_down_top, enable_rear_side);
            fisheye_up_imgs_gray.clear();
            fisheye_down_imgs_gray.clear();
            for (auto & img: fisheye_up_imgs) {
                cv::Mat gray;
                if(!img.empty()) {
                    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
                }
                fisheye_up_imgs_gray.push_back(gray);
            }

            for (auto & img: fisheye_down_imgs) {
                cv::Mat gray;
                if(!img.empty()) {
                    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
                }
                fisheye_down_imgs_gray.push_back(gray);
            }
        } else {
            fisheys_undists[0].stereo_flatten(img1, img2, &fisheys_undists[1], 
                fisheye_up_imgs_gray, fisheye_down_imgs_gray, false, 
                enable_up_top, enable_rear_side, enable_down_top, enable_rear_side);
        }

        buf_lock.lock();
        fisheye_buf_t.push(t);
        fisheye_buf_up.push(fisheye_up_imgs_gray);
        fisheye_buf_down.push(fisheye_down_imgs_gray);

        if (is_color) {
            fisheye_buf_up_color.push(fisheye_up_imgs);
            fisheye_buf_down_color.push(fisheye_down_imgs);
        }
        full_left_undist_gray_buf.push(full_left_undist_gray);
        size_t dropped = 0;
        while (fisheye_buf_t.size() > MAX_FISHEYE_BUFFER_SIZE) {
            fisheye_buf_t.pop();
            popIfNotEmpty(fisheye_buf_up);
            popIfNotEmpty(fisheye_buf_down);
            popIfNotEmpty(full_left_undist_gray_buf);
            if (is_color) {
                popIfNotEmpty(fisheye_buf_up_color);
                popIfNotEmpty(fisheye_buf_down_color);
            }
            dropped++;
        }
        if (dropped > 0) {
            ROS_WARN_THROTTLE(2.0, "[fisheye_buffer] dropped %zu stale frames; processing is behind input", dropped);
        }

        buf_lock.unlock();
    }

    double tf = t_f.toc();
    if (ENABLE_PERF_OUTPUT) {
        ROS_INFO("img_callback cost %fms flatten %fms Flatten AVG %fms", t_f.toc(), tf, flatten_time_sum/count);
    }

    flatten_time_sum += t_f.toc();
}

bool FisheyeFlattenHandler::has_image_in_buffer() {
    std::lock_guard<std::mutex> lock(buf_lock);
    return !fisheye_buf_t.empty();
}

double FisheyeFlattenHandler::pop_from_buffer(
            CvCudaImages & up_gray, CvCudaImages & down_gray,
            CvCudaImages & up_color, CvCudaImages & down_color,
            cv::Mat *full_left_undist_gray) {
    if (!USE_GPU)
        return -1;
    std::lock_guard<std::mutex> lock(buf_lock);
    if (fisheye_buf_t.empty())
        return -1;
    if (fisheye_cuda_buf_up.empty() || fisheye_cuda_buf_down.empty() ||
        (is_color && (fisheye_cuda_buf_up_color.empty() ||
                      fisheye_cuda_buf_down_color.empty()))) {
        ROS_ERROR_THROTTLE(2.0, "[fisheye_buffer] inconsistent CUDA queues");
        return -1;
    }

    const double t = fisheye_buf_t.front();
    up_gray = fisheye_cuda_buf_up.front();
    down_gray = fisheye_cuda_buf_down.front();
    if (full_left_undist_gray && !full_left_undist_gray_buf.empty())
        *full_left_undist_gray = full_left_undist_gray_buf.front();
    if (is_color) {
        up_color = fisheye_cuda_buf_up_color.front();
        down_color = fisheye_cuda_buf_down_color.front();
    }

    fisheye_buf_t.pop();
    fisheye_cuda_buf_up.pop();
    fisheye_cuda_buf_down.pop();
    if (!full_left_undist_gray_buf.empty())
        full_left_undist_gray_buf.pop();
    if (is_color) {
        fisheye_cuda_buf_up_color.pop();
        fisheye_cuda_buf_down_color.pop();
    }
    return t;
}

double FisheyeFlattenHandler::pop_from_buffer(
            CvImages & up_gray, CvImages & down_gray,
            CvImages & up_color, CvImages & down_color,
            cv::Mat *full_left_undist_gray) {
    if (USE_GPU)
        return -1;
    std::lock_guard<std::mutex> lock(buf_lock);
    if (fisheye_buf_t.empty())
        return -1;
    if (fisheye_buf_up.empty() || fisheye_buf_down.empty() ||
        (is_color && (fisheye_buf_up_color.empty() ||
                      fisheye_buf_down_color.empty()))) {
        ROS_ERROR_THROTTLE(2.0, "[fisheye_buffer] inconsistent CPU queues");
        return -1;
    }

    const double t = fisheye_buf_t.front();
    up_gray = fisheye_buf_up.front();
    down_gray = fisheye_buf_down.front();
    if (full_left_undist_gray && !full_left_undist_gray_buf.empty())
        *full_left_undist_gray = full_left_undist_gray_buf.front();
    if (is_color) {
        up_color = fisheye_buf_up_color.front();
        down_color = fisheye_buf_down_color.front();
    }

    fisheye_buf_t.pop();
    fisheye_buf_up.pop();
    fisheye_buf_down.pop();
    if (!full_left_undist_gray_buf.empty())
        full_left_undist_gray_buf.pop();
    if (is_color) {
        fisheye_buf_up_color.pop();
        fisheye_buf_down_color.pop();
    }
    return t;
}

void FisheyeFlattenHandler::setup_extrinsic(fishloop_vins::FlattenImages & images, const Estimator & estimator) {
    static Eigen::Quaterniond t_left = Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d(1, 0, 0)));
    static Eigen::Quaterniond t_down = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d(1, 0, 0)));

    std::vector<Eigen::Quaterniond> t_dirs;
    t_dirs.push_back(Eigen::Quaterniond::Identity());
    t_dirs.push_back(t_left);
    for (unsigned int i = 0; i < 3; i ++) {
        t_dirs.push_back(t_dirs.back() * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0)));
    }

    for (unsigned int i = 0; i < 4; i ++) {
        images.extrinsic_up_cams.push_back(
            pose_from_PQ(estimator.tic[0], Eigen::Quaterniond(estimator.ric[0])*t_dirs[i])
        );
        images.extrinsic_down_cams.push_back(
            pose_from_PQ(estimator.tic[1], Eigen::Quaterniond(estimator.ric[1])*t_down*t_dirs[i])
        );
    }
}

void FisheyeFlattenHandler::pack_and_send(ros::Time stamp, 
        cv::InputArray fisheye_up_imgs, cv::InputArray fisheye_down_imgs, 
        cv::InputArray fisheye_up_imgs_gray, cv::InputArray fisheye_down_imgs_gray, 
        const Estimator & estimator) {
    TicToc t_p;
    fishloop_vins::FlattenImages images;
    fishloop_vins::FlattenImages images_gray;
    static double pack_send_time = 0;

    setup_extrinsic(images, estimator);
    setup_extrinsic(images_gray, estimator);

    images.header.stamp = stamp;
    images_gray.header.stamp = stamp;
    static int count = 0;
    count ++;

    CvCudaImages fisheye_up_imgs_cuda, fisheye_down_imgs_cuda;
    CvCudaImages fisheye_up_imgs_cuda_gray, fisheye_down_imgs_cuda_gray;
    if (USE_GPU) {
        fisheye_up_imgs.getGpuMatVector(fisheye_up_imgs_cuda);
        fisheye_down_imgs.getGpuMatVector(fisheye_down_imgs_cuda);
        fisheye_up_imgs_gray.getGpuMatVector(fisheye_up_imgs_cuda_gray);
        fisheye_down_imgs_gray.getGpuMatVector(fisheye_down_imgs_cuda_gray);
        // std::cout << "fisheye_up_imgs_cuda_gray size: " << fisheye_up_imgs_cuda_gray.size() << std::endl;

    }

    size_t _size = fisheye_up_imgs_cuda.size();
    
    if (!is_color) {
        _size = fisheye_up_imgs_cuda_gray.size();
    }

    for (unsigned int i = 0; i < _size; i++) {
        cv_bridge::CvImage outImg;
        cv_bridge::CvImage outImg_gray;

        outImg.encoding = "8UC3";
        outImg_gray.encoding = "mono8";
        outImg.header = images.header;
        outImg_gray.header = images.header;
        TicToc to;
        
        if (USE_GPU) {
            if (is_color)
                fisheye_up_imgs_cuda[i].download(outImg.image);
            // std::cout << "Sending image " << i << " size " << fisheye_up_imgs_cuda_gray[i].size() << std::endl;
            fisheye_up_imgs_cuda_gray[i].download(outImg_gray.image);
        } else {
            if (is_color)
                outImg.image = fisheye_up_imgs.getMat(i);
            outImg_gray.image = fisheye_up_imgs_gray.getMat(i);
        }
        
        if (is_color) {
            images.up_cams.push_back(*outImg.toImageMsg());
        }

        images_gray.up_cams.push_back(*outImg_gray.toImageMsg());
    }

    for (unsigned int i = 0; i < _size; i++) {
        cv_bridge::CvImage outImg;
        cv_bridge::CvImage outImg_gray;

        outImg.encoding = "8UC3";
        outImg_gray.encoding = "mono8";

        if (USE_GPU) {
            if (is_color) {
                fisheye_down_imgs_cuda[i].download(outImg.image);
            }
            fisheye_down_imgs_cuda_gray[i].download(outImg_gray.image);
        } else {
            if (is_color) {
                outImg.image = fisheye_down_imgs.getMat(i);
            }
            outImg_gray.image = fisheye_down_imgs_gray.getMat(i);
        }
        
        if (is_color) {
            images.down_cams.push_back(*outImg.toImageMsg());
        }

        images_gray.down_cams.push_back(*outImg_gray.toImageMsg());
    }

    if (is_color) {
        flatten_pub.publish(images);
    }

    flatten_gray_pub.publish(images_gray);
    pack_send_time += t_p.toc();

    if (ENABLE_PERF_OUTPUT) {
        ROS_INFO("Pack and send AVG %fms this %fms", pack_send_time/count, t_p.toc());
    }
}


void FisheyeFlattenHandler::readIntrinsicParameter(const vector<string> &calib_file)
{
    for (size_t i = 0; i < calib_file.size(); i++)
    {
        if (FISHEYE) {
            ROS_INFO("Flatten read fisheye %s, id %ld", calib_file[i].c_str(), i);
            FisheyeUndist un(calib_file[i].c_str(), i, FISHEYE_FOV, USE_GPU, WIDTH);
            fisheys_undists.push_back(un);
        }
    }
}


void VinsNodeBaseClass::pack_and_send_thread(const ros::TimerEvent & e) {               
    if (need_to_pack_and_send && cur_frame_t > t_last_send) {
        //need to pack and send
        pack_and_send_mtx.lock();
        t_last_send = cur_frame_t;
        need_to_pack_and_send = false;
        if (USE_GPU) {
            fisheye_handler->pack_and_send(ros::Time(t_last_send), 
                cur_up_color_cuda, cur_down_color_cuda,
                cur_up_gray_cuda, cur_down_gray_cuda,
                estimator);
        } else {
                fisheye_handler->pack_and_send(ros::Time(t_last_send), 
                cur_up_color, cur_down_color,
                cur_up_gray, cur_down_gray,
                estimator);
        }
        pack_and_send_mtx.unlock();
    }
}

void VinsNodeBaseClass::processFlattened(const ros::TimerEvent & e) {
    TicToc t0;
    if (fisheye_handler->has_image_in_buffer()) {
        pack_and_send_mtx.lock();
        cv::Mat full_left_undist_gray;

        if (USE_GPU) {
            cur_frame_t = fisheye_handler->pop_from_buffer(
                cur_up_gray_cuda,
                cur_down_gray_cuda,
                cur_up_color_cuda,
                cur_down_color_cuda,
                &full_left_undist_gray
            );

            bool is_odometry_frame = estimator.is_next_odometry_frame();

            if (is_odometry_frame) {
                need_to_pack_and_send = true;
            }
            if (!full_left_undist_gray.empty()) {
                estimator.cacheKeyframeImage(cur_frame_t, full_left_undist_gray);
            }
            estimator.inputFisheyeImage(cur_frame_t, cur_up_gray_cuda, cur_down_gray_cuda);
        } else {
            cur_frame_t = fisheye_handler->pop_from_buffer(
                cur_up_gray,
                cur_down_gray,
                cur_up_color,
                cur_down_color,
                &full_left_undist_gray
            );

            bool is_odometry_frame = estimator.is_next_odometry_frame();

            if (is_odometry_frame) {
                need_to_pack_and_send = true;
            }
            ROS_INFO_THROTTLE(2.0,
                "[VINS-DBG][to_estimator] up_gray=%zu sub-imgs (e.g. %dx%d) "
                "down_gray=%zu sub-imgs (e.g. %dx%d) odom_frame=%d t=%.3f",
                cur_up_gray.size(),
                cur_up_gray.empty()   ? 0 : cur_up_gray[0].cols,
                cur_up_gray.empty()   ? 0 : cur_up_gray[0].rows,
                cur_down_gray.size(),
                cur_down_gray.empty() ? 0 : cur_down_gray[0].cols,
                cur_down_gray.empty() ? 0 : cur_down_gray[0].rows,
                int(is_odometry_frame), cur_frame_t);
            if (!full_left_undist_gray.empty()) {
                estimator.cacheKeyframeImage(cur_frame_t, full_left_undist_gray);
            }
            estimator.inputFisheyeImage(cur_frame_t, cur_up_gray, cur_down_gray);
        }
        double t_0 = t0.toc();
        //Need to wait for pack and send to endft
        pack_and_send_mtx.unlock();

        if(ENABLE_PERF_OUTPUT) {
            ROS_INFO("[processFlattened]Input Image: %fms, whole %fms", t_0, t0.toc());
        }

    }
}

void VinsNodeBaseClass::fisheye_imgs_callback(const sensor_msgs::ImageConstPtr &img1_msg, const sensor_msgs::ImageConstPtr &img2_msg) {
    TicToc tic_input;
    ROS_INFO_THROTTLE(2.0,
        "[VINS-DBG][raw_cb] cam0=%dx%d enc=%s cam1=%dx%d enc=%s t=%.3f",
        img1_msg->width, img1_msg->height, img1_msg->encoding.c_str(),
        img2_msg->width, img2_msg->height, img2_msg->encoding.c_str(),
        img1_msg->header.stamp.toSec());
    fisheye_handler->imgs_callback(img1_msg, img2_msg);

    if (img1_msg->header.stamp.toSec() - t_last > 0.11) {
        ROS_WARN("Duration between two images is %fms", img1_msg->header.stamp.toSec() - t_last);
    }
    t_last = img1_msg->header.stamp.toSec();
}

void VinsNodeBaseClass::fisheye_comp_imgs_callback(const sensor_msgs::CompressedImageConstPtr &img1_msg, const sensor_msgs::CompressedImageConstPtr &img2_msg) {
    TicToc tic_input;
    auto img1 = getImageFromMsg(img1_msg);
    auto img2 = getImageFromMsg(img2_msg);
    ROS_INFO_THROTTLE(2.0,
        "[VINS-DBG][comp_cb] cam0=%dx%d cam1=%dx%d t=%.3f bytes=(%zu,%zu)",
        img1.cols, img1.rows, img2.cols, img2.rows,
        img1_msg->header.stamp.toSec(),
        img1_msg->data.size(), img2_msg->data.size());

    fisheye_handler->imgs_callback(img1_msg->header.stamp.toSec(), img1, img2);

    if (img1_msg->header.stamp.toSec() - t_last > 0.11) {
        ROS_WARN("Duration between two images is %fms", img1_msg->header.stamp.toSec() - t_last);
    }
    t_last = img1_msg->header.stamp.toSec();
}

void VinsNodeBaseClass::imgs_callback(const sensor_msgs::ImageConstPtr &img1_msg, const sensor_msgs::ImageConstPtr &img2_msg)
{
    auto img1 = getImageFromMsg(img1_msg);
    auto img2 = getImageFromMsg(img2_msg);
    estimator.inputImage(img1_msg->header.stamp.toSec(), img1->image, img2->image);
}


void VinsNodeBaseClass::comp_imgs_callback(const sensor_msgs::CompressedImageConstPtr &img1_msg, const sensor_msgs::CompressedImageConstPtr &img2_msg)
{
    auto img1 = getImageFromMsg(img1_msg);
    auto img2 = getImageFromMsg(img2_msg);
    estimator.inputImage(img1_msg->header.stamp.toSec(), img1, img2);
}

void VinsNodeBaseClass::imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);
    estimator.inputIMU(t, acc, gyr);
}

void VinsNodeBaseClass::restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        estimator.resetRuntimeState();
    }
    return;
}


void VinsNodeBaseClass::Init(ros::NodeHandle & n)
{
    std::string config_file;
    n.getParam("config_file", config_file);
    bool fisheye_external_flatten;
    n.getParam("fisheye_external_flatten", fisheye_external_flatten);
    
    std::cout << "config file is " << config_file << '\n';
    readParameters(config_file);

    if (USE_GPU && !cudaPipelineSmokeTest()) {
        ROS_WARN("Configured use_gpu=1, but this OpenCV/CUDA build cannot run on the current device. Falling back to CPU/OpenMP.");
        USE_GPU = 0;
    }

    estimator.setParameter();

    ROS_INFO("Will %d GPU", USE_GPU);
    if (ENABLE_DEPTH) {
        FisheyeUndist *fun = nullptr;
        if (USE_GPU) {
            auto ft = (BaseFisheyeFeatureTracker<cv::cuda::GpuMat> *)
                estimator.featureTracker;
            fun = ft->get_fisheye_undist(0);
        } else {
            auto ft = (BaseFisheyeFeatureTracker<cv::Mat> *)
                estimator.featureTracker;
            fun = ft->get_fisheye_undist(0);
        }

        cam_manager = new DepthCamManager(n, fun);
        cam_manager -> init_with_extrinsic(estimator.ric[0], estimator.tic[0], estimator.ric[1], estimator.tic[1]);
        estimator.depth_cam_manager = cam_manager;
    }
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    if (FISHEYE) {
        fisheye_handler = new FisheyeFlattenHandler(n, FLATTEN_COLOR);
    }

    //We use blank images to initialize cuda before every thing
    if (USE_GPU) {
        TicToc blank;
        cv::Mat mat(fisheye_handler->raw_height(), fisheye_handler->raw_width(),
            CV_8UC3, cv::Scalar::all(0));
        fisheye_handler->imgs_callback(0, mat, mat, true);
            estimator.inputFisheyeImage(0, 
            fisheye_handler->fisheye_up_imgs_cuda_gray, fisheye_handler->fisheye_down_imgs_cuda_gray, true);
        std::cout<< "Initialize with blank cost" << blank.toc() << std::endl;
    }

    sub_imu = n.subscribe(IMU_TOPIC, 2000, &VinsNodeBaseClass::imu_callback, (VinsNodeBaseClass*)this, ros::TransportHints().tcpNoDelay(true));
    sub_restart = n.subscribe("/vins_restart", 100, &VinsNodeBaseClass::restart_callback, (VinsNodeBaseClass*)this, ros::TransportHints().tcpNoDelay(true));

    if (IS_COMP_IMAGES) {
        ROS_INFO("Will directly receive compressed images %s and %s", COMP_IMAGE0_TOPIC.c_str(), COMP_IMAGE1_TOPIC.c_str());
        comp_image_sub_l = new message_filters::Subscriber<sensor_msgs::CompressedImage> (n, COMP_IMAGE0_TOPIC, STEREO_SYNC_QUEUE_SIZE, ros::TransportHints().tcpNoDelay(true));
        comp_image_sub_r = new message_filters::Subscriber<sensor_msgs::CompressedImage> (n, COMP_IMAGE1_TOPIC, STEREO_SYNC_QUEUE_SIZE, ros::TransportHints().tcpNoDelay(true));
        comp_sync = new message_filters::TimeSynchronizer<sensor_msgs::CompressedImage, sensor_msgs::CompressedImage> (*comp_image_sub_l, *comp_image_sub_r, STEREO_SYNC_QUEUE_SIZE);
        if (FISHEYE) {
            comp_sync->registerCallback(boost::bind(&VinsNodeBaseClass::fisheye_comp_imgs_callback, (VinsNodeBaseClass*)this, _1, _2));
        } else {    
            comp_sync->registerCallback(boost::bind(&VinsNodeBaseClass::comp_imgs_callback, (VinsNodeBaseClass*)this, _1, _2));
        }
    } else {
        ROS_INFO("Will directly receive raw images %s and %s", IMAGE0_TOPIC.c_str(), IMAGE1_TOPIC.c_str());
        image_sub_l = new message_filters::Subscriber<sensor_msgs::Image> (n, IMAGE0_TOPIC, STEREO_SYNC_QUEUE_SIZE, ros::TransportHints().tcpNoDelay(true));
        image_sub_r = new message_filters::Subscriber<sensor_msgs::Image> (n, IMAGE1_TOPIC, STEREO_SYNC_QUEUE_SIZE, ros::TransportHints().tcpNoDelay(true));
        sync = new message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> (*image_sub_l, *image_sub_r, STEREO_SYNC_QUEUE_SIZE);
        if (FISHEYE) {
            sync->registerCallback(boost::bind(&VinsNodeBaseClass::fisheye_imgs_callback, (VinsNodeBaseClass*)this, _1, _2));
        } else {    
            sync->registerCallback(boost::bind(&VinsNodeBaseClass::imgs_callback, (VinsNodeBaseClass*)this, _1, _2));
        }
    }


    timer1 = n.createTimer(ros::Duration(0.004), boost::bind(&VinsNodeBaseClass::processFlattened, (VinsNodeBaseClass*)this, _1 ));
    if (PUB_FLATTEN) {
        timer2 = n.createTimer(ros::Duration(1/PUB_FLATTEN_FREQ), boost::bind(&VinsNodeBaseClass::pack_and_send_thread, (VinsNodeBaseClass*)this, _1 ));
    }
}
