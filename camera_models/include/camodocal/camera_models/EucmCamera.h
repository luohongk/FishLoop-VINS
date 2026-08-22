#ifndef EUCM_CAMERA_H
#define EUCM_CAMERA_H

#include <opencv2/core/core.hpp>
#include <string>

#include "Camera.h"

namespace camodocal
{

// Enhanced Unified Camera Model as used by Kalibr:
// intrinsics = [alpha, beta, fx, fy, cx, cy].
class EucmCamera: public Camera
{
public:
    class Parameters: public Camera::Parameters
    {
    public:
        Parameters();
        Parameters(const std::string& cameraName, int w, int h,
                   double alpha, double beta,
                   double fx, double fy, double cx, double cy);

        double& alpha();
        double& beta();
        double& fx();
        double& fy();
        double& cx();
        double& cy();

        double alpha() const;
        double beta() const;
        double fx() const;
        double fy() const;
        double cx() const;
        double cy() const;

        bool readFromYamlFile(const std::string& filename);
        void writeToYamlFile(const std::string& filename) const;

        Parameters& operator=(const Parameters& other);
        friend std::ostream& operator<<(std::ostream& out, const Parameters& params);

    private:
        double m_alpha;
        double m_beta;
        double m_fx;
        double m_fy;
        double m_cx;
        double m_cy;
    };

    EucmCamera();
    explicit EucmCamera(const Parameters& params);
    EucmCamera(const std::string& cameraName, int imageWidth, int imageHeight,
               double alpha, double beta,
               double fx, double fy, double cx, double cy);

    Camera::ModelType modelType() const;
    const std::string& cameraName() const;
    int imageWidth() const;
    int imageHeight() const;

    void estimateIntrinsics(const cv::Size& boardSize,
                            const std::vector<std::vector<cv::Point3f> >& objectPoints,
                            const std::vector<std::vector<cv::Point2f> >& imagePoints);

    void liftSphere(const Eigen::Vector2d& p, Eigen::Vector3d& P) const;
    void liftProjective(const Eigen::Vector2d& p, Eigen::Vector3d& P) const;
    void spaceToPlane(const Eigen::Vector3d& P, Eigen::Vector2d& p) const;
    void undistToPlane(const Eigen::Vector2d& p_u, Eigen::Vector2d& p) const;

    cv::Mat initUndistortRectifyMap(cv::Mat& map1, cv::Mat& map2,
                                    float fx = -1.0f, float fy = -1.0f,
                                    cv::Size imageSize = cv::Size(0, 0),
                                    float cx = -1.0f, float cy = -1.0f,
                                    cv::Mat rmat = cv::Mat::eye(3, 3, CV_32F)) const;

    int parameterCount() const;
    const Parameters& getParameters() const;
    void setParameters(const Parameters& parameters);
    void readParameters(const std::vector<double>& parameters);
    void writeParameters(std::vector<double>& parameters) const;
    void writeParametersToYamlFile(const std::string& filename) const;
    std::string parametersToString() const;

private:
    Parameters mParameters;
    double m_inv_fx;
    double m_inv_fy;
};

typedef boost::shared_ptr<EucmCamera> EucmCameraPtr;
typedef boost::shared_ptr<const EucmCamera> EucmCameraConstPtr;

}

#endif
