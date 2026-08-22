#include "camodocal/camera_models/EucmCamera.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>

#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace camodocal
{

EucmCamera::Parameters::Parameters()
 : Camera::Parameters(EUCM)
 , m_alpha(0.5)
 , m_beta(1.0)
 , m_fx(1.0)
 , m_fy(1.0)
 , m_cx(0.0)
 , m_cy(0.0)
{
}

EucmCamera::Parameters::Parameters(const std::string& cameraName, int w, int h,
                                   double alpha, double beta,
                                   double fx, double fy, double cx, double cy)
 : Camera::Parameters(EUCM, cameraName, w, h)
 , m_alpha(alpha)
 , m_beta(beta)
 , m_fx(fx)
 , m_fy(fy)
 , m_cx(cx)
 , m_cy(cy)
{
}

double& EucmCamera::Parameters::alpha() { return m_alpha; }
double& EucmCamera::Parameters::beta() { return m_beta; }
double& EucmCamera::Parameters::fx() { return m_fx; }
double& EucmCamera::Parameters::fy() { return m_fy; }
double& EucmCamera::Parameters::cx() { return m_cx; }
double& EucmCamera::Parameters::cy() { return m_cy; }
double EucmCamera::Parameters::alpha() const { return m_alpha; }
double EucmCamera::Parameters::beta() const { return m_beta; }
double EucmCamera::Parameters::fx() const { return m_fx; }
double EucmCamera::Parameters::fy() const { return m_fy; }
double EucmCamera::Parameters::cx() const { return m_cx; }
double EucmCamera::Parameters::cy() const { return m_cy; }

bool EucmCamera::Parameters::readFromYamlFile(const std::string& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
        return false;

    std::string modelType;
    fs["model_type"] >> modelType;
    if (modelType != "EUCM" && modelType != "eucm")
        return false;

    m_modelType = EUCM;
    fs["camera_name"] >> m_cameraName;
    m_imageWidth = static_cast<int>(fs["image_width"]);
    m_imageHeight = static_cast<int>(fs["image_height"]);

    const cv::FileNode n = fs["projection_parameters"];
    m_alpha = static_cast<double>(n["alpha"]);
    m_beta = static_cast<double>(n["beta"]);
    m_fx = static_cast<double>(n["fx"]);
    m_fy = static_cast<double>(n["fy"]);
    m_cx = static_cast<double>(n["cx"]);
    m_cy = static_cast<double>(n["cy"]);
    return true;
}

void EucmCamera::Parameters::writeToYamlFile(const std::string& filename) const
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "model_type" << "EUCM";
    fs << "camera_name" << m_cameraName;
    fs << "image_width" << m_imageWidth;
    fs << "image_height" << m_imageHeight;
    fs << "projection_parameters";
    fs << "{" << "alpha" << m_alpha
              << "beta" << m_beta
              << "fx" << m_fx
              << "fy" << m_fy
              << "cx" << m_cx
              << "cy" << m_cy << "}";
}

EucmCamera::Parameters& EucmCamera::Parameters::operator=(const Parameters& other)
{
    if (this != &other)
    {
        m_modelType = other.m_modelType;
        m_cameraName = other.m_cameraName;
        m_imageWidth = other.m_imageWidth;
        m_imageHeight = other.m_imageHeight;
        m_alpha = other.m_alpha;
        m_beta = other.m_beta;
        m_fx = other.m_fx;
        m_fy = other.m_fy;
        m_cx = other.m_cx;
        m_cy = other.m_cy;
    }
    return *this;
}

std::ostream& operator<<(std::ostream& out, const EucmCamera::Parameters& params)
{
    out << "Camera Parameters:\n"
        << "    model_type EUCM\n"
        << "    camera_name " << params.m_cameraName << "\n"
        << "    image_width " << params.m_imageWidth << "\n"
        << "    image_height " << params.m_imageHeight << "\n"
        << "    alpha " << params.m_alpha << "\n"
        << "    beta " << params.m_beta << "\n"
        << "    fx " << params.m_fx << "\n"
        << "    fy " << params.m_fy << "\n"
        << "    cx " << params.m_cx << "\n"
        << "    cy " << params.m_cy << std::endl;
    return out;
}

EucmCamera::EucmCamera()
 : m_inv_fx(1.0)
 , m_inv_fy(1.0)
{
}

EucmCamera::EucmCamera(const Parameters& params)
{
    setParameters(params);
}

EucmCamera::EucmCamera(const std::string& cameraName, int imageWidth, int imageHeight,
                       double alpha, double beta,
                       double fx, double fy, double cx, double cy)
 : mParameters(cameraName, imageWidth, imageHeight, alpha, beta, fx, fy, cx, cy)
 , m_inv_fx(1.0 / fx)
 , m_inv_fy(1.0 / fy)
{
}

Camera::ModelType EucmCamera::modelType() const { return mParameters.modelType(); }
const std::string& EucmCamera::cameraName() const { return mParameters.cameraName(); }
int EucmCamera::imageWidth() const { return mParameters.imageWidth(); }
int EucmCamera::imageHeight() const { return mParameters.imageHeight(); }

void EucmCamera::estimateIntrinsics(const cv::Size&,
                                    const std::vector<std::vector<cv::Point3f> >&,
                                    const std::vector<std::vector<cv::Point2f> >&)
{
    std::cerr << "EUCM intrinsic estimation is not implemented; use Kalibr parameters." << std::endl;
}

void EucmCamera::liftSphere(const Eigen::Vector2d& p, Eigen::Vector3d& P) const
{
    liftProjective(p, P);
    P.normalize();
}

void EucmCamera::liftProjective(const Eigen::Vector2d& p, Eigen::Vector3d& P) const
{
    const double mx = (p(0) - mParameters.cx()) * m_inv_fx;
    const double my = (p(1) - mParameters.cy()) * m_inv_fy;
    const double r2 = mx * mx + my * my;
    const double alpha = mParameters.alpha();
    const double beta = mParameters.beta();
    const double discriminant = std::max(0.0, 1.0 - (2.0 * alpha - 1.0) * beta * r2);
    const double denominator = alpha * std::sqrt(discriminant) + (1.0 - alpha);
    const double mz = (1.0 - beta * alpha * alpha * r2) /
                      std::max(denominator, 1e-12);
    P << mx, my, mz;
}

void EucmCamera::spaceToPlane(const Eigen::Vector3d& P, Eigen::Vector2d& p) const
{
    const double d = std::sqrt(mParameters.beta() * (P(0) * P(0) + P(1) * P(1)) +
                               P(2) * P(2));
    const double denominator = mParameters.alpha() * d +
                               (1.0 - mParameters.alpha()) * P(2);
    const double inv = std::abs(denominator) > 1e-12 ? 1.0 / denominator : 1e12;
    p << mParameters.fx() * P(0) * inv + mParameters.cx(),
         mParameters.fy() * P(1) * inv + mParameters.cy();
}

void EucmCamera::undistToPlane(const Eigen::Vector2d& p_u, Eigen::Vector2d& p) const
{
    spaceToPlane(Eigen::Vector3d(p_u(0), p_u(1), 1.0), p);
}

cv::Mat EucmCamera::initUndistortRectifyMap(cv::Mat& map1, cv::Mat& map2,
                                            float fx, float fy, cv::Size imageSize,
                                            float cx, float cy, cv::Mat rmat) const
{
    if (imageSize == cv::Size(0, 0))
        imageSize = cv::Size(mParameters.imageWidth(), mParameters.imageHeight());

    cv::Mat mapX(imageSize, CV_32F);
    cv::Mat mapY(imageSize, CV_32F);
    Eigen::Matrix3f R;
    cv::cv2eigen(rmat, R);

    Eigen::Matrix3f K;
    K << (fx > 0.0f ? fx : static_cast<float>(mParameters.fx())), 0.0f,
         (cx >= 0.0f ? cx : imageSize.width / 2.0f),
         0.0f, (fy > 0.0f ? fy : static_cast<float>(mParameters.fy())),
         (cy >= 0.0f ? cy : imageSize.height / 2.0f),
         0.0f, 0.0f, 1.0f;
    const Eigen::Matrix3f transform = R.inverse() * K.inverse();

    for (int v = 0; v < imageSize.height; ++v)
    {
        for (int u = 0; u < imageSize.width; ++u)
        {
            const Eigen::Vector3f ray = transform * Eigen::Vector3f(u, v, 1.0f);
            Eigen::Vector2d source;
            spaceToPlane(ray.cast<double>(), source);
            mapX.at<float>(v, u) = static_cast<float>(source(0));
            mapY.at<float>(v, u) = static_cast<float>(source(1));
        }
    }

    cv::convertMaps(mapX, mapY, map1, map2, CV_32FC1, false);
    cv::Mat Kcv;
    cv::eigen2cv(K, Kcv);
    return Kcv;
}

int EucmCamera::parameterCount() const { return 6; }
const EucmCamera::Parameters& EucmCamera::getParameters() const { return mParameters; }

void EucmCamera::setParameters(const Parameters& parameters)
{
    mParameters = parameters;
    m_inv_fx = 1.0 / mParameters.fx();
    m_inv_fy = 1.0 / mParameters.fy();
}

void EucmCamera::readParameters(const std::vector<double>& parameters)
{
    if (parameters.size() != static_cast<size_t>(parameterCount()))
        return;
    Parameters p = mParameters;
    p.alpha() = parameters[0];
    p.beta() = parameters[1];
    p.fx() = parameters[2];
    p.fy() = parameters[3];
    p.cx() = parameters[4];
    p.cy() = parameters[5];
    setParameters(p);
}

void EucmCamera::writeParameters(std::vector<double>& parameters) const
{
    parameters.resize(parameterCount());
    parameters[0] = mParameters.alpha();
    parameters[1] = mParameters.beta();
    parameters[2] = mParameters.fx();
    parameters[3] = mParameters.fy();
    parameters[4] = mParameters.cx();
    parameters[5] = mParameters.cy();
}

void EucmCamera::writeParametersToYamlFile(const std::string& filename) const
{
    mParameters.writeToYamlFile(filename);
}

std::string EucmCamera::parametersToString() const
{
    std::ostringstream oss;
    oss << mParameters;
    return oss.str();
}

}
