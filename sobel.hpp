#ifndef _SOBEL_HPP_
#define _SOBEL_HPP_

#include <string>
#include <opencv2/opencv.hpp>

namespace imgproc
{

    class Filter
    {
    public:
        Filter();
        Filter(const std::string filename, int flags = CV_LOAD_IMAGE_COLOR);
        virtual ~Filter();
        bool readImage(const std::string filename, int flags = CV_LOAD_IMAGE_COLOR);
        const cv::Mat& getSourceMatrix() const;
        const cv::Mat& getDestMatrix() const;
        const int getRowSize() const;
        const int getColSize() const;
    protected:
        virtual void process() = 0;
        cv::Mat src, dst;
    private:
    };

    class SobelFilter : public Filter
    {
    public:
        SobelFilter(const std::string filename);
        ~SobelFilter();
    protected:
        void process() override;
    private:
    };
}

#endif