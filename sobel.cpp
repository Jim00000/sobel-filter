#include <cstring>
#include <iostream>
#include <vector>
#include <boost/filesystem.hpp>
#include "sobel.hpp"

using namespace imgproc;

extern void _sobel_process_(unsigned char* src, unsigned char* dst, int row, int col);
extern void _split_channel_(unsigned char* src, unsigned char* r, unsigned char* g, unsigned char* b, int row, int col);

Filter::Filter()
{
}

Filter::Filter(const std::string filename, int flags)
{
    this->readImage(filename, flags);
}

Filter::~Filter()
{
}

bool Filter::readImage(const std::string filename, int flags)
{
    using namespace cv;
    using namespace boost::filesystem;
    path filepath{filename};
    bool isFileExisted = exists(filepath);
    if(isFileExisted == false) {
        std::cerr << "File " << filename << " does not exist" << std::endl;
        return false;
    }

    this->src = std::move(imread(filename, flags));

    return true;
}

const cv::Mat &Filter::getSourceMatrix() const
{
    return this->src;
}

const cv::Mat &Filter::getDestMatrix() const
{
    return this->dst;
}

const int Filter::getRowSize() const
{
    return this->src.rows;
}

const int Filter::getColSize() const
{
    return this->src.rows;
}

SobelFilter::SobelFilter(const std::string filename) : Filter(filename, CV_LOAD_IMAGE_GRAYSCALE)
{
    this->process();
}

SobelFilter::~SobelFilter()
{
}

void SobelFilter::process()
{
    using namespace cv;
    using namespace std;
    Mat src = Filter::getSourceMatrix();
    int row = src.rows;
    int col = src.cols;
    Mat dst = Mat(row, col, src.type());
    _sobel_process_(src.data, dst.data, row, col);
    Filter::dst = std::move(dst);
}
