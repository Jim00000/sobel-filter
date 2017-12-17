#include <string>
#include "sobel.hpp"

int main()
{
    using namespace cv;
    using namespace imgproc;
    SobelFilter sobel{"Lenna.png"};
    imwrite("output.jpg", sobel.getDestMatrix());
    return 0;
}