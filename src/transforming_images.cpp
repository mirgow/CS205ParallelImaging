// c++ code explaining how to
// save an image to a defined
// location in OpenCV

// loading library files
//#include <highlevelmonitorconfigurationapi.h>
//#include <opencv2\highgui\highgui.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{

    // Reading the image file from a given location in system
    Mat img = imread("landscape4k.jpg", IMREAD_GRAYSCALE);

    // if there is no image
    // or in case of error
    if (img.empty()) {
        cout << "Can not open or image is not present" << endl;

        // wait for any key to be pressed
        cin.get();
        return -1;
    }

    // You can make any changes
    // like blurring, transformation
    auto star2 = std::chrono::steady_clock::now( );

    resize(img, img, Size(), .5, .5, cv::INTER_LANCZOS4);

    auto elapsed2 = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::steady_clock::now( ) - star2 );
    cout << "milliseconds for resizing: " << elapsed2.count( ) << '\n';



    // writing the image to a defined location as JPEG
    bool check = imwrite("lessresolutelandscape4k.jpg", img);

    auto star = std::chrono::steady_clock::now( );

    cv::cuda::GpuMat src;
    src.upload(img);
    src.download(img);

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::steady_clock::now( ) - star );


    cout << "milliseconds for upload and download: " << elapsed.count( ) << '\n';

    // if the image is not saved
    if (check == false) {
        cout << "Mission - Saving the image, FAILED" << endl;

        // wait for any key to be pressed
        cin.get();
        return -1;
    }

    cout << "Successfully saved the image. " << endl;

    // Naming the window
    //String geek_window = "MY SAVED IMAGE";

    // Creating a window
    //namedWindow(geek_window);

    // Showing the image in the defined window
    //imshow(geek_window, img);

    // waiting for any key to be pressed
    //waitKey(0);

    // destroying the creating window
    //destroyWindow(geek_window);

    return 0;
}
