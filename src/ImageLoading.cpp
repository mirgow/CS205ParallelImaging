// Test time to copy images to PGU 



#include <iostream>
#include <opencv2/opencv.hpp>
//#include <opencv2/gpu/gpu.hpp>
#include <opencv2/core/cuda.hpp> //Opencv4 uses the CUDA modules 
#include <time.h>
#include <chrono>

using namespace cv;
using namespace std;




int main (int argc, char* argv[])
{

    int num_devices = cv::cuda::getCudaEnabledDeviceCount();
    cout << "Number Of GPUS " << num_devices << endl;

    //CUDA context Initialization, Initializing CUDA for the first time seems
    // to have an overhead of about 0.1 second
    cv::cuda::GpuMat test;
    test.create(1, 1, CV_8U); // Just to initialize context
    try
    {
       
        cv::Mat src_host = cv::imread("dogongrass.png");
        cv::Mat result_host;

        // Test round trip time uploading to GPU and downloading to host
        auto star = std::chrono::steady_clock::now( );

        cv::cuda::GpuMat src;
        src.upload(src_host);
        src.download(result_host);

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::steady_clock::now( ) - star );
        
        
        cout << "milliseconds since start: " << elapsed.count( ) << '\n';
        imwrite("out.jpg", result_host);
    }
    catch(const cv::Exception& ex)
    {
        std::cout << "Error: " << ex.what() << std::endl;
    }
    return 0;
}
