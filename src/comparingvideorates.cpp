#include <iostream>

#include "opencv2/opencv_modules.hpp"

#if defined(HAVE_OPENCV_CUDACODEC)

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

#include <opencv2/core.hpp>
//#include <opencv2/core/opengl.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/opencv.hpp>
//#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <omp.h>

using namespace cv;
using namespace std;

int main(int argc, const char* argv[])
{
    if (argc != 2) {
        std::cout << "Not enough arguments\n";
        return -1;
    }

    const std::string fname(argv[1]);

    //cv::namedWindow("CPU", cv::WINDOW_NORMAL);
    //cv::namedWindow("GPU", cv::WINDOW_OPENGL);
    //cv::cuda::setGlDevice(0);

    cv::Mat frame;
    cv::VideoCapture reader(fname);

    int frame_width = reader.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = reader.get(cv::CAP_PROP_FRAME_HEIGHT);
    int frame_count = reader.get(CAP_PROP_FRAME_COUNT);

    cout << "frame width and frame height: " << frame_width << '\t' << frame_height << '\n';
    cout << "frame count: " << frame_count << '\n';

    cv::VideoWriter video("output.mov", cv::VideoWriter::fourcc('m','p','4','v'), 20, Size(frame_width*0.5,frame_height*0.5), true);
    if(!video.isOpened()) {
        cout <<"Error! Unable to open video file for output." << '\n';
        return -1;
    }


    auto star = std::chrono::steady_clock::now( );
    cv::cuda::GpuMat d_frame(frame);
    cuda::GpuMat d_dst;


    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::steady_clock::now( ) - star );
    cout << "milliseconds to initialize GPU: " << elapsed.count( ) << '\n';
    //cv::Ptr<cv::cudacodec::VideoReader> d_reader = cv::cudacodec::createVideoReader(fname);

    cv::TickMeter tm;
    std::vector<double> cpu_times;
    std::vector<double> gpu_times;

    int gpu_frame_count=0, cpu_frame_count=0;

    //#pragma omp parallel for shared(cpu_times, video)
    for (int i = 0; i < frame_count; i++)
    {
        Mat resized;
        Mat grayscale;
        Mat frame;

        tm.reset(); tm.start();
        if (!reader.read(frame))
            continue;

        // greyscaling (on CPU, so long as it isn't after the d_frame.upload() function)
        //cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        //cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);

        // resizing (on CPU, again before the d_frame.upload() function)
        //resize(frame, frame, Size(), 0.5, 0.5, INTER_AREA);

        // line below uploads to GPU
        d_frame.upload(frame);

        // greyscaling (on GPU)
        cuda::cvtColor(d_frame, d_frame, cv::COLOR_BGR2GRAY);
        cuda::cvtColor(d_frame, d_frame, cv::COLOR_GRAY2BGR);

        // resizing (on GPU)
        cuda::resize(d_frame, d_frame, Size(), 0.5, 0.5, INTER_AREA);

        // line below downloads from GPU
        d_frame.download(frame);


        tm.stop();
        cpu_times.push_back(tm.getTimeMilli());
        //cpu_frame_count++;

        video.write(frame);
        //video << frame;

        //cv::imshow("CPU", frame);

        //if (cv::waitKey(3) > 0)
            //break;
    }

    // for (;;)
    // {
    //     tm.reset(); tm.start();
    //     if (!d_reader->nextFrame(d_frame))
    //         break;
    //     tm.stop();
    //     gpu_times.push_back(tm.getTimeMilli());
    //     gpu_frame_count++;
    //
    //     //cv::imshow("GPU", d_frame);
    //
    //     if (cv::waitKey(3) > 0)
    //         break;
    // }

    if (cpu_times.empty()) {
        std::cout << "CPU EMPTY\n";
        return -1;
    }

    std::cout << std::endl << "Results:" << std::endl;

    std::sort(cpu_times.begin(), cpu_times.end());

    double cpu_avg = std::accumulate(cpu_times.begin(), cpu_times.end(), 0.0) / cpu_times.size();

    std::cout << "CPU : Avg : " << cpu_avg << " ms FPS : " << 1000.0 / cpu_avg << " Frames " << frame_count << std::endl;


    video.release();
    return 0;
}

#else

int main()
{
    std::cout << "OpenCV was built without CUDA Video decoding support\n" << std::endl;
    return 0;
}

#endif
