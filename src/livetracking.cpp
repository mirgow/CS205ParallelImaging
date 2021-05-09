#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <time.h>
#include <chrono>
#include <omp.h>
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;
using namespace std;

// Fill the vector with random colors
void getRandomColors(vector<Scalar>& colors, int numColors)
{
  RNG rng(0);
  for(int i=0; i < numColors; i++)
    colors.push_back(Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255)));
}



int main( int argc, const char* argv[]) {


    // set default values for tracking algorithm and video

    // create a video capture object to read videos
    //cout << cv::getBuildInformation();

    //http://www.earthcam.com/usa/newyork/timessquare/?cam=tsrobo1
    //cv::VideoCapture cap("https://video3.earthcam.com:1935/fecnetwork/7384.flv/chunklist.m3u8");
    // Florida https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1620607922/ei/Ui-YYIudFYXh8wSR3aMI/ip/141.154.51.239/id/Zv1fgmd1pr4.1/itag/96/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D137/hls_chunk_host/r3---sn-8xgp1vo-cvnl.googlevideo.com/playlist_duration/30/manifest_duration/30/vprv/1/playlist_type/DVR/initcwndbps/15520/mh/C7/mm/44/mn/sn-8xgp1vo-cvnl/ms/lva/mv/m/mvi/3/pl/16/dover/11/keepalive/yes/fexp/24001373,24007246/mt/1620585864/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,playlist_duration,manifest_duration,vprv,playlist_type/sig/AOq0QJ8wRAIgf6UfneE53tRkF3zhrzb6xKKyamf7fXnqYKs6IeXOK1ICIBS9_HAk3y19YN8Qim4WsjT1b8k9TCpwlaiYWd6NzZTi/lsparams/hls_chunk_host,initcwndbps,mh,mm,mn,ms,mv,mvi,pl/lsig/AG3C_xAwRQIhAN6hkmQdcCBppY3cv8Ut3aVpdGbChoBqadmXNG4ZHjEfAiB4WcCXikegA8QWd5w5ppYLRzI_8TMd9jWgZUYmX-I2bA%3D%3D/playlist/index.m3u8
    //cv::VideoCapture cap("https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1620607391/ei/Py2YYMGDB5P2hgbOl4jIAQ/ip/141.154.51.239/id/wc4Vy1T6hcE.0/itag/95/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D136/hls_chunk_host/r7---sn-8xgp1vo-cvne.googlevideo.com/playlist_duration/30/manifest_duration/30/vprv/1/playlist_type/DVR/initcwndbps/15170/mh/6H/mm/44/mn/sn-8xgp1vo-cvne/ms/lva/mv/m/mvi/7/pl/16/dover/11/keepalive/yes/fexp/24001373,24007246/beids/9466586/mt/1620585383/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,playlist_duration,manifest_duration,vprv,playlist_type/sig/AOq0QJ8wRgIhAOKie1HXIo1e6EQi0pSlY4uwgSi0ILbXco36_NxaVdl4AiEAsoazdxXJ2MzuxiVI3-aV6CmiW56-7yfHrQjqroKlmQ8%3D/lsparams/hls_chunk_host,initcwndbps,mh,mm,mn,ms,mv,mvi,pl/lsig/AG3C_xAwRgIhAJlBcjDdVTtIoChQxcuGdp6_Rp65qlmgs3pe7PZGDa8mAiEA0IMqmbv2-q0JQAqIaW2knmoNt9k3Ttquz8I-jCeZ2v8%3D/playlist/index.m3u8");
    cv::VideoCapture cap("https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1620609118/ei/_jOYYJSoC4vZhgaa9JfoDA/ip/141.154.51.239/id/cmkAbDUEoyA.1/itag/96/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D137/hls_chunk_host/r4---sn-8xgp1vo-cvne.googlevideo.com/playlist_duration/30/manifest_duration/30/vprv/1/playlist_type/DVR/initcwndbps/15020/mh/zA/mm/44/mn/sn-8xgp1vo-cvne/ms/lva/mv/m/mvi/4/pl/16/dover/11/keepalive/yes/fexp/24001373,24007246/mt/1620586836/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,playlist_duration,manifest_duration,vprv,playlist_type/sig/AOq0QJ8wRgIhANuJAC0uACCtApr96pWHTY0U3jGIkPZ5d905d9A7vVIbAiEA-Z3DBcyzYu9o88AHVTn664GR4QVToY92y0J3jJTjO3M%3D/lsparams/hls_chunk_host,initcwndbps,mh,mm,mn,ms,mv,mvi,pl/lsig/AG3C_xAwRQIhALYMAr51qhz45JwygYCTk_jIJ2OtsvGOkW_iqFZ4iiXfAiADE07ijMhQK7-gCR5gAu5d1j3icaobjUujU-L-TFnAaA%3D%3D/playlist/index.m3u8");
    Mat frame;
    //http://video3.earthcam.com:1935/fecnetwork/7132.flv/chunklist.m3u8
    //rtsp://wowza.floridakeysmedia.com:8090/sloppyjoesbar/sloppyjoesbar.stream/chunklist_w1500397604.m3u8
    //set factor for downsizing images, the lower the value the higher the FPS, though it'll trade off accuracy.
    float factor = 0.5;


    int size=10;
    cap.set(cv::CAP_PROP_BUFFERSIZE,size);
    cap.set(cv::CAP_PROP_FPS, 30);

    // quit if unabke to read video file
    if(!cap.isOpened())
    {
    cout << "Error Connecting to Video Stream "<< endl;
    return -1;
    }


    // read first frame
    cap >> frame;

    // initialize GPU
    cv::cuda::GpuMat d_frame;
    d_frame.upload(frame);

    int num_devices = cv::cuda::getCudaEnabledDeviceCount();
    cout << "Number Of GPUS " << num_devices << endl;

    int devicenum = cv::cuda::getDevice();
    cout << "Hello from GPU " << devicenum << endl;


    // image preprocessing functions (CPU)
    // cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    // cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
    // resize(frame, frame, Size(), 0.25, 0.25, INTER_AREA);

    // cuda::cvtColor(d_frame, d_frame, cv::COLOR_BGR2GRAY);
    // cuda::cvtColor(d_frame, d_frame, cv::COLOR_GRAY2BGR);
    cuda::resize(d_frame, d_frame, Size(), factor, factor, INTER_AREA);

    d_frame.download(frame);

    // Print Key Statistics
    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    int input_fps = cap.get(CAP_PROP_FPS);

    cout << "Frame Width " << frame_width << endl;
    cout << "Frame Height " << frame_height << endl;
    cout << "Input FPS " << input_fps << endl;

    VideoWriter video("livedemo2.mp4", VideoWriter::fourcc('m','p','4','v'),input_fps, Size(frame_width*factor,frame_height*factor),true);

    cv::HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    vector<Rect> bboxes;
    vector<double> weights;
    hog.detectMultiScale(frame, bboxes, weights);
    // quit if there are no objects to track
    if(bboxes.size() < 1){
        cout << "Warning, No objects found" << endl;
        return 0;
    }

    vector<Scalar> colors;
    getRandomColors(colors, bboxes.size());
    // Specify the tracker type
    string trackerType = "KCF";
    cout << trackerType << endl;
    vector<Ptr<Tracker>> trackers;
    // ReWrite to create multiple trackers
    for(int i=0; i < bboxes.size(); i++){
      // Initialize tracker with box
      Ptr<Tracker> tracker = TrackerKCF::create();
      tracker->init(frame, bboxes[i]);
      // Add tracker to vector of tracker objects
      trackers.push_back(tracker);
    }

    // Initialize stats
    int i = 0;
    auto start = std::chrono::steady_clock::now( );
    while(cap.isOpened() or i < 1000) {
      // get frame from the video
      cap >> frame;
      // Stop the program if reached end of video
      if (frame.empty()) break;
      // Track number of frames
      i++;
      cout << "Frame " << i << endl;

      // line below uploads to GPU
      d_frame.upload(frame);

      auto star = std::chrono::steady_clock::now( );


      // greyscaling (on CPU)
      // cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
      // cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
      // // resizing (on CPU)
      // resize(frame, frame, Size(), factor, factor, INTER_AREA);

      // greyscaling (on GPU)
      //cuda::cvtColor(d_frame, d_frame, cv::COLOR_BGR2GRAY);
      //cuda::cvtColor(d_frame, d_frame, cv::COLOR_GRAY2BGR);

      // resizing (on GPU)
      cuda::resize(d_frame, d_frame, Size(), factor, factor, INTER_AREA);

      // downloading from GPU
      d_frame.download(frame);


      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::steady_clock::now( ) - star );
      cout << "milliseconds to preprocess: " << elapsed.count( ) << '\n';

      // Track FPS of processing
      if (i % 20 == 0){
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::steady_clock::now( ) - start );
        float fps = 20000.0/elapsed.count( );
        cout << "FPS: " << fps << endl;
        start = std::chrono::steady_clock::now( );
      }
      bool found;
      #pragma omp parallel
      #pragma omp for
      for(unsigned i=0; i<bboxes.size(); i++){
        Rect2d temp = bboxes[i];
        found = trackers[i]->update(frame, temp);
        if (found){
          bboxes[i] = temp;
          rectangle(frame, bboxes[i], colors[i], 2, 1);
        }
      }

      video.write(frame);
      // quit on x button
      waitKey(10);
      if  (waitKey(1) == 27) break;

    }
    video.release();
    cap.release();
}
