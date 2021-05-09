#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <time.h>
#include <chrono>
#include <omp.h>
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;
using namespace std;

vector<string> trackerTypes = {"BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"};

// create tracker by name
Ptr<Tracker> createTrackerByName(string trackerType)
{
  Ptr<Tracker> tracker;
  if (trackerType ==  trackerTypes[0])
    tracker = TrackerBoosting::create();
  else if (trackerType == trackerTypes[1])
    tracker = TrackerMIL::create();
  else if (trackerType == trackerTypes[2])
    tracker = TrackerKCF::create();
  else if (trackerType == trackerTypes[3])
    tracker = TrackerTLD::create();
  else if (trackerType == trackerTypes[4])
    tracker = TrackerMedianFlow::create();
  else if (trackerType == trackerTypes[5])
    tracker = TrackerGOTURN::create();
  else if (trackerType == trackerTypes[6])
    tracker = TrackerMOSSE::create();
  else if (trackerType == trackerTypes[7])
    tracker = TrackerCSRT::create();
  else {
    cout << "Incorrect tracker name" << endl;
    cout << "Available trackers are: " << endl;
    for (vector<string>::iterator it = trackerTypes.begin() ; it != trackerTypes.end(); ++it)
      std::cout << " " << *it << endl;
  }
  return tracker;
}


// Fill the vector with random colors
void getRandomColors(vector<Scalar>& colors, int numColors)
{
  RNG rng(0);
  for(int i=0; i < numColors; i++)
    colors.push_back(Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255)));
}



int main( int argc, const char* argv[]) {

    if (argc != 2) {
        std::cout << "Not enough arguments\n";
        return -1;
    }

    const std::string fname(argv[1]);
    // set default values for tracking algorithm and video

    // create a video capture object to read videos
    cv::VideoCapture cap(fname);
    Mat frame;

    //set factor for downsizing images, the lower the value the higher the FPS, though it'll trade off accuracy.
    float factor = 0.25;

    // quit if unabke to read video file
    if(!cap.isOpened())
    {
    cout << "Error opening video file " << fname << endl;
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

    cuda::cvtColor(d_frame, d_frame, cv::COLOR_BGR2GRAY);
    cuda::cvtColor(d_frame, d_frame, cv::COLOR_GRAY2BGR);
    cuda::resize(d_frame, d_frame, Size(), factor, factor, INTER_AREA);

    d_frame.download(frame);

        // Print Key Statistics
    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    int input_fps = cap.get(CAP_PROP_FPS);


    cout << "Frame Width " << frame_width << endl;
    cout << "Frame Height " << frame_height << endl;
    cout << "Input FPS " << input_fps << endl;


    VideoWriter video("trackingupdated.mp4", VideoWriter::fourcc('m','p','4','v'),10, Size(frame_width*factor,frame_height*factor),true);
    // Get bounding boxes for first frame
    // selectROI's default behaviour is to draw box starting from the center
    // when fromCenter is set to false, you can draw box starting from top left corner
    // bool showCrosshair = true;
    // bool fromCenter = false;
    // cout << "\n==========================================================\n";
    // cout << "OpenCV says press c to cancel objects selection process" << endl;
    // cout << "It doesn't work. Press Escape to exit selection process" << endl;
    // cout << "\n==========================================================\n";
    // cv::selectROIs("MultiTracker", frame, bboxes, showCrosshair, fromCenter);
    // TODO run object detection on first frame instead of manually specifying bounding boxes.


    cv::HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    vector<Rect> bboxes;
    vector<double> weights;
    hog.detectMultiScale(frame, bboxes, weights);
    // quit if there are no objects to track
    if(bboxes.size() < 1){
        cout << "No objects found, quitting" << endl;
        return 0;
    }

    //int omp_get_max_threads();
    //int omp_get_num_threads();
    //cout << "Max threads:" << omp_get_max_threads() << '\t' << "Threads used: " << omp_get_num_threads() << endl;


    vector<Scalar> colors;
    getRandomColors(colors, bboxes.size());
    // Specify the tracker type
    string trackerType = "KCF";
    cout << trackerType << endl;
    vector<Ptr<Tracker>> trackers;
    // ReWrite to create multiple trackers
    for(int i=0; i < bboxes.size(); i++){
      // Initialize tracker with box
      Ptr<Tracker> tracker = createTrackerByName(trackerType);//TrackerKCF::create();
      tracker->init(frame, bboxes[i]);
      // Add tracker to vector of tracker objects
      trackers.push_back(tracker);
    }

    // Initialize stats
    int i = 0;
    auto start = std::chrono::steady_clock::now( );
    while(cap.isOpened()) {
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
      //
      // // resizing (on CPU)
      // resize(frame, frame, Size(), factor, factor, INTER_AREA);

      // greyscaling (on GPU)
      cuda::cvtColor(d_frame, d_frame, cv::COLOR_BGR2GRAY);
      cuda::cvtColor(d_frame, d_frame, cv::COLOR_GRAY2BGR);

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
        auto start = std::chrono::steady_clock::now( );
      }
      //Update the tracking result with new frame
      //auto star = std::chrono::steady_clock::now( );
      //auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::steady_clock::now( ) - star );
      //cout << "milliseconds since start: " << elapsed.count( ) << '\n';
      // Todo Update each tracker
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
      if  (waitKey(1) == 27) break;

    }
    video.release();
    cap.release();
}
