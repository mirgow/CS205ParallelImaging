#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <time.h>
#include <chrono>


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



int main( int argc, const char** argv ) {
// set default values for tracking algorithm and video
  string videoPath = "../data/ped1test.mp4";

  // create a video capture object to read videos
  cv::VideoCapture cap(videoPath);
  Mat frame;

  // quit if unabke to read video file
  if(!cap.isOpened()) 
  {
    cout << "Error opening video file " << videoPath << endl;
    return -1;
  }

  // read first frame
  cap >> frame;

     // Print Key Statistics
  int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
  int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
  int input_fps = cap.get(CAP_PROP_FPS);


  cout << "Frame Width " << frame_width << endl;
  cout << "Frame Height " << frame_height << endl; 
  cout << "Input FPS " << input_fps << endl;


  VideoWriter video("tracking.mp4", VideoWriter::fourcc('m','p','4','v'),10, Size(frame_width,frame_height),true);

  HOGDescriptor hog;
  hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

  vector<Rect> bboxes;
  vector<double> weights;
  hog.detectMultiScale(frame, bboxes, weights);

  // quit if there are no objects to track
  if(bboxes.size() < 1)
    return 0;

  vector<Scalar> colors;  
  getRandomColors(colors, bboxes.size()); 


  // Specify the tracker type
  string trackerType = "KCF";
  cout << trackerType; 
  // Create multitracker
  Ptr<MultiTracker> multiTracker = cv::MultiTracker::create();

// Initialize multitracker
  for(int i=0; i < bboxes.size(); i++)
    multiTracker->add(createTrackerByName(trackerType), frame, Rect2d(bboxes[i]));  
  }



  // Initialize stats
  int i = 0; 
  time_t start, end;
  time(&start);

  while(cap.isOpened()) 
  {
    // get frame from the video
    cap >> frame;
    // Track number of frames
    i++;
    cout << "Frame " << i << endl;

      // Track FPS of processing
    if (i % 20 == 0){
        time(&end);
        double seconds = difftime (end, start);
        double fps = 20/seconds;
        cout << "Processing Frames Per Second = " << fps << endl; 
        time(&start);
    }

    // Stop the program if reached end of video
    if (frame.empty()) break;

    //Update the tracking result with new frame
    auto star = std::chrono::steady_clock::now( );
    multiTracker->update(frame);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::steady_clock::now( ) - star );
    cout << "milliseconds since start: " << elapsed.count( ) << '\n';
    // Draw tracked objects
    for(unsigned i=0; i<multiTracker->getObjects().size(); i++)
    {
      rectangle(frame, multiTracker->getObjects()[i], colors[i], 2, 1);
    }
    video.write(frame);
    // quit on x button
    if  (waitKey(1) == 27) break;
    
  }
  video.release();
  cap.release();
}