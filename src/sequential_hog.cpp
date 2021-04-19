#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>


using namespace std;
using namespace cv;
using namespace cv::ml;

int main( int argc, const char** argv )
{
    /// Use the cmdlineparser to process input arguments
    CommandLineParser parser(argc, argv,
        "{ help h            |      | show this message }"
        "{ video v           |      | (required) path to video }"
    );
    /// If help is entered
    if (parser.has("help")){
        parser.printMessage();
        return 0;
    }
    /// Parse arguments
    //string video_location(parser.get<string>("video"));
    //if (video_location.empty()){
    //    parser.printMessage();
    //    return -1;
    //}
    /// Create a videoreader interface
    string video_location = "../data/ped1.mp4";
    VideoCapture cap(video_location);
    Mat current_frame;

    // Print Key Statistics
    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    int input_fps = cap.get(CAP_PROP_FPS);


    cout << "Frame Width " << frame_width << endl;
    cout << "Frame Height " << frame_height << endl; 
    cout << "Input FPS " << input_fps << endl;

    //Initialize Writer to write output
    VideoWriter video("out.mp4", VideoWriter::fourcc('m','p','4','v'),10, Size(frame_width,frame_height),true);

    /// Set up the pedestrian detector --> let us take the default one
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    /// Set up tracking vector
    vector<Point> track;
    int i = 0; 
    time_t start, end;
    time(&start);
    while(true){
        /// Grab a single frame from the video
        cap >> current_frame;

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
       
        /// Check if the frame has content
        if(current_frame.empty()){
            cerr << "Video has ended or bad frame was read. Quitting." << endl;
            return 0;
        }
        /// run the detector with default parameters. to get a higher hit-rate
        /// (and more false alarms, respectively), decrease the hitThreshold and
        /// groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
        ///image, vector of rectangles, hit threshold, win stride, padding, scale, group th
        Mat img = current_frame.clone();
        resize(img,img,Size(img.cols*2, img.rows*2));
        vector<Rect> found;
        vector<double> weights;
        hog.detectMultiScale(img, found, weights);

        /// draw detections and store location
        cout << found.size() << " people were detected in the image" << endl;
        for( size_t i = 0; i < found.size(); i++ )
        {
            Rect r = found[i];
            rectangle(img, found[i], cv::Scalar(0,0,255), 3);
            stringstream temp;
            temp << weights[i];
            putText(img, temp.str(),Point(found[i].x,found[i].y+50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255));
            track.push_back(Point(found[i].x+found[i].width/2,found[i].y+found[i].height/2));
        }

        /// plot the track so far
        for(size_t i = 1; i < track.size(); i++){
            line(img, track[i-1], track[i], Scalar(255,255,0), 2);
        }
        resize(img,img,Size(img.cols/2, img.rows/2));
        video.write(img);

        /// Show
        //imshow("detected person", img);
        //waitKey(1);
    }
    video.release();
    cap.release();
    return 0;
}
