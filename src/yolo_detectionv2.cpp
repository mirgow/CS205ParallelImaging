// Code adapted from https://funvision.blogspot.com/2020/04/simple-opencv-tutorial-for-yolo-darknet.html

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;
using namespace dnn;



int main()
{
    VideoCapture cap("/home/ubuntu/CS205ParallelImaging/data/ped1.mp4");
    std::string model = "/home/ubuntu/CS205ParallelImaging/src/yolov3.weights";  //findFile(parser.get<String>("model"));
    std::string config = "/home/ubuntu/CS205ParallelImaging/src/yolov3.cfg"; //findFile(parser.get<String>("config"));
    Net network = readNet(model, config,"Darknet");
    network.setPreferableBackend(DNN_BACKEND_DEFAULT);
    network.setPreferableTarget(DNN_TARGET_OPENCL);


    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    int input_fps = cap.get(CAP_PROP_FPS);


    cout << "Frame Width " << frame_width << endl;
    cout << "Frame Height " << frame_height << endl; 
    cout << "Input FPS " << input_fps << endl;
    VideoWriter video("detection.mp4", VideoWriter::fourcc('m','p','4','v'),10, Size(frame_width,frame_height),true);


    ifstream txtfile("objectclasses.txt");
    vector<string> classes;
    for (string line; getline( txtfile, line ); /**/ ){
        classes.push_back( line );
    }
        





    Mat img;
    int i = 0;
    auto start = std::chrono::steady_clock::now( );
    while(cap.isOpened()){
        cap >> img;
        if (img.empty()) break;
        if (!cap.isOpened()) {
            cout << "Video Capture Fail" << endl;
            break;
        }
        cout << "Frame: " << i << endl;
        i++;
        if (i % 20 == 0){
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::steady_clock::now( ) - start );
            float fps = 20000.0/elapsed.count( );
            cout << "FPS: " << fps << endl;
            auto start = std::chrono::steady_clock::now( );
        }

        

        static Mat blobFromImg;
        bool swapRB = true;
        blobFromImage(img, blobFromImg, 1.0, Size(416, 416), Scalar(), swapRB, false,  CV_8U);
        cout << blobFromImg.size << endl; 
        float scale = 1.0/ 255.0;
        Scalar mean = 0;
        network.setInput(blobFromImg, "", scale, mean);

        Mat outMat;
        network.forward(outMat);
        // rows represent number of detected object (proposed region)
        int rowsNoOfDetection = outMat.rows;
        // The coluns looks like this, The first is region center x, center y, width
        // height, The class 1 - N is the column entries, which gives you a number, 
        // where the biggist one corrsponding to most probable class. 
        // [x ; y ; w; h; class 1 ; class 2 ; class 3 ;  ; ;....]
        //  
        int colsCoordinatesPlusClassScore = outMat.cols;
        // Loop over number of detected object. 
        // TODO profile and parallelize this step
        cout << "No objects detected: " << rowsNoOfDetection << endl;
        for (int j = 0; j < rowsNoOfDetection; ++j)
        {
            // for each row, the score is from element 5 up
            // to number of classes index (5 - N columns)
            Mat scores = outMat.row(j).colRange(5, colsCoordinatesPlusClassScore);

            Point PositionOfMax;
            double confidence;

            // This function find indexes of min and max confidence and related index of element. 
            // The actual index is match to the concrete class of the object.
            // First parameter is Mat which is row [5fth - END] scores,
            // Second parameter will gives you min value of the scores. NOT needed 
            // confidence gives you a max value of the scores. This is needed, 
            // Third parameter is index of minimal element in scores
            // the last is position of the maximum value.. This is the class!!
            minMaxLoc(scores, 0, &confidence, 0, &PositionOfMax);
        
            //cout << confidence;
            if (confidence > 0.01) //TUNE this value?
            {
// thease four lines are
// [x ; y ; w; h;
                //cout << "Object Detected" << endl;
                int centerX = (int)(outMat.at<float>(j, 0) * img.cols); 
                int centerY = (int)(outMat.at<float>(j, 1) * img.rows); 
                int width =   (int)(outMat.at<float>(j, 2) * img.cols+20); 
                int height =   (int)(outMat.at<float>(j, 3) * img.rows+100); 

                int left = centerX - width / 2;
                int top = centerY - height / 2;


                stringstream ss;
                ss << PositionOfMax.x;
                string clas = ss.str();
                int color = PositionOfMax.x * 10;
                string category = classes[(int)PositionOfMax.x];
                putText(img, category, Point(left, top), 1, 2, Scalar(color, 255, 255), 2, false);
                stringstream ss2;
                ss << confidence;
                string conf = ss.str();

                rectangle(img, Rect(left, top, width, height), Scalar(0, 128, 0), 2, 8, 0);
            }
        }
        

        video.write(img);
        //namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
        //imshow("Display window", img);
        //waitKey(25);
    }
    video.release();
    cap.release();
    return 0;
}
