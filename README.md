# CS205ParallelImaging

## What're We Exploring?

Massively speeding up the detection of objects in video, and tracking their movements. We've found current common methods to operate at slow real-time speeds, such as .25 FPS. 
Optimizing and speeding up an object detection algorithm poses to help in realms such as sports analytics, medical scanning, surveillancce and security, self-driving cars, and more. We were initially focused on the Histogram of Oriented Gradients (HOG) algorithm, but through testing and comparison of other object tracking methods, we've identified other faster baseline algorithms such as Kernalized Correlation Filters (KCF), as we'll demonstrate below. Taking these algorithms, we've gone on to create and apply some parallel applications of the code to further improve performance, increasing speedup to FPS rates above 10. Methods and all replicability information are outlined below. 

tl;dr: Speeding up object tracking.

#### How're We Exploring This?

We're utilizing the beautiful OpenCV library in C++ for simpler and more efficient commands in video processing and machine vision. OpenCV conveniently can also be deployed with CUDA, the parallel computing platform developed by NVIDIA to rapidly speed up code, particularly on a GPU. That, combined with the flexibility of operating on C++ (over python, which has identical functionality), can enable use of both multiple GPUs and multithreading through OpenMP. 

It's also worth to note the documentation on OpenCV+CUDA C++ framework is pretty terrible, and almost devoid for the python version (another reason we preferred use on C++).

One significant drawback to this method is the inability to execute complicated functions in the GPU space, as there are [limited operations](https://docs.opencv.org/3.4/d0/d60/classcv_1_1cuda_1_1GpuMat.html) that can be called on a `GpuMat` instance. So, tracking operations will have to be held in the CPU memory. However, we came with workarounds. 

#### HOG Algorithm
This is a feature descriptor that extracts both the gradient and orientation of the edges of features in an image, which are then pooled into localized parts. Then, a unique histogram is generated for each part. 
Steps include:
1. Gradient for every pixel in image, formation of matrices in x and y directions.
2. Pythagoreas theorem to evaluate magnitude/direction for each pixel.
3. Generate histograms, most striaghtforward method adds up frequency of pixels based on their orientations. 
4. Combine features from smaller matrix chunks into larger.
5. Generate final overlay, which would appear as something like this:
![HOG on dog example](https://github.com/mirgow/CS205ParallelImaging/blob/main/img/doghog.png)

7. An identifier placed on top can now use the HOG to identify an object. 

#### Schematic, Proposed Solution
Sketched and modeled below is the rought ideas + framework to the project. 
![frameworkgraph](https://github.com/mirgow/CS205ParallelImaging/blob/main/img/Framework.jpg)


## Initial Benchmarking

### Frames Preprocessing

All of these timing results include uploading the frames to the GPU, testing here the difference in preprocessing images on CPU or GPU. In the case where the location is CPU, frames are uploaded and downloaded at the end. If location is GPU, frames are uploaded before and downloaded after. 
These were run through the script [comparingvideorates.cpp](https://github.com/mirgow/CS205ParallelImaging/blob/main/src/comparingvideorates.cpp), with sample videos ped1, ped1_Trim, and ped1test, 4k videos respectively at 597, 299, and 25 frames. 

| Location  | Operation | Frame Count / Quality | Time/Frame (ms/frame) | FPS |
| ------------- | ------------- | ------ | ----- | ----- |
| CPU | greyscaling frames  | 299 / 4k | 60.064 | 16.649 | 
| CPU | resizing frames | 299 / 4k | 20.909 | 47.827 | 
| CPU | greyscaling+resizing frames  | 25 / 4k | 72.907 | 13.716 |  
| CPU | greyscaling+resizing frames  | 299 / 4k | 55.245 | 18.101 | 
| CPU | greyscaling+resizing frames  | 597 / 4k | 54.123 | 18.477 | 
| GPU | greyscaling frames | 299 / 4k | 28.350 | 35.274 |
| GPU | resizing frames | 299 / 4k | 23.787 | 42.04 |
| GPU | greyscaling+resizing frames | 25 / 4k | 46.277 | 21.609 |
| GPU | greyscaling+resizing frames | 299 / 4k | 29.947 | 33.393 |
| GPU | greyscaling+resizing frames | 597 / 4k | 28.861 | 34.649 |

Note: Video Quality for 4k is described as a 3840x2160 frame size (in pixels). 

![Graph for preprocessing data](https://github.com/mirgow/CS205ParallelImaging/blob/6c62f3b8d479bbcc78d2d0fd0feefd3fd6ec9565/img/CPU%20vs.%20GPU%20FPS%20Image%20Preprocessing.png)

### Main Algorithms

| Algorithm  | Sequential FPS |
| ------------- | ------------- |
| HOG object detection  | 0.238  |
| KCF object tracking  | 0.689  |

### Overheads

Quantified with scripts in src directory,
| Overheads | Read one 4k image using OpenCV | Copy image to/back from GPU | GPU initilization | Resizing on CPU | Resizing on GPU |
| -- |-- |-- |-- |-- | -- |
| Time (ms) | 20 | 1 | ~200 | 27 | 7 |



## Application Technical Specs
- Testing carried out on AWS Computing Platform, 1 g3.8xlarge instance, consisting of 2 NVIDIA Tesla M60 GPUs. 32 cores (threads in parallel). 
- Hybrid Parallel Processing Framework: OpenCV CUDA multi-GPU use + OpenMP. (Hybrid distributed and shared-memory application)
  - OpenCV CUDA module allows the optimization of code through GPUs, and with `cv::cuda::setDevice` the partitioning of different sections of code into separate GPUs. All usage of multiple GPUs has to be hardcoded/specified in the code. In our case, that would be either applying `cv::cuda::setDevice(0)` or `cv::cuda::setDevice(1)` before a chunk of code to tell the module to copy over information and carry out operations on either the frist or second GPU on our 2-GPU g3.8xlarge instance. This is the distributed-memory part of our application, as we are forced to copy over information from CPU --> GPU. 
  - OpenMP then enables us to optimize the code in the massive multi-threading environment of GPUs. Employing `#pragma` statements in the code and specifying thread count greatly speeds up operations along both GPUs. This is the shared-memory aspect of our model, as OpenMP is applied as multi-threading on one node/GPU, and all the memory is contained in the GPU. 
- 

### Software Specs
- Ubuntu distro 18.04
- C++ compiler ver 7.5.0
- OpenCL
- OpenCV 4.2.0
- NVIDIA CUDA ver 10.0
- CMake 3.10.2

### Replicability Information
This is a lengthy section, just because installation of OpenCV + CUDA on an AWS instance for GPU processing can be quite complicated. 
1. Spin up an AWS instance with at least 2 GPUs. We used g3.8xlarge, but anything bigger in the G family or anything bigger than p3.8xlarge in the P family will work. 
2. Install CUDA driver and toolkit, among other dependencies. 
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential cmake unzip pkg-config
sudo apt-get install gcc-6 g++-6
sudo apt-get install screen
sudo apt-get install libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install libopenblas-dev libatlas-base-dev liblapack-dev gfortran
sudo apt-get install libhdf5-serial-dev
sudo apt-get install python3-dev python3-tk python-imaging-tk
sudo apt-get install libgtk-3-dev
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-driver-418
```
3. Reboot to take effect `sudo reboot`
4. Check existence of GPUs with `nvidia-smi`
5. Continue 
```
mkdir installers
cd installers/
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
mv cuda_10.0.130_410.48_linux cuda_10.0.130_410.48_linux.run
chmod +x cuda_10.0.130_410.48_linux.run
sudo ./cuda_10.0.130_410.48_linux.run --override
```
6. After EULA agreement, respond to all questions as yes or default except: 'Install NVIDIA Accelerated Graphics Driver for Linux...': reply with n; 'Enter CUDA Samples Location': reply with '/usr/local/cuda-9.2'.
7. PATHS to bashrc file `sudo vim ~/.bashrc` and add 
```
# NVIDIA CUDA Toolkit
export PATH=/usr/local/cuda-10.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
```
at the end of the file. Type :wq to save and quit. 

8. `source ~/.bashrc`
`nvcc -V`

9. More updates and packages 
``` 
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential cmake unzip pkg-config
sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install libgtk-3-dev
```
10. Download, open OpenCV materials.
``` 
cd ~
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.2.0.zip
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip
unzip opencv.zip
unzip opencv_contrib.zip
mv opencv-4.2.0 opencv
mv opencv_contrib-4.2.0 opencv_contrib
```
11. Setup Virtual Environment with Python 
```
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py`
sudo pip install virtualenv virtualenvwrapper
sudo rm -rf ~/get-pip.py ~/.cache/pip
```
12. Edit bashrc file again, `sudo vim ~/.bashrc` and insert 
```
# virtualenv and virtualenv wrapper
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
source /usr/local/bin/virtualenvwrapper.sh
```
13. Run `source ~/.bashrc`
14. Create python virtual environment
```
mkvirtualenv opencv_cuda -p python3
pip install numpy
cd ~/opencv
mkdir build
cd build
```
15. use `nvidia-smi` to find GPU model. If using g3.8xlarge, it should be a M60. Go to https://developer.nvidia.com/cuda-gpus to find compute capability. That value will substitute the value placed after `CUDA_ARCH_BIN=` in the next command.
16. Run cmake command, might take around 5 minutes. Remember to replace `CUDA_ARCH_BIN=` value!
```
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_PYTHON_EXAMPLES=ON -D INSTALL_C_EXAMPLES=OFF -D OPENCV_ENABLE_NONFREE=ON -D WITH_CUDA=ON -D WITH_CUDNN=OFF -D OPENCV_DNN_CUDA=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D CUDA_ARCH_BIN=5.2 -D WITH_CUBLAS=1 -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules -D HAVE_opencv_python3=ON -D PYTHON_EXECUTABLE=~/.virtualenvs/opencv_cuda/bin/python -D BUILD_EXAMPLES=ON -D OPENCV_GENERATE_PKGCONFIG=ON ..
```
17. Build the OpenCV library and implicate the maximum number of threads for doing so. Here, using all cores. Should take around 20 minutes. Press enter on keyboard if using PuTTY to prevent knock-out from system for being idle for 15 minutes. 
```
make -j$(nproc)
sudo make install
sudo ldconfig
```
18. Continue with sym-link to environment
```
ls -l /usr/local/lib/python3.6/site-packages/cv2/python-3.6
cd ~/.virtualenvs/opencv_cuda/lib/python3.6/site-packages/
ln -s /usr/local/lib/python3.6/site-packages/cv2/python-3.6/cv2.cpython-36m-x86_64-linux-gnu.so cv2.so
```
19. If logging out of node and coming back, to restart virtual environment, use `source ~/.virtualenvs/opencv_cuda/bin/activate`
20. Make sure OpenMP materials are ready
```
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
```
21. With using g++ compiler to run C++ codes, insert ``pkg-config opencv4 --cflags --libs`` as one of the flags to indicate the libraries to use. 

## Execution How-To
After the instance is made, run and create the environment for which to work in:
```
nvidia-smi
source ~/.virtualenvs/opencv_cuda/bin/activate
export OMP_NUM_THREADS=16
```

At this point, we're assuming you've downloaded our github repo, and have access to the scripts in /src and test examples in /data

### Image Preprocessing
Running [comparingvideorates.cpp](https://github.com/mirgow/CS205ParallelImaging/blob/main/src/comparingvideorates.cpp) to evaluate the different timings related to image preprocessing, involving greyscaling and resizing. 
Create the executable: 
```
g++ comparingvideorates.cpp -fopenmp `pkg-config opencv4 --cflags --libs` -o comparingvideorates
```
The script is designed so that you can input the video file as the argument following the executable, here's an example:
```
./comparingvideorates ped1test.mp4
```
By default, the script should display timing results of greyscaling and resizing with the GPU. ![Example picture.](https://github.com/mirgow/CS205ParallelImaging/blob/main/img/runningcomparingvideotests.png)
To change and test out other timing results, editing of lines 80-95 are necessary. 
CPU timing requires unhighlighting of lines 81 and 82 for greyscaling, line 85 for resizing, and highlighting of lines 91, 92, and 95.
GPU timing of only resizing requires highlighting of lines 91 and 92; timing of only greyscaling requires highlighting of frame 95. 

### KCF Object Tracking
Running [omp_tracking_updated.cpp](https://github.com/mirgow/CS205ParallelImaging/blob/main/src/omp_tracking_updated.cpp) to work with actual implementation of tracking people in a video. 
Create the executable:
```
g++ omp_tracking_updated.cpp -fopenmp `pkg-config opencv4 --cflags --libs` -o omp_tracking_updated
```
This script also takes video as first argument upon execution:
```
./omp_tracking_updated ped1test.mp4
```
The default running should look something like this, but with every frame in between. ![sample output](https://github.com/mirgow/CS205ParallelImaging/blob/main/img/trackinggpuspeedup.png)

Some adjustables within the script:
- Line 69 `float factor` can be changed to anything between 0-1. It represents the downscaling of the 4k input video. Beware, lower values lead to lower accuracy (higher FN rates) but higher FPS. We defaulted at .25, so producing 1/2 the quality of HD video.
- Lines 213-214 with the `pragma`'s are the implementation of OpenMP over the trackers. Can deactivate to test FPS (will lower).


## Results


### Multithreaded Object Tracking


#### Comparison Of Object Tracking Algorithms

We compared the multithreaded implementations of the various image tracking algorithms in openCV. This verified the literature reported results that KCF tracking presented the best tradeoff between tracking quality and speed. We were nto able to benchmark the GOTURN tracker available in openCV since this deep learning based algorithm required too much memory overhead to initialize multiple trackers.

| Algorithm  | Multithreaded FPS |
| ------------- | ------------- |
| KCF | 2.5 |
| MOSSE | 6.66  |
| BOOSTING  | 2.8 |
| MIL  | 1.5  |
| TLD  | 0.645 |
| MEDIANFLOW  | 3.333 |
| CSRT | 1.111  |



#### KCF results

The process of updating each object tracker with the new frame can be parallelized across threads usign openMP. The speedups achieved using different numbers of threads is shown on the graph. Overall we were able to achieve a maximum 3.6x speedup using openMP. 

![OpenMP graph](https://github.com/mirgow/CS205ParallelImaging/blob/main/img/openmptracking.png)

Here are the speedups with the other implementations, of aggregating each of the features outlined in the framework. 
![Speedups](https://github.com/mirgow/CS205ParallelImaging/blob/main/img/KCF%20Algorithm%20Speedup%20With%20Built-Up%20Features.png)

### Object Detection

To evaluate for accuracy in tandem with FPS for a holistic representation of the quality of our methods, we can define bins for machine vision:
| Bin | Description |
| -- | -- |
| True Positive | Human identified |
| False Positive | Non-human identified |
| True Negative | Tricky to describe, but everything non-human, not identified |
| False Negative | Human not identified |

Then, we can define sensitivity as TP/TP+FN. (Specificity is another potential value to measure, but we can't quantify TN's here). 

| Algorithm | Sensitivity |
| --- | --- |
| Base KCF | ~67% |
| Final Accelerated KCF | ~42% |

So, there definitely is a tradeoff, particularly with the downsizing scale, as that removes data and possible adds artifacts. Although it's also worth to note we're downsizing with OpenCV's `INTER_AREA` algorithm, which quote 'gives moire'-free results,' and is the most preferred method for image decimation. This means artifacts will be limited. 

An alternative to online object tracking algorithms is simply to treat each frame as independent and detect objects as they come in. This has the advantage of eliminating any dependencies on the previous video frame but does not track an individually identifyable object over time. We used the yolov3 pretrained deeplearning model with openCL support. The baseline implementation was able to run at approximately 5 fps with around 40% utilization of a single Tesla M60 GPU. 

## Sources

- https://developer.nvidia.com/cuda-gpus
- https://learnopencv.com/histogram-of-oriented-gradients/
- https://towardsdatascience.com/opencv-cuda-aws-ec2-no-more-tears-60af2b751c46
- https://www.opencv-srf.com/2017/11/load-and-display-image.html
- https://opencv.org/platforms/cuda/
- https://queue.acm.org/detail.cfm?id=2206309
- https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/
- https://github.com/opencv/opencv/tree/master/samples/gpu
- https://medium.com/dropout-analytics/opencv-cuda-for-videos-f3dcf346e398
- 

