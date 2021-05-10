# CS205ParallelImaging

## What're We Exploring?

Massively speeding up the detection of objects in video, and tracking their movements. We've found current common methods to operate at slow real-time speeds, such as .25 FPS.
Optimizing and speeding up an object detection algorithm poses to help in realms such as sports analytics, medical scanning, surveillance and security, self-driving cars, and more. We were initially focused on the Histogram of Oriented Gradients (HOG) algorithm, but through testing and comparison of other object tracking methods, we've identified other faster baseline algorithms such as Kernalized Correlation Filters (KCF), as we'll demonstrate below. Taking these algorithms, we've gone on to create and apply some parallel applications of the code to further improve performance, increasing speedup to FPS rates above 10. Methods and all replicability information are outlined below.

tl;dr: Speeding up object tracking.

#### How're We Exploring This?

We're utilizing the beautiful OpenCV library in C++ for simpler and more efficient commands in video processing and machine vision. OpenCV conveniently can also be deployed with CUDA, the parallel computing platform developed by NVIDIA to rapidly speed up code, particularly on a GPU. That, combined with the flexibility of operating on C++ (over python, which has identical functionality), can enable use of both multiple GPUs and multithreading through OpenMP.

It's also worth to note the documentation on OpenCV+CUDA C++ framework is pretty terrible, and almost nonexistent for the python version (another reason we preferred on C++).

One significant drawback to this method is the inability to execute complicated functions in the GPU space, as there are [limited operations](https://docs.opencv.org/3.4/d0/d60/classcv_1_1cuda_1_1GpuMat.html) that can be called on a `GpuMat` instance. So, tracking operations will have to be held in the CPU memory. However, we came up with workarounds.

#### Kernelized Correlation Filters (KCF) Algorithm

This is an algorithm that produces the optimal image filter such that there is a fitted response for the filtration with the input image (in our case, a humanoid). After that initial identification, multiple 'trackers' are made, with rectangular boxes placed around them, follow the movement of the objects (humans). More information about the KCF algorithm can be read about [here](<https://cw.fel.cvut.cz/b172/courses/mpv/labs/4_tracking/4b_tracking_kcf#:~:text=References-,Tracking%20with%20Correlation%20Filters%20(KCF),by%20a%20rectangular%20bounding%20box.&text=The%20filter%20is%20trained%20from,new%20position%20of%20the%20target>)

Steps include:

1. Finding of optimal linear filter, solved as a least squares problem. Adds complexity of O(n^2)
2. Fourier transformations rapidly speed up solution process.
3. Map input data through a non-linear function, leading to kernelized ridge regression.
4. Obtain linear correlation tracker through the kernel, in our case a RBF Gaussian kernel.
5. Update through every frame with minimal distance.

#### Schematic, Proposed Solution

Sketched and modeled below is the rough ideas + framework to the project.
![frameworkgraph](https://github.com/mirgow/CS205ParallelImaging/blob/main/img/Framework.jpg)

As shown, we used two main methods to speedup our object tracking.

1. Parallelize multiple object trackers across multiple threads
2. Downsize the image prior to detecting objects.

The second method is a little cheap and leads to tradeoffs in the quality of object tracking a we will show below. However, this step was necessary in order to reach our goal of object detection on a live video feed and presented interesting challenges in terms of how we could speed up the downsizing operation using a GPU.

## Initial Benchmarking

### Frames Preprocessing

All of these timing results include uploading the frames to the GPU, testing here the difference in preprocessing images on CPU or GPU. In the case where the location is CPU, frames are uploaded and downloaded at the end. If location is GPU, frames are uploaded before and downloaded after.
These were run through the script [comparingvideorates.cpp](https://github.com/mirgow/CS205ParallelImaging/blob/main/src/comparingvideorates.cpp), with sample videos ped1, ped1_Trim, and ped1test, 4k videos respectively at 597, 299, and 25 frames.

| Location | Operation                   | Frame Count / Quality | Time/Frame (ms/frame) | FPS    |
| -------- | --------------------------- | --------------------- | --------------------- | ------ |
| CPU      | greyscaling frames          | 299 / 4k              | 60.064                | 16.649 |
| CPU      | resizing frames             | 299 / 4k              | 20.909                | 47.827 |
| CPU      | greyscaling+resizing frames | 25 / 4k               | 72.907                | 13.716 |
| CPU      | greyscaling+resizing frames | 299 / 4k              | 55.245                | 18.101 |
| CPU      | greyscaling+resizing frames | 597 / 4k              | 54.123                | 18.477 |
| GPU      | greyscaling frames          | 299 / 4k              | 28.350                | 35.274 |
| GPU      | resizing frames             | 299 / 4k              | 23.787                | 42.04  |
| GPU      | greyscaling+resizing frames | 25 / 4k               | 46.277                | 21.609 |
| GPU      | greyscaling+resizing frames | 299 / 4k              | 29.947                | 33.393 |
| GPU      | greyscaling+resizing frames | 597 / 4k              | 28.861                | 34.649 |

Note: Video Quality for 4k is described as a 3840x2160 frame size (in pixels).

![Graph for preprocessing data](https://github.com/mirgow/CS205ParallelImaging/blob/6c62f3b8d479bbcc78d2d0fd0feefd3fd6ec9565/img/CPU%20vs.%20GPU%20FPS%20Image%20Preprocessing.png)

### Main Algorithms

We initially benchmarked two main algorithms with sequential execution on a CPU; HOG object detection on KCF object tracking. The initial results were very slow on the order of seconds per frame rather than frames per second.

| Algorithm            | Sequential FPS |
| -------------------- | -------------- |
| HOG object detection | 0.238          |
| KCF object tracking  | 0.689          |

### Overheads

We also quantified the main relevant overheads for moving and resizing images on the CPU and GPU.

| Overheads | Read in one 4k frame using OpenCV | Copy image to/back from GPU | GPU initialization | Resizing on CPU | Resizing on GPU |
| --------- | --------------------------------- | --------------------------- | ------------------ | --------------- | --------------- |
| Time (ms) | 20                                | 1                           | ~200               | 27              | 7               |

## Application Technical Specs

-   Testing was carried out on AWS Computing Platform, specifically on one g3.8xlarge instance, consisting of 2 NVIDIA Tesla M60 GPUs. 32 cores (threads in parallel).
-   Hybrid Parallel Processing Framework: OpenCV CUDA multi-GPU use + OpenMP. (Hybrid distributed and shared-memory application)
    -   OpenCV CUDA module allows the optimization of code through GPUs, and with `cv::cuda::setDevice` the partitioning of different sections of code into separate GPUs. All usage of multiple GPUs has to be hardcoded/specified in the code. In our case, that would be either applying `cv::cuda::setDevice(0)` or `cv::cuda::setDevice(1)` before a chunk of code to tell the module to copy over information and carry out operations on either the first or second GPU on our 2-GPU g3.8xlarge instance. This is the distributed-memory part of our application, as we are forced to copy over information from CPU to the GPU.
    -   OpenMP then enables us to optimize the code in the massive multi-threading environment of GPUs. Employing `#pragma` statements in the code and specifying thread count greatly speeds up operations along both GPUs. This is the shared-memory aspect of our model, as OpenMP is applied as multi-threading on one node/GPU, and all the memory is contained in the GPU.
-

### Software Specs

-   Ubuntu distro 18.04
-   C++ compiler version 7.5.0
-   OpenCL
-   OpenCV 4.2.0
-   NVIDIA CUDA version 10.0
-   CMake 3.10.2

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

at the end of the file. Type ":wq" to save and quit.

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

21. With using g++ compiler to run C++ codes, insert `pkg-config opencv4 --cflags --libs` as one of the flags to indicate the libraries to use.

### Notable Lines of Code

##### Resizing: `cv::resize(source, destination, Size(), factorinx, factoriny, INTER_AREA);`

For GPU usage, GpuMat and not (CPU) Mat must be supplied to 'source', and resize should be initialized as `cv::cuda::resize()` to indicate an operation in the GPU memory space.

The 'factorinx' and 'factoriny' refer to the downscaling ratio.

'INTER_AREA' is the specification of the resizing algorithm - many different ones are available. We decided `INTER_AREA` was optimal.

##### Greyscaling:

```
cv::cvtColor(source, dest, cv::COLOR_BGR2GRAY);
cv::cvtColor(source, dest, cv::COLOR_GRAY2BGR);
```

Similarly, for GPU usage use `cv::cuda::cvtColor()`, and feed a GpuMat.

Two are placed back to back, because the Mat/GpuMat organization is corrupted in the gray format as dictated by `BGR2GRAY`, and needs to be restored to the color channels (without color).

##### GPU uploading/downloading:

Creation of a GpuMat container: `cv::cuda::GpuMat d_frame`

Uploading to GPU memory for a single frame: `d_frame.upload(frame)`

Downloading for videowriter: `d_frame.download(frame)`

## Execution How-To

### Environment Establishment

After the instance is made, run and create the environment for which to work in:

```
nvidia-smi
source ~/.virtualenvs/opencv_cuda/bin/activate
export OMP_NUM_THREADS=16
```

At this point, we're assuming you've downloaded our Github repo, and have access to the scripts in /src and test examples in /data

For the scripts we've placed in our repo, the backbones of them are obtained from our sources, so they're not original scripts. However, we have edited and reformatted them to a great degree.

### Data Sources and Testing Info

These sample videos are located in the [/data](https://github.com/mirgow/CS205ParallelImaging/tree/main/data) folder
| 4k sample video | Frames |
| -- | -- |
| [ped1test.mp4](https://github.com/mirgow/CS205ParallelImaging/blob/main/data/ped1test.mp4) | 25 |
| ped1_Trim.mp4 | 299 |
| [ped1.mp4](https://github.com/mirgow/CS205ParallelImaging/blob/main/data/ped1.mp4) | 597 |

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
Create the executable (note the `-fopenmp` flag must be used.):

```
g++ omp_tracking_updated.cpp -fopenmp `pkg-config opencv4 --cflags --libs` -o omp_tracking_updated
```

This script also takes video as first argument upon execution:

```
./omp_tracking_updated ped1test.mp4
```

The default running should look something like this, but with every frame in between. ![sample output](https://github.com/mirgow/CS205ParallelImaging/blob/main/img/trackinggpuspeedup.png)

The output video will be automatically saved as 'trackingupdated.mp4'.

Some adjustable areas within the script:

-   Line 69 `float factor` can be changed to anything between 0-1. It represents the downscaling of the 4k input video. Beware, lower values lead to lower accuracy (higher FN rates) but higher FPS. We defaulted at .25, so producing 1/2 the quality of HD video.
-   Lines 213-214 with the `pragma`'s are the implementation of OpenMP over the trackers. Can deactivate to test FPS (will lower).
-   Line 115 contains the name of the output file: `VideoWriter video("trackingupdated.mp4",`... and changing 'trackingupdated.mp4' will produce different named final video.

### YOLO object Detection

The yolo object detection algorithm can be run using the `yolo_detectionv2.cpp` script. YOLO is a deep learning based object detection method notable for only requiring a single forward pass through the network. It can be compiled using the standard flags for OpenCV.

```
g++ yolo_detectionv2.cpp `pkg-config opencv4 --cflags --libs` -o yolo
```

To run, the neural network weights and configuration file must be downloaded and placed in the same directory as the executable.

## Results

### Multithreaded Object Tracking

#### KCF results

For the tracking algorithms a set of objects is first detected using the HOG object detector with the SVM trained to detect pedestrians. These objects are then tracked for the rest of the video. The process of updating each object tracker with the new frame can be parallelized across threads using openMP. The speedups achieved using different numbers of threads is shown on the graph. Overall we were able to achieve a maximum 3.6x speedup using openMP.

![OpenMP graph](https://github.com/mirgow/CS205ParallelImaging/blob/main/img/openmptracking.png)

#### Comparison Of Object Tracking Algorithms

We compared the multithreaded implementations of the various image tracking algorithms in openCV. This verified the literature reported results that KCF tracking presented the best tradeoff between tracking quality and speed. We were not able to benchmark the GOTURN tracker available in openCV since this deep learning based algorithm required too much memory overhead to initialize multiple trackers. The table below shows benchmarks of the main tracking algorithms on a 4K video with multiple objects and 16 threads.

| Algorithm  | Multithreaded FPS |
| ---------- | ----------------- |
| KCF        | 2.5               |
| MOSSE      | 6.66              |
| BOOSTING   | 2.8               |
| MIL        | 1.5               |
| TLD        | 0.645             |
| MEDIANFLOW | 3.333             |
| CSRT       | 1.111             |

Here are the speedups with the other implementations, of aggregating each of the features outlined in the framework.

![Speedups](https://github.com/mirgow/CS205ParallelImaging/blob/main/img/KCF%20Algorithm%20Speedup%20With%20Built-Up%20Features.png)

We considered using a GPU implementation of the KCF object tracker. https://denismerigoux.github.io/GPU-tracking/. They achieved good results for 4K video object tracking with an observed 2x speedup. However based on simple calculations, using the multithreaded CPU version is better for our use case with multiple objects to track. For example parallelizing tracking of 12 objects across twelve threads would be faster than executing a 2x faster GPU version 12x sequentially.

### Object Detection

To evaluate for accuracy in tandem with FPS for a holistic representation of the quality of our methods, we can define bins for machine vision:
| Bin | Description |
| -- | -- |
| True Positive | Human identified |
| False Positive | Non-human identified |
| True Negative | Tricky to describe, but everything non-human, not identified |
| False Negative | Human not identified |

Then, we can define sensitivity as TP/TP+FN. (Specificity is another potential value to measure, but we can't quantify TN's here).
To that end, we parsed through final videos produced by algorithms detailed below by eye to identify FNs, because there is no other tool for that.

| Algorithm             | Sensitivity |
| --------------------- | ----------- |
| Base KCF              | ~67%        |
| Final Accelerated KCF | ~42%        |

So, there definitely is a tradeoff, particularly with the downsizing scale, as that removes data and possible adds artifacts. Although it's also worth to note we're downsizing with OpenCV's `INTER_AREA` algorithm, which quote 'gives moire'-free results,' and is the most preferred method for image decimation. This means artifacts will be limited.

### Deep learning for Object Detection

An alternative to online object tracking algorithms is simply to treat each frame as independent and detect objects as they come in. This has the advantage of eliminating any dependencies on the previous video frame but does not track an individually identifiable object over time. We used the yolov3 pre-trained deep learning model with openCL support. The baseline implementation was able to run at approximately 5 fps with around 40% utilization of a single Tesla M60 GPU. Note the the openCL implementation was unable to take advantage of multiple GPUs. Unfortunately we did not get to try out the CUDA accelerated version since the cuDNN library would not work with the Tesla M60. The openCL version was actually slower than the CPU. However, we would expect much better performance using the GPU version with CUDA. Qualitatively, the yolo object detection gives different results than the tracking algorithms. Each frame is evaluated independently so there is less coherence in the location of detected objects across frames. We can also see how yolo is able to assign a class label to the objects it detects.

| Backend | FPS |
| ------- | --- |
| CPU     | 8   |
| openCL  | 5   |

## Files

`src/`

-   `ImageLoading.cpp`: tests the time to copy images to PGU
-   `comparingvideorates.cpp`: used to create initial benchmarks for frames preprocessing. See more and results at [Frames Preprocessing](#frames-preprocessing)
-   `dogongrass.png`: used for image loading testing
-   `livetracking.cpp`: demo with tracking for live video stream
-   `objectclasses.txt`: labels for yolo detection
-   `omp_tracking_updated.cpp`: main file that implements of tracking people in a video
-   `yolo_detectionv2.cpp`: object detection with YOLO algorithm
-   `yolov3.cfg`: configuration file needed for `yolo_detectionv2.cpp`
-   `yolov3.weights`: weights file needed for `yolo_detectionv2.cpp`

`data/`

-   `crosswalk.avi.icloud`: cross walk video sample
-   `ped1.mp4`: ~30 second sample of people walking
-   `ped1test.mp4`: ~1 second sample of people walking


## Sources

-   https://developer.nvidia.com/cuda-gpus
-   https://learnopencv.com/histogram-of-oriented-gradients/
-   https://towardsdatascience.com/opencv-cuda-aws-ec2-no-more-tears-60af2b751c46
-   https://www.opencv-srf.com/2017/11/load-and-display-image.html
-   https://opencv.org/platforms/cuda/
-   https://queue.acm.org/detail.cfm?id=2206309
-   https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/
-   https://github.com/opencv/opencv/tree/master/samples/gpu
-   https://medium.com/dropout-analytics/opencv-cuda-for-videos-f3dcf346e398
-   https://cw.fel.cvut.cz/b172/courses/mpv/labs/4_tracking/4b_tracking_kcf#:~:text=References-,Tracking%20with%20Correlation%20Filters%20(KCF),by%20a%20rectangular%20bounding%20box.&text=The%20filter%20is%20trained%20from,new%20position%20of%20the%20target.
-   https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3
