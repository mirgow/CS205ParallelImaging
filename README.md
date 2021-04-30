# CS205ParallelImaging

## What're We Exploring?

Massively speeding up the detection of objects in video, with an emphasis on histogram of oriented gradients (HOG) algorithms. 
This poses to help in realms such as sports analytics, medical scanning, surveillancce and security, self-driving cars, and more. 
As computer vision applications with HOG stand now, 


tl;dr: Speeding up object tracking.

#### HOG Algorithm
This is a feature descriptor that extracts both the gradient and orientation of the edges of features in an image, which are then pooled into localized parts. Then, a unique histogram is generated for each part. 
Steps include:
1. Gradient for every pixel in image, formation of matrices in x and y directions.
2. Pythagoreas theorem to evaluate magnitude/direction for each pixel.
3. Generate histograms, most striaghtforward method adds up frequency of pixels based on their orientations. 
4. Combine features from smaller matrix chunks into larger.
5. Generate final overlay, which would appear as something like this:
![HOG on dog example]("/img/doghog.png")

7. An identifier placed on top can now use the HOG to identify an object. 


## Initial Benchmarking


### Main Algorithms

| Algorithm  | Sequential FPS |
| ------------- | ------------- |
| HOG object detection  | 0.238  |
| KCF object tracking  | 0.689  |

### Overheads

The time taken to read one 4K image (one frame of video) using opencv is 20ms.

The time taken to copy one 4K image to and from the GPU is approximately 1ms. 


## Application Technical Specs
- Testing carried out on AWS Computing Platform, 1 g3.8xlarge instance, consisting of 2 NVIDIA Tesla M60 GPUs. 32 cores (threads in parallel). 
- Hybrid Parallel Processing Framework: [TO-DO]
  - 
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
2. Install CUDA driver and toolkit, among other dependencies. `sudo apt-get update`
`sudo apt-get upgrade`
`sudo apt-get install build-essential cmake unzip pkg-config`
`sudo apt-get install gcc-6 g++-6`
`sudo apt-get install screen`
`sudo apt-get install libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev`
`sudo apt-get install libjpeg-dev libpng-dev libtiff-dev`
`sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev`
`sudo apt-get install libxvidcore-dev libx264-dev`
`sudo apt-get install libopenblas-dev libatlas-base-dev liblapack-dev gfortran`
`sudo apt-get install libhdf5-serial-dev`
`sudo apt-get install python3-dev python3-tk python-imaging-tk`
`sudo apt-get install libgtk-3-dev`
`sudo add-apt-repository ppa:graphics-drivers/ppa`
`sudo apt-get update`
`sudo apt-get install nvidia-driver-418`
3. Reboot to take effect `sudo reboot`
4. Check existence of GPUs with `nvidia-smi`
5. Continue `mkdir installers`
`cd installers/`
`wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux`
`mv cuda_10.0.130_410.48_linux cuda_10.0.130_410.48_linux.run`
`chmod +x cuda_10.0.130_410.48_linux.run`
`sudo ./cuda_10.0.130_410.48_linux.run --override`
6. After EULA agreement, respond to all questions as yes or default except: 'Install NVIDIA Accelerated Graphics Driver for Linux...': reply with n; 'Enter CUDA Samples Location': reply with '/usr/local/cuda-9.2'.
7. PATHS to bashrc file `sudo vim ~/.bashrc` and add `# NVIDIA CUDA Toolkit
export PATH=/usr/local/cuda-10.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH` at the end of the file. Type :wq to save and quit. 
8. `source ~/.bashrc`
`nvcc -V`
9. More updates and packages `sudo apt-get update`
`sudo apt-get upgrade`
`sudo apt-get install build-essential cmake unzip pkg-config`
`sudo apt-get install libjpeg-dev libpng-dev libtiff-dev`
`sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev`
`sudo apt-get install libgtk-3-dev`
10. `cd ~`
`wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.2.0.zip`
`wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip`
`unzip opencv.zip`
`unzip opencv_contrib.zip`
`mv opencv-4.2.0 opencv`
`mv opencv_contrib-4.2.0 opencv_contrib`
11. Setup Virtual Environment with Python `wget https://bootstrap.pypa.io/get-pip.py`
`sudo python3 get-pip.py`
`sudo pip install virtualenv virtualenvwrapper`
`sudo rm -rf ~/get-pip.py ~/.cache/pip`
12. Edit bashrc file again, `sudo vim ~/.bashrc` and insert 

## Results


### Multithreaded Object Tracking


#### Comparison Of Object Tracking Algorithms

We compared the multithreaded implementations of the various image tracking algorithms in openCV. This verified the literature reported results that KCF tracking presented the best tradeoff between tracking quality and speed. 


| Algorithm  | Multithreaded FPS |
| ------------- | ------------- |
| KCF | 2.5 |
| MOSSE | 6.66  |
| BOOSTING  | 2.8 |
| MIL  | 1.5  |
| TLD  | 0.645 |
| MEDIANFLOW  | 3.333 |
| GOTURN  | CSRT  |
| CSRT | 1.111  |



#### KCF results

The process of updating each object tracker with the new frame can be parallelized across threads usign openMP. The speedups achieved using different numbers of threads is shown on the graph. Overall we were able to achieve a maximum 3.6x speedup using openMP. 

![OpenMP graph]("./img/openmptracking.png")


### 

## Sources

- https://developer.nvidia.com/cuda-gpus
- https://learnopencv.com/histogram-of-oriented-gradients/
- https://towardsdatascience.com/opencv-cuda-aws-ec2-no-more-tears-60af2b751c46
- https://www.opencv-srf.com/2017/11/load-and-display-image.html
- https://opencv.org/platforms/cuda/
- https://queue.acm.org/detail.cfm?id=2206309
- https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/

