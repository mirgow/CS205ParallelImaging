# CS205ParallelImaging

## What're We Exploring?
Massively speeding up the detection of objects in video, with an emphasis on histogram of oriented gradients (HOG) algorithms. 
This poses to help in realms such as:
- 

tl;dr: Speeding up object tracking.


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
- Testing carried out on AWS Computing Platform, 1 g3.8xlarge instance, consisting of 2 NVIDIA Tesla M60 GPUs.
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

## Results


### Multithreaded Object Tracking

The process of updating each object tracker with the new frame can be parallelized across threads usign openMP. The speedups achieved using different numbers of threads is shown on the graph. Overall we were able to achieve a maximum 3.6x speedup using openMP. 

![openMP graph]("./img/openmptracking.png")


### 

## Sources

https://developer.nvidia.com/cuda-gpus
https://learnopencv.com/histogram-of-oriented-gradients/
https://towardsdatascience.com/opencv-cuda-aws-ec2-no-more-tears-60af2b751c46


