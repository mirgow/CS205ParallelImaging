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

