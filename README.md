# CS205ParallelImaging

## Initial Benchmarking


### Main Algorithms

| Algorithm  | Sequential FPS |
| ------------- | ------------- |
| HOG object detection  | 0.238  |
| KCF object tracking  | 0.689  |

### Overheads

The time taken to read one 4K image (one frame of video) using opencv is 20ms.

The time taken to copy one 4K image to and from the GPU is approximately 1ms. 



## Results


### Multithreaded Object Tracking

The process of updating each object tracker with the new frame can be parallelized across threads usign openMP. The speedups achieved using different numbers of threads is shown on the graph. Overall we were able to achieve a maximum 3.6x speedup using openmp. 

![opemMP graph]("img/openmptracking.png")