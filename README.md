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