# rpi5-hailo8l-yolov8-utility
Machine Learning Tool for Yolov8 Object Detection


Built ground up using the Hailo resource library to optimize for Raspberry Pi 5, allows you via the command line to do the following:

Use Yolov8 model (of any Yolov8 model size) to perform object detection in the following manner

- batch process images in a folder and store inferred images in an output folder
- batch process videos in a folder and store inferred videos in an output folder
- plug in a webcam (any camera not just rpicam) and record footage, perform inference and store an output video
- ability to select h.264 or h.265 codec for compression of video (unfortunately no hardware support as removed in the RPI 5 by the Raspberry Pi Foundation)

Optimizations were performed to enable operating even on the smaller 4GB RPI5. The goal was a focus on the Hailo 8L.

<ins>Noteworthy features:</ins>

1. **Multi-threading:**
    Implemented a producer-consumer pattern using C++17's std::thread. The main thread can capture frames and add them to a queue, while a separate thread processes frames from the queue. Think mutex

3. **Use OpenCV's GPU module:**
    While the Raspberry Pi 5's GPU isn't as powerful as discrete GPUs, it can still accelerate some operations. Use OpenCV's GPU module for operations like resizing and color space conversion.

5. **Optimize memory usage:**
   Preallocate buffers for input and output tensors to avoid frequent (re)allocations.

7. **Batch processing:**
   Implement batch processing for inference to maximize the Hailo 8L's throughput.

9. **Use OpenMP for parallelization:**
   Leverage OpenMP to parallelize CPU-bound operations, especially with software-based encoding and any other tasks.

11. **Optimize I/O operations:**
   Use asynchronous I/O for writing processed frames to disk to avoid blocking the main processing loop.

WORK IN PROGRESS:


13. **Use NEON intrinsic:**
   For performance-critical sections, I am looking at ARM NEON intrinsics to leverage SIMD instruction. This could be especially useful for fast image normalization. Still researching. I want to avoid bad segmentation faults or other issues that challenge robustness of the code for a marginal gain. 
   

## Compile settings

<ins>Note we are working with a locally built opencv4, this takes a long time to build on a Raspberry Pi 5. This is not necessary for deployments</ins>
Don't forget '-g' for GDB when needed

```g++ -std=c++17 -O2 -o yolohailo yolov8.cpp -lhailort `pkg-config --cflags --libs opencv4` -lstdc++fs```

