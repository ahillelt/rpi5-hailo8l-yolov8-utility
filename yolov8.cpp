#include <hailo/hailort.h>
#include <hailo/hailort_common.hpp>
#include <hailo/vdevice.hpp>
#include <hailo/infer_model.hpp>

#include <chrono>
#include <filesystem>
#include <vector>
#include <string>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>

#include "output_tensor.h"
#include "debug.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>


namespace fs = std::filesystem;

// Default values
std::string hefFile             = "yolov8s.hef";
float       confidenceThreshold = 0.5f;
float       nmsIoUThreshold     = 0.45f;

std::string inputMode           = "image_folder";
std::string compressionCodec    = "";
int         CAPWIDTH            = 1280;
int         CAPHEIGHT           = 720;

bool        useCamera           = false;
std::string cameraMode          = "10s"; // Default to 10 seconds
const int   FRAMES_PER_SECOND   = 24.0; //recording fps camera

// Function declarations
void printUsage();
void parseArguments(int argc, char** argv);
std::string exec(const char* cmd);

void compressVideo(const fs::path& inputPath, const fs::path& outputPath, const std::string& codec);
int processCameraFeed(hailort::InferModel& infer_model, hailort::ConfiguredInferModel& configured_infer_model);

cv::Mat processFrame(const cv::Mat& frame, hailort::InferModel& infer_model, hailort::ConfiguredInferModel& configured_infer_model);
int processImage(const std::string& imgFilename, hailort::InferModel& infer_model, hailort::ConfiguredInferModel& configured_infer_model);
int processVideo(const std::string& videoFilename, hailort::InferModel& infer_model, hailort::ConfiguredInferModel& configured_infer_model);
int run();

void printUsage() {
    printf("Usage: yolohailo [options]\n");
    printf("Options:\n");
    printf("  --model <model_file>        Specify the Hailo model file (default: yolov8s.hef)\n");
    printf("  --confidenceThreshold <threshold>  Set the confidence threshold for object detection (default: 0.5)\n");
    printf("  --nmsIoUThreshold <threshold>     Set the NMS IoU threshold for object detection (default: 0.45)\n");
    printf("  --image_folder              Process all images in the 'input_image' directory\n");
    printf("  --video_folder              Process all videos in the 'input_video' directory\n");
    printf("  --compress <codec>          Compress the output video using the specified codec (e.g., 'h.265')\n");
    printf("  --camera <mode>             Use the default camera for input. Mode can be:\n");
    printf("                              'always-on', '<number>s' for seconds, or '<number>m' for minutes\n");
    printf("  --help                      Display this help message\n");
}

void parseArguments(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            hefFile = argv[++i];
        } else if (arg == "--confidenceThreshold" && i + 1 < argc) {
            confidenceThreshold = std::stof(argv[++i]);
        } else if (arg == "--nmsIoUThreshold" && i + 1 < argc) {
            nmsIoUThreshold = std::stof(argv[++i]);
        } else if (arg == "--image_folder" || arg == "--video_folder") {
            inputMode = arg.substr(2);  // remove "--"
        } else if (arg == "--compress" && i + 1 < argc) {
            compressionCodec = argv[++i];
        } else if (arg == "--camera" && i + 1 < argc) {
            useCamera = true;
            cameraMode = argv[++i];
        } else if (arg == "--help") {
            printUsage();
            exit(0);
        }
    }
}

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

void compressVideo(const fs::path& inputPath, const fs::path& outputPath, const std::string& codec, std::string& ffmpeg_output) {
    fs::path tempOutputPath = outputPath.parent_path() / ("temp_" + outputPath.filename().string());
    std::string ffmpeg_cmd;
    if (!codec.empty()) {
        if (codec == "h.265" || codec == "H.265") {
            ffmpeg_cmd = "ffmpeg -y -i " + inputPath.string() + 
                         " -c:v libx265 -preset fast -crf 32 " + 
                         tempOutputPath.string();
        } else {
            // Use H.264 as the default codec
            ffmpeg_cmd = "ffmpeg -y -i " + inputPath.string() + 
                         " -c:v libx264 -preset fast -crf 28 " + 
                         tempOutputPath.string();
        }
    } else {
        // Use H.264 as the default codec
        ffmpeg_cmd = "ffmpeg -y -i " + inputPath.string() + 
                     " -c:v libx264 -preset fast -crf 28 " + 
                     tempOutputPath.string();
    }

    printf("Compressing video with command: %s\n", ffmpeg_cmd.c_str());
    ffmpeg_output = exec(ffmpeg_cmd.c_str());
    printf("FFmpeg output:\n%s\n", ffmpeg_output.c_str());

    // Check if the temporary output file was created
    if (!fs::exists(tempOutputPath)) {
        printf("Error: FFmpeg failed to create the temporary output file at %s\n", tempOutputPath.string().c_str());
        return;
    }

    // Remove the original file if it exists
    if (fs::exists(outputPath)) {
        try {
            fs::remove(outputPath);
        } catch (const std::filesystem::filesystem_error& e) {
            printf("Error removing original file: %s\n", e.what());
            return;
        }
    }

    // Rename the temp file
    try {
        fs::rename(tempOutputPath, outputPath);
    } catch (const std::filesystem::filesystem_error& e) {
        printf("Error renaming temporary file: %s\n", e.what());
        return;
    }

    printf("Video compression completed successfully.\n");
}

int processCameraFeed(hailort::InferModel& infer_model, hailort::ConfiguredInferModel& configured_infer_model) {
    cv::VideoCapture cap(0, cv::CAP_V4L2);  // Open the default camera
    if (!cap.isOpened()) {
        printf("Error opening camera\n");
        return 1;
    }
    
    cap.set(cv::CAP_PROP_FRAME_WIDTH, CAPWIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, CAPHEIGHT);
    
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = FRAMES_PER_SECOND;  // fallback if unable to get FPS from camera

    fs::create_directory("output_cam");
    fs::path tempOutputPath = fs::path("output_cam") / "temp_camera_output.mp4";
    fs::path finalOutputPath = fs::path("output_cam") / "camera_output.mp4";

    cv::VideoWriter video(tempOutputPath.string(), cv::VideoWriter::fourcc('a','v','c','1'), fps, cv::Size(frame_width, frame_height));

    if (!video.isOpened()) {
        printf("Error creating temporary output video file.\n");
        return 1;
    }

    cv::Mat frame;
    int frame_count = 0;
    auto start = std::chrono::steady_clock::now();
    
    int duration_seconds = -1;
    bool always_on = false;

    if (cameraMode == "always-on") {
        always_on = true;
    } else if (cameraMode.back() == 's') {
        duration_seconds = std::stoi(cameraMode.substr(0, cameraMode.length() - 1));
    } else if (cameraMode.back() == 'm') {
        duration_seconds = std::stoi(cameraMode.substr(0, cameraMode.length() - 1)) * 60;
    } else {
        printf("Invalid camera mode. Using default 10 seconds.\n");
        duration_seconds = 10;
    }

    printf("Camera mode: %s\n", always_on ? "Always on" : ("Recording for " + std::to_string(duration_seconds) + " seconds"));

    while (true) {
        cap.read(frame);
        if (frame.empty()) {
            printf("Error capturing frame\n");
            break;
        }

        cv::Mat processed_frame = processFrame(frame, infer_model, configured_infer_model);
        video.write(processed_frame);
        frame_count++;

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();

        if (!always_on && elapsed >= duration_seconds) {
            break;
        }

        if (always_on) {
            // Check for key press (non-blocking)
            int key = cv::waitKey(1);
            if (key >= 0) {
                printf("Key pressed. Stopping recording.\n");
                break;
            }
        }
    }

    cap.release();
    video.release();

    printf("Recorded %d frames from camera\n", frame_count);

    // Compress the video
    std::string ffmpeg_output;
    compressVideo(tempOutputPath, finalOutputPath, compressionCodec, ffmpeg_output);
    if (!fs::exists(finalOutputPath)) {
        printf("FFmpeg command failed. Output:\n%s\n", ffmpeg_output.c_str());
        return 1;
    }

    // Remove temporary file
    fs::remove(tempOutputPath);

    printf("Final output video size: %.2f MB\n", fs::file_size(finalOutputPath) / (1024.0 * 1024.0));

    return 0;
}

cv::Mat processFrame(const cv::Mat& frame, hailort::InferModel& infer_model, hailort::ConfiguredInferModel& configured_infer_model) {
    using namespace hailort;

    int nnWidth  = infer_model.inputs()[0].shape().width;
    int nnHeight = infer_model.inputs()[0].shape().height;

    const std::string& input_name = infer_model.get_input_names()[0];
    size_t input_frame_size = infer_model.input(input_name)->get_frame_size();

    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(nnWidth, nnHeight));
    cv::Mat rgb_frame;
    cv::cvtColor(resized_frame, rgb_frame, cv::COLOR_BGR2RGB);

    auto bindings_exp = configured_infer_model.create_bindings();
    if (!bindings_exp) {
        printf("Failed to get infer model bindings\n");
        return frame;
    }
    ConfiguredInferModel::Bindings bindings = std::move(bindings_exp.release());

    auto status = bindings.input(input_name)->set_buffer(MemoryView((void*)(rgb_frame.data), input_frame_size));
    if (status != HAILO_SUCCESS) {
        printf("Failed to set memory buffer: %d\n", (int)status);
        return frame;
    }

    std::vector<OutTensor> output_tensors;

    for (auto const& output_name : infer_model.get_output_names()) {
        size_t output_size = infer_model.output(output_name)->get_frame_size();
        uint8_t* output_buffer = (uint8_t*)malloc(output_size);
        if (!output_buffer) {
            printf("Could not allocate an output buffer!");
            return frame;
        }

        status = bindings.output(output_name)->set_buffer(MemoryView(output_buffer, output_size));
        if (status != HAILO_SUCCESS) {
            printf("Failed to set infer output buffer, status = %d", (int)status);
            return frame;
        }

        const std::vector<hailo_quant_info_t> quant = infer_model.output(output_name)->get_quant_infos();
        const hailo_3d_image_shape_t          shape = infer_model.output(output_name)->shape();
        const hailo_format_t                  format = infer_model.output(output_name)->format();
        output_tensors.emplace_back(output_buffer, output_name, quant[0], shape, format);
    }

    status = configured_infer_model.wait_for_async_ready(std::chrono::seconds(1));
    if (status != HAILO_SUCCESS) {
        printf("Failed to wait for async ready, status = %d", (int)status);
        return frame;
    }

    Expected<AsyncInferJob> job_exp = configured_infer_model.run_async(bindings);
    if (!job_exp) {
        printf("Failed to start async infer job, status = %d\n", (int)job_exp.status());
        return frame;
    }
    AsyncInferJob job = job_exp.release();
    job.detach();

    std::sort(output_tensors.begin(), output_tensors.end(), OutTensor::SortFunction);

    status = job.wait(std::chrono::seconds(1));
    if (status != HAILO_SUCCESS) {
        printf("Failed to wait for inference to finish, status = %d\n", (int)status);
        return frame;
    }

    // Create a copy of the original frame
    cv::Mat output_frame = frame.clone();

    bool nmsOnHailo = infer_model.outputs().size() == 1 && infer_model.outputs()[0].is_nms();

    if (nmsOnHailo) {
        OutTensor* out = &output_tensors[0];
        const float* raw = (const float*)out->data;

        size_t numClasses = (size_t)out->shape.height;
        size_t classIdx = 0;
        size_t idx = 0;

        while (classIdx < numClasses) {
            size_t numBoxes = (size_t)raw[idx++];
            for (size_t i = 0; i < numBoxes; i++) {
                float ymin = raw[idx];
                float xmin = raw[idx + 1];
                float ymax = raw[idx + 2];
                float xmax = raw[idx + 3];
                float confidence = raw[idx + 4];
                if (confidence >= confidenceThreshold) {
                    cv::Point pt1(int(xmin * frame.cols), int(ymin * frame.rows));
                    cv::Point pt2(int(xmax * frame.cols), int(ymax * frame.rows));
                    cv::rectangle(output_frame, pt1, pt2, cv::Scalar(0, 0, 255), 2);

                    std::string label = cv::format("Class %d: %.2f", classIdx, confidence);
                    int baseline = 0;
                    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                    cv::rectangle(output_frame, pt1, cv::Point(pt1.x + labelSize.width, pt1.y - labelSize.height - baseline), cv::Scalar(0, 0, 255), cv::FILLED);
                    cv::putText(output_frame, label, cv::Point(pt1.x, pt1.y - baseline), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                }
                idx += 5;
            }
            classIdx++;
        }
    } else {
        printf("No support in this example for NMS on CPU. See other Hailo examples\n");
    }

    for (auto& tensor : output_tensors) {
        free(tensor.data);
    }

    return output_frame;
}

int processImage(const std::string& imgFilename, hailort::InferModel& infer_model, hailort::ConfiguredInferModel& configured_infer_model) {
    cv::Mat img = cv::imread(imgFilename);
    if (img.empty()) {
        printf("Failed to load image %s\n", imgFilename.c_str());
        return 1;
    }

    cv::Mat processed_img = processFrame(img, infer_model, configured_infer_model);

    fs::path outputPath = fs::path("output_image") / fs::path(imgFilename).filename();
    cv::imwrite(outputPath.string(), processed_img);

    return 0;
}

int processVideo(const std::string& videoFilename, hailort::InferModel& infer_model, hailort::ConfiguredInferModel& configured_infer_model) {
    cv::VideoCapture cap(videoFilename);
    if (!cap.isOpened()) {
        printf("Error opening video file %s\n", videoFilename.c_str());
        return 1;
    }

    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);

    fs::path tempOutputPath = fs::path("output_video") / fs::path(videoFilename).filename().replace_extension(".mp4");
    fs::path finalOutputPath = fs::path("output_video") / ("processed_" + fs::path(videoFilename).filename().string());

    cv::VideoWriter video(tempOutputPath.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frame_width, frame_height));

    printf("Temporary video file size: %.2f MB\n", fs::file_size(tempOutputPath) / (1024.0 * 1024.0));

    if (!video.isOpened()) {
        printf("Error creating temporary output video file.\n");
        return 1;
    }

    cv::Mat frame;
    int frame_count = 0;
    while (cap.read(frame)) {
        cv::Mat processed_frame = processFrame(frame, infer_model, configured_infer_model);
        video.write(processed_frame);
        frame_count++;
    }

    cap.release();
    video.release();

    printf("Processed %d frames\n", frame_count);

    // Compress the video
    std::string ffmpeg_output;
    compressVideo(tempOutputPath, finalOutputPath, compressionCodec, ffmpeg_output);

    // Remove temporary file
    if (fs::exists(tempOutputPath)) {
        fs::remove(tempOutputPath);
    }

    // Check if the final output file exists and print its size
    if (fs::exists(finalOutputPath)) {
        try {
            uintmax_t fileSize = fs::file_size(finalOutputPath);
            printf("Final output video size: %.2f MB\n", fileSize / (1024.0 * 1024.0));
        } catch (const std::filesystem::filesystem_error& e) {
            printf("Error getting file size: %s\n", e.what());
        }
    } else {
        printf("Error: Final output video was not created at %s\n", finalOutputPath.string().c_str());
        // Check if the temporary file still exists
        if (fs::exists(tempOutputPath)) {
            printf("Temporary file still exists at %s\n", tempOutputPath.string().c_str());
        }
    }

    return 0;
}

int run() {
    using namespace hailort;

    Expected<std::unique_ptr<VDevice>> vdevice_exp = VDevice::create();
    if (!vdevice_exp) {
        printf("Failed to create vdevice\n");
        return vdevice_exp.status();
    }
    std::unique_ptr<VDevice> vdevice = vdevice_exp.release();

    Expected<std::shared_ptr<InferModel>> infer_model_exp = vdevice->create_infer_model(hefFile);
    if (!infer_model_exp) {
        printf("Failed to create infer model\n");
        return infer_model_exp.status();
    }
    std::shared_ptr<InferModel> infer_model = infer_model_exp.release();
    infer_model->set_hw_latency_measurement_flags(HAILO_LATENCY_MEASURE);
    infer_model->output()->set_nms_score_threshold(confidenceThreshold);
    infer_model->output()->set_nms_iou_threshold(nmsIoUThreshold);

    Expected<ConfiguredInferModel> configured_infer_model_exp = infer_model->configure();
    if (!configured_infer_model_exp) {
        printf("Failed to get configured infer model\n");
        return configured_infer_model_exp.status();
    }
    ConfiguredInferModel configured_infer_model = configured_infer_model_exp.release();

    if (useCamera) {
        return processCameraFeed(*infer_model, configured_infer_model);
    } else if (inputMode == "image_folder") {
        fs::create_directory("output_image");
        for (const auto & entry : fs::directory_iterator("input_image")) {
            if (entry.is_regular_file()) {
                std::string imgFilename = entry.path().string();
                printf("Processing image %s\n", imgFilename.c_str());
                int result = processImage(imgFilename, *infer_model, configured_infer_model);
                if (result != 0) {
                    printf("Failed to process %s\n", imgFilename.c_str());
                }
            }
        }
    } else if (inputMode == "video_folder") {
        fs::create_directory("output_video");
        for (const auto & entry : fs::directory_iterator("input_video")) {
            if (entry.is_regular_file()) {
                std::string videoFilename = entry.path().string();
                printf("Processing video %s\n", videoFilename.c_str());
                int result = processVideo(videoFilename, *infer_model, configured_infer_model);
                if (result != 0) {
                    printf("Failed to process %s\n", videoFilename.c_str());
                }
            }
        }
    } else {
        printf("Invalid input mode. Use --image_folder or --video_folder\n");
        return 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    parseArguments(argc, argv);

    printf("Using model: %s\n", hefFile.c_str());
    printf("Confidence threshold: %.2f\n", confidenceThreshold);
    printf("NMS IoU threshold: %.2f\n", nmsIoUThreshold);
    printf("Input mode: %s\n", inputMode.c_str());
    printf("Compression codec: %s\n", compressionCodec.empty() ? "None" : compressionCodec.c_str());
	
    if (useCamera) {
        printf("Camera mode: %s\n", cameraMode.c_str());
    }

    int status = run();
	
    if (status == 0)
        printf("SUCCESS\n");
    else
        printf("Failed with error code %d\n", status);
	
    return status;
}
