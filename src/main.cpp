#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp> 
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

// Manual CCM implementation
class ManualCCM {
private:
    cv::Mat colorMatrix; // 3x3 color correction matrix
    
    // Auto white balance function with ROI
    cv::Mat autoWhiteBalance(const cv::Mat& image, double roiRatio = 0.5) {
        if (image.empty()) {
            return cv::Mat();
        }
        
        // Calculate ROI dimensions (center portion of the image)
        int roiWidth = static_cast<int>(image.cols * roiRatio);
        int roiHeight = static_cast<int>(image.rows * roiRatio);
        int roiX = (image.cols - roiWidth) / 2;
        int roiY = (image.rows - roiHeight) / 2;
        
        // Extract ROI
        cv::Rect roi(roiX, roiY, roiWidth, roiHeight);
        cv::Mat roiImage = image(roi);
        
        // Calculate mean values for each channel in ROI
        cv::Scalar meanValues = cv::mean(roiImage);
        
        // Calculate the maximum mean value (target for white balance)
        double maxMean = std::max({meanValues[0], meanValues[1], meanValues[2]});
        
        // Calculate scaling factors to make all channels equal to the maximum
        double scaleR = maxMean / (meanValues[0] > 0 ? meanValues[0] : 1.0);
        double scaleG = maxMean / (meanValues[1] > 0 ? meanValues[1] : 1.0);
        double scaleB = maxMean / (meanValues[2] > 0 ? meanValues[2] : 1.0);
        
        // Apply white balance to the entire image
        cv::Mat balancedImage = image.clone();
        std::vector<cv::Mat> channels(3);
        cv::split(balancedImage, channels);
        
        // Scale each channel
        channels[0] *= scaleR; // Red channel
        channels[1] *= scaleG; // Green channel
        channels[2] *= scaleB; // Blue channel
        
        // Merge channels back
        cv::merge(channels, balancedImage);
        
        // Clamp values to valid range [0, 255]
        cv::threshold(balancedImage, balancedImage, 255, 255, cv::THRESH_TRUNC);
        
        std::cout << "White balance factors - R:" << scaleR << " G:" << scaleG << " B:" << scaleB << std::endl;
        
        return balancedImage;
    }
    
public:
    ManualCCM() {
        // Initialize with identity matrix (no correction)
        colorMatrix = cv::Mat::eye(3, 3, CV_32F);
        
        // Example: Simple color correction matrix
        // This matrix can be adjusted for different color corrections
        // R' = 1.2*R + 0.1*G + 0.0*B
        // G' = 0.0*R + 1.1*G + 0.0*B  
        // B' = 0.0*R + 0.0*G + 0.9*B
        colorMatrix.at<float>(0,0) = 1.3f; // Red channel gain
        colorMatrix.at<float>(0,1) = -0.141f;
        colorMatrix.at<float>(0,2) = -0.168f;
        colorMatrix.at<float>(1,0) = -0.191f;
        colorMatrix.at<float>(1,1) = 1.2f; // Green channel gain
        colorMatrix.at<float>(1,2) = -0.118f;
        colorMatrix.at<float>(2,0) = -0.181f;
        colorMatrix.at<float>(2,1) = -0.183f;
        colorMatrix.at<float>(2,2) = 1.25f; // Blue channel gain
        
        std::cout << "Color correction matrix:" << std::endl << colorMatrix << std::endl;
    }
    
    // Set custom color correction matrix
    void setColorMatrix(const cv::Mat& matrix) {
        if (matrix.rows == 3 && matrix.cols == 3 && matrix.type() == CV_32F) {
            colorMatrix = matrix.clone();
            std::cout << "Updated color correction matrix:" << std::endl << colorMatrix << std::endl;
        } else {
            std::cout << "Error: Matrix must be 3x3 float type" << std::endl;
        }
    }
    
    // Main color correction function
    cv::Mat correctColors(const cv::Mat& image) {
        if (image.empty()) {
            return cv::Mat();
        }
        
        // Apply auto white balance first (ROI = 0.5)
        cv::Mat whiteBalancedImage = autoWhiteBalance(image, 0.5);
        
        // Convert BGR to RGB first
        cv::Mat rgbImage;
        cv::cvtColor(whiteBalancedImage, rgbImage, cv::COLOR_BGR2RGB);
        
        // Convert to float for precise calculations
        cv::Mat floatImage;
        rgbImage.convertTo(floatImage, CV_32F);
        
        // Create output image
        cv::Mat correctedImage = cv::Mat::zeros(image.size(), CV_32FC3);
        
        // Apply color correction matrix to each pixel
        for (int y = 0; y < floatImage.rows; y++) {
            for (int x = 0; x < floatImage.cols; x++) {
                cv::Vec3f pixel = floatImage.at<cv::Vec3f>(y, x);
                
                // Apply 3x3 matrix transformation
                cv::Vec3f correctedPixel;
                correctedPixel[0] = colorMatrix.at<float>(0,0) * pixel[0] + 
                                   colorMatrix.at<float>(0,1) * pixel[1] + 
                                   colorMatrix.at<float>(0,2) * pixel[2];
                correctedPixel[1] = colorMatrix.at<float>(1,0) * pixel[0] + 
                                   colorMatrix.at<float>(1,1) * pixel[1] + 
                                   colorMatrix.at<float>(1,2) * pixel[2];
                correctedPixel[2] = colorMatrix.at<float>(2,0) * pixel[0] + 
                                   colorMatrix.at<float>(2,1) * pixel[1] + 
                                   colorMatrix.at<float>(2,2) * pixel[2];
                
                // Clamp values to valid range [0, 255]
                correctedPixel[0] = std::max(0.0f, std::min(255.0f, correctedPixel[0]));
                correctedPixel[1] = std::max(0.0f, std::min(255.0f, correctedPixel[1]));
                correctedPixel[2] = std::max(0.0f, std::min(255.0f, correctedPixel[2]));
                
                correctedImage.at<cv::Vec3f>(y, x) = correctedPixel;
            }
        }
        
        // Convert back to 8-bit
        cv::Mat result;
        correctedImage.convertTo(result, CV_8U);
        
        // Convert back to BGR for OpenCV display
        cv::Mat bgrResult;
        cv::cvtColor(result, bgrResult, cv::COLOR_RGB2BGR);
        
        return bgrResult;
    }
    
};

int main(int argc, char* argv[])
{
    // RTSP URL - can be passed as command line argument or set default
    std::string rtspUrl = "rtsp://172.32.0.93/live/0"; // Default RTSP URL
    
    if (argc > 1) {
        rtspUrl = argv[1];
    }
    
    std::cout << "Connecting to RTSP stream: " << rtspUrl << std::endl;
    
    // Open RTSP stream
    cv::VideoCapture cap;
    bool streamOpened = false;
    
    // Try different backends for RTSP
    std::vector<int> backends = {cv::CAP_FFMPEG, cv::CAP_GSTREAMER, cv::CAP_ANY};
    for (int backend : backends) {
        cap = cv::VideoCapture(rtspUrl, backend);
        if (cap.isOpened()) {
            std::cout << "RTSP stream opened successfully with backend " << backend << std::endl;
            streamOpened = true;
            break;
        }
    }
    
    if (!streamOpened) {
        std::cout << "Error: Could not open RTSP stream: " << rtspUrl << std::endl;
        std::cout << "Please check:" << std::endl;
        std::cout << "1. RTSP URL is correct" << std::endl;
        std::cout << "2. Network connectivity" << std::endl;
        std::cout << "3. Camera is online and streaming" << std::endl;
        return 1;
    }
    
    // Set stream properties
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);
    
    // Print actual stream properties
    double actualWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double actualHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double actualFPS = cap.get(cv::CAP_PROP_FPS);
    
    std::cout << "Stream properties:" << std::endl;
    std::cout << "  Width: " << actualWidth << std::endl;
    std::cout << "  Height: " << actualHeight << std::endl;
    std::cout << "  FPS: " << actualFPS << std::endl;
    
    // Create manual CCM processor
    ManualCCM ccmProcessor;
    
    cv::Mat frame;
    int frameCount = 0;
    
    std::cout << "Reading frames from RTSP stream..." << std::endl;
    
    while (true) {
        // Capture frame from RTSP
        cap >> frame;
        
        // Check if frame is empty
        if (frame.empty()) {
            std::cout << "Warning: Empty frame received" << std::endl;
            continue;
        }
        
        frameCount++;

        // Apply manual color correction
        cv::Mat corrected = ccmProcessor.correctColors(frame);

        cv::imshow("Frame", corrected);
        
        // Wait for key press (essential for GUI to work)
        char key = cv::waitKey(1);
        if (key == 27 || key == 'q') { // ESC or 'q' to quit
            break;
        }
    }
    
    // Clean up
    cap.release();
    cv::destroyAllWindows();
    
    std::cout << "RTSP session ended. Processed " << frameCount << " frames." << std::endl;
    return 0;
}
