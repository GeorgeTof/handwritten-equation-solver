# include "old_helper_functions.h"

#include <fstream>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;

void showImg(Mat image, string ImageName) {
    namedWindow(ImageName, WINDOW_AUTOSIZE);
    imshow(ImageName, image);
    cout << "Press any key to continue..." << endl;
    waitKey();
}

void showImgNoWait(Mat image, std::string windowName) {
    namedWindow(windowName, WINDOW_AUTOSIZE);
    imshow(windowName, image);
}

vector<Point2d> read_points_from_image(string filename) {
    vector<Point2d> points = vector<Point2d>();

    Mat img = imread(filename, cv::IMREAD_GRAYSCALE);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img.at<uchar>(i,j)==0) {
                Point2d p = Point2d(j, i);
                points.push_back(p);
            }
        }
    }

    return points;
}

void drawLineOnImg(string filename, Point2d p1, Point2d p2) {
    Mat img = imread(filename, cv::IMREAD_GRAYSCALE);

    line(img, p1, p2, Scalar(0, 0, 0));

    showImg(img);
}

bool is_inside(int i, int j, int rows, int cols) {
    if (i < 0 || j < 0 || i >= rows || j >= cols) {
        return false;
    }
    return true;
}

void writeMatToCSV(const cv::Mat& matrix, const std::string& filename, int precision) {

    if (matrix.type() != CV_32FC1) {
        std::cerr << "Error: Matrix is not of type CV_32FC1 (float)." << std::endl;
        return;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file for writing: " << filename << std::endl;
        return;
    }

    file << std::fixed << std::setprecision(precision);

    for (int i = 0; i < matrix.rows; ++i) {
        const float* row_ptr = matrix.ptr<float>(i);

        for (int j = 0; j < matrix.cols; ++j) {
            file << row_ptr[j];

            if (j < matrix.cols - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
}

std::vector<Point> getPoints(Mat &image) {
    std::vector<Point> points;
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            Vec3b pixel = image.at<cv::Vec3b>(i, j);
            if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0) {
                points.emplace_back(j, i);
            }
        }
    }
    return points;
}

double distance(const Point p1, const Point p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

