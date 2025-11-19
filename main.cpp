#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <filesystem>
#include "old_helper_functions.h"
#include "constants.h"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;


vector<FeatureVector> feature_matrix;
vector<string> Y;  // class labels

int getVerticalSymmetry(const Mat& image) {
    return image.rows;  // Todo actual symmetry
}

int getHorizontalSymmetry(const Mat& image) {
    return image.cols;  // Todo actual symmetry
}

FeatureVector getFeaturesFromImage(const Mat& image, bool show = false) {
    FeatureVector vector = {};
    vector[0] = getVerticalSymmetry(image);
    vector[1] = getHorizontalSymmetry(image);
    if (show) cout << vector[0] << " " << vector[1] << endl;
    return vector;
}

void readImagesFromFolder(const string& class_folder) {
    string folder_path = SYMBOLS_PATH + class_folder + "/";
    if (!fs::exists(folder_path)) {
        std::cerr << "ERROR: Folder does not exist: " << folder_path << std::endl;
        return;
    }
    bool first = true;
    cout << "Example from class " << class_folder << " ";
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            if (entry.path().extension() == ".jpg") {
                string image_path = entry.path().string();
                Mat image = imread(image_path, IMREAD_GRAYSCALE);
                if (image.empty()) {
                    std::cerr << "WARNING: Failed to read image: " << image_path << std::endl;
                    continue;
                }
                // showImg(image, image_path);  // just for debug
                // maybe preprocess input data?
                feature_matrix.push_back(getFeaturesFromImage(image, first));
                Y.push_back(class_folder);
                first = false;
            }
        }
    }
}

void readTrainingData() {
    for (const string& class_folder: FOLDER_NAMES) {
        printf("Reading from from %s\ncurrent feature matrix size %d\n", class_folder.c_str(), feature_matrix.size());
        readImagesFromFolder(class_folder);
    }
}

void test() {
    string path = SYMBOLS_PATH + FOLDER_NAMES[0] + "/!_7731.jpg";
    Mat img = imread(path,IMREAD_COLOR);
    if (img.empty()) {
        cerr << "ERROR: Failed to load image from path: " << path << endl;
        return;
    }
    showImg(img, "exclamation");
}

int main () {
    cout << "Hello OpenCV!";
    readTrainingData();
    return 0;
}

