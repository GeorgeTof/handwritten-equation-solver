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
#include <unordered_map>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;


vector<FeatureVector> feature_matrix;
vector<string> Y;  // class labels

float getVerticalSymmetry(const Mat& image) {
    int left, right;
    int score = 0;
    for (left = 0; left < image.cols; ++left) {
        right = image.cols - 1 - left;
        if (left >= right) break;
        for (int i = 0; i < image.rows; ++i)
            if (image.at<uchar>(i, left) < 255 && image.at<uchar>(i, right) < 255)
                score ++;
    }
    return (float)score / (image.cols * image.rows) * 2.0f;
}

float getHorizontalSymmetry(const Mat& image) {
    int up, down;
    int score = 0;
    for (up = 0; up < image.rows; up++) {
        down = image.rows - 1 - up;
        if (down <= up) break;
        for (int j = 0; j < image.cols; j++) {
            if (image.at<uchar>(down, j) < 255 && image.at<uchar>(up, j) < 255)
                score++;
        }
    }
    return (float)score / (image.cols * image.rows) * 2.0f;
}

float getVerticalProjection(const Mat& image) {
    int count = 0;
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            if (image.at<uchar>(i, j) < 255) {
                count++;
                break;
            }
        }
    }
    return (float)count/image.rows;
}

float getHorizontalProjection(const Mat& image) {
    int count = 0;
    for(int j = 0; j < image.cols; j++) {
        for(int i = 0; i < image.rows; i++) {
            if (image.at<uchar>(i, j) < 255) {
                count++;
                break;
            }
        }
    }
    return (float)count/image.cols;
}

float getSurface(const Mat& image) {
    int count = 0;
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            if (image.at<uchar>(i, j) < 255) {
                count++;
            }
        }
    }

    return (float)count/(image.rows * image.cols);
}

float getPerimeter(const Mat& image) {
    int count = 0;
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            bool isBoundary = false;
            if(image.at<uchar>(i, j) < 255) {
                if(i == 0 || i == image.rows - 1 || j == 0 || j == image.cols - 1) {
                    isBoundary = true;
                }
                else if(image.at<uchar>(i-1, j) == 255 ||
                    image.at<uchar>(i+1, j) == 255 ||
                    image.at<uchar>(i, j-1) == 255 ||
                    image.at<uchar>(i, j+1) == 255){

                    isBoundary = true;
                }
                if(isBoundary) {count++;}
            }
        }
    }

    return (float)count/(image.rows * image.cols);
}

float getElongation(const Mat& image) {
    int minRow = image.rows, maxRow = 0;
    int minCol = image.cols, maxCol = 0;

    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            if(image.at<uchar>(i, j) < 255) {
                if(i < minRow) minRow = i;
                if(i > maxRow) maxRow = i;
                if(j < minCol) minCol = j;
                if(j > maxCol) maxCol = j;
            }
        }
    }

    int height = maxRow - minRow + 1;
    int width = maxCol - minCol + 1;

    if(height == 0 || width == 0) return 1.0f;

    return (float)min(height, width)/max(height,width);
}

FeatureVector getFeaturesFromImage(const Mat& image, bool show = false) {
    FeatureVector vector = {};
    vector[0] = getVerticalSymmetry(image);
    vector[1] = getHorizontalSymmetry(image);
    vector[2] = getVerticalProjection(image);
    vector[3] = getHorizontalProjection(image);
    vector[4] = getSurface(image);
    vector[5] = getPerimeter(image);
    vector[6] = getElongation(image);
    if (show) cout << vector << endl;
    return vector;
}

void readImagesFromFolder(const string& class_folder) {
    string folder_path = TRAIN_SYMBOLS_PATH + class_folder + "/";
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

string knnForImage(const Mat& image) {
    FeatureVector thisVector = getFeaturesFromImage(image);

    vector<pair<float, string>> neighbors;

    int idx = 0;
    for (const FeatureVector& v: feature_matrix) {
        float dist = 0.0;
        for (int i = 0; i < FEATURE_LENGTH; i++) {
            dist += pow(thisVector[i] - v[i], 2);
        }
        dist /= FEATURE_LENGTH;
        neighbors.push_back(pair(dist, Y[idx++]));
    }

    sort(neighbors.begin(), neighbors.end());

    std::unordered_map<string, int> classVotes;
    int mostVotes = -1;
    string mostVotedClass;

    for (int i = 0; i < K; i++) {
        string label = neighbors[i].second;
        classVotes[label]++;
    }

    for (const auto& [label, count] : classVotes) {
        if (count > mostVotes) {
            mostVotes = count;
            mostVotedClass = label;
        }
    }

    return mostVotedClass;
}

void testSingleImage() {
    string path = TRAIN_SYMBOLS_PATH + FOLDER_NAMES[0] + "/!_7731.jpg";
    Mat img = imread(path,IMREAD_COLOR);
    if (img.empty()) {
        cerr << "ERROR: Failed to load image from path: " << path << endl;
        return;
    }
    showImg(img, "exclamation");
}

void testKnn() {
    string symbolClass = FOLDER_NAMES[0];
    string path = TEST_SYMBOLS_PATH + symbolClass + "/(_22.jpg";
    Mat img = imread(path,IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "ERROR: Failed to load image from path: " << path << endl;
        return;
    }
    showImgNoWait(img, symbolClass);
    printf("Prediction for %s is %s\n\n", symbolClass.c_str(), knnForImage(img).c_str());

    symbolClass = FOLDER_NAMES[1];
    path = TEST_SYMBOLS_PATH + symbolClass + "/)_15.jpg";
    img = imread(path,IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "ERROR: Failed to load image from path: " << path << endl;
        return;
    }
    showImgNoWait(img, symbolClass);
    printf("Prediction for %s is %s\n\n", symbolClass.c_str(), knnForImage(img).c_str());

    symbolClass = FOLDER_NAMES[2];
    path = TEST_SYMBOLS_PATH + symbolClass + "/+_10.jpg";
    img = imread(path,IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "ERROR: Failed to load image from path: " << path << endl;
        return;
    }
    printf("Prediction for %s is %s\n\n", symbolClass.c_str(), knnForImage(img).c_str());
    showImgNoWait(img, symbolClass);

    symbolClass = FOLDER_NAMES[3];
    path = TEST_SYMBOLS_PATH + symbolClass + "/-_121.jpg";
    img = imread(path,IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "ERROR: Failed to load image from path: " << path << endl;
        return;
    }
    printf("Prediction for %s is %s\n\n", symbolClass.c_str(), knnForImage(img).c_str());
    showImgNoWait(img, symbolClass);

    symbolClass = FOLDER_NAMES[4];
    path = TEST_SYMBOLS_PATH + symbolClass + "/=_3.jpg";
    img = imread(path,IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "ERROR: Failed to load image from path: " << path << endl;
        return;
    }
    printf("Prediction for %s is %s\n\n", symbolClass.c_str(), knnForImage(img).c_str());
    showImgNoWait(img, symbolClass);
}

int main () {
    cout << "Hello OpenCV!";
    readTrainingData();
    testKnn();
    return 0;
}

