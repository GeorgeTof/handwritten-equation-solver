#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>

void showImg(cv::Mat image, std::string ImageName = "Loaded image");

void showImgNoWait(cv::Mat image, std::string windowName);

std::vector<cv::Point2d> read_points_from_image(std::string filename);

void drawLineOnImg(std::string filename, cv::Point2d p1, cv::Point2d p2);

bool is_inside(int i, int j, int rows, int cols);

void writeMatToCSV(const cv::Mat& matrix, const std::string& filename, int precision = 8);

std::vector<cv::Point> getPoints(cv::Mat &image);

double distance(const cv::Point p1, const cv::Point p2);