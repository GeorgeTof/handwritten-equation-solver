#pragma once

#include <string>
#include <vector>
#include <array>
#include <iterator>

/** FEATURES:
 *  vertical symmetry
 *  horizontal symmetry
 *  vertical projection
 *  horizontal projection
 *  surface
 *  perimeter
 *  elongation
 *  7-10: Hu Moments (h1, h2, h3, h4)
 *  11-19: Zoning (3x3 grid densities)
 */
const int FEATURE_LENGTH = 2 + 2 + 3 + 4 + 9;

using FeatureVector = std::array<float, FEATURE_LENGTH>;
std::ostream& operator<<(std::ostream& os, const FeatureVector& v) {
    os << "[";
    std::copy(v.begin(), v.end(),
              std::ostream_iterator<float>(os, ", "));
    os << "\b\b]";
    return os;
}
const int TRAIN_LIMIT = 1500;
const int TEST_LIMIT = 600;

const int CANVAS_SIZE = 45;
const int CANVAS_SCALE = 10;  // Display scale for easier drawing

const int K = 11;

const double LEARNING_RATE = 0.01;
const int EPOCHS = 50;

enum ModelType {
    MODEL_KNN,
    MODEL_PERCEPTRON,
    MODEL_NAIVE_BAYES
};

inline const std::string TRAIN_SYMBOLS_PATH = "../symbols/";
inline const std::string TEST_SYMBOLS_PATH = "../testSymbols/";

inline const std::vector<std::string> FOLDER_NAMES = {
    "(",
    ")",
    "+",
    "-",
    "=",
    "times",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "b",
    "C"
};