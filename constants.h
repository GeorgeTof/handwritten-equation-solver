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
 */
const int FEATURE_LENGTH = 2 + 2 + 3;

using FeatureVector = std::array<int, FEATURE_LENGTH>;
std::ostream& operator<<(std::ostream& os, const FeatureVector& v) {
    os << "[";
    std::copy(v.begin(), v.end(),
              std::ostream_iterator<int>(os, ", "));
    os << "\b\b]";
    return os;
}

const int K = 11;

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