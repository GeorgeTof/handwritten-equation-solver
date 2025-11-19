#pragma once

#include <string>
#include <vector>
#include <array>

/** FEATURES:
 *  vertical symmetry
 *  horizontal symmetry
 */
const int FEATURE_LENGTH = 2;
using FeatureVector = std::array<int, FEATURE_LENGTH>;

const int K = 11;

inline const std::string SYMBOLS_PATH = "../symbols/";

inline const std::vector<std::string> FOLDER_NAMES = {
    "!",
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
    "b", // to types of b, problem??
    "C"
};