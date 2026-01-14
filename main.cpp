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
#include <iomanip>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

class Perceptron {
public:
    vector<double> weights;
    double bias;
    double learning_rate;

    Perceptron(int num_features, double lr = LEARNING_RATE)
        : learning_rate(lr), bias(0.0) {
        weights.resize(num_features);
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-0.01, 0.01);
        for (int i = 0; i < num_features; i++) weights[i] = dis(gen);
    }

    double getActivation(const FeatureVector& features) {
        double activation = bias;
        for (int i = 0; i < FEATURE_LENGTH; i++) {
            activation += weights[i] * features[i];
        }
        return activation;
    }

    int predictBinary(const FeatureVector& features) {
        return getActivation(features) >= 0.0 ? 1 : -1;
    }

    void trainSample(const FeatureVector& features, int target) {
        int prediction = predictBinary(features);
        int error = target - prediction;

        if (error != 0) {
            for (int i = 0; i < FEATURE_LENGTH; i++) {
                weights[i] += learning_rate * error * features[i];
            }
            bias += learning_rate * error;
        }
    }
};

class MultiClassPerceptron {
private:
    vector<Perceptron> classifiers;
    vector<string> class_labels;

public:
    MultiClassPerceptron() {
        class_labels = FOLDER_NAMES;
        for (size_t i = 0; i < class_labels.size(); i++) {
            classifiers.push_back(Perceptron(FEATURE_LENGTH));
        }
    }

    void train(const vector<FeatureVector>& X, const vector<string>& y) {
        cout << "Training " << class_labels.size() << " Perceptrons for " << EPOCHS << " epochs..." << endl;

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            int total_errors = 0;

            for (size_t i = 0; i < class_labels.size(); i++) {
                string target_class = class_labels[i];

                for (size_t j = 0; j < X.size(); j++) {
                    int label = (y[j] == target_class) ? 1 : -1;

                    if (classifiers[i].predictBinary(X[j]) != label) total_errors++;

                    classifiers[i].trainSample(X[j], label);
                }
            }

            if ((epoch + 1) % 10 == 0) {
                cout << "Epoch " << (epoch + 1) << "/" << EPOCHS << " complete." << endl;
            }
        }
        cout << "Training Complete!" << endl;
    }

    string predict(const FeatureVector& features) {
        double max_score = -1e9; // Start very low
        int best_index = -1;

        for (size_t i = 0; i < classifiers.size(); i++) {
            double score = classifiers[i].getActivation(features);
            if (score > max_score) {
                max_score = score;
                best_index = i;
            }
        }

        if (best_index != -1) return class_labels[best_index];
        return "Unknown";
    }
};

class GaussianNaiveBayes {
private:
    struct ClassStats {
        vector<double> means;
        vector<double> variances;
        double prior;
        int sample_count;
    };

    unordered_map<string, ClassStats> model;

    const double EPSILON = 1e-9;

public:
    void train(const vector<FeatureVector>& X, const vector<string>& Y) {
        model.clear();
        int total_samples = X.size();

        for (size_t i = 0; i < X.size(); i++) {
            string label = Y[i];

            if (model.find(label) == model.end()) {
                model[label].means.resize(FEATURE_LENGTH, 0.0);
                model[label].variances.resize(FEATURE_LENGTH, 0.0);
                model[label].sample_count = 0;
            }

            model[label].sample_count++;

            for (int f = 0; f < FEATURE_LENGTH; f++) {
                model[label].means[f] += X[i][f];
            }
        }

        for (auto& [label, stats] : model) {
            stats.prior = (double)stats.sample_count / total_samples;

            for (int f = 0; f < FEATURE_LENGTH; f++) {
                stats.means[f] /= stats.sample_count;
            }
        }

        for (size_t i = 0; i < X.size(); i++) {
            string label = Y[i];
            ClassStats& stats = model[label];

            for (int f = 0; f < FEATURE_LENGTH; f++) {
                double diff = X[i][f] - stats.means[f];
                stats.variances[f] += diff * diff;
            }
        }

        for (auto& [label, stats] : model) {
            for (int f = 0; f < FEATURE_LENGTH; f++) {
                stats.variances[f] /= stats.sample_count;
                // Add epsilon to avoid division by zero later
                stats.variances[f] += EPSILON;
            }
        }

        cout << "Naive Bayes trained on " << model.size() << " classes." << endl;
    }

    double calculateLogLikelihood(double x, double mean, double var) {
        return -0.5 * log(2 * M_PI * var) - pow(x - mean, 2) / (2 * var);
    }

    string predict(const FeatureVector& features) {
        string best_class;
        double max_log_prob = -1e18;

        for (const auto& [label, stats] : model) {
            double log_prob = log(stats.prior);

            for (int f = 0; f < FEATURE_LENGTH; f++) {
                log_prob += calculateLogLikelihood(features[f], stats.means[f], stats.variances[f]);
            }

            if (log_prob > max_log_prob) {
                max_log_prob = log_prob;
                best_class = label;
            }
        }
        return best_class;
    }
};

vector<FeatureVector> feature_matrix;
vector<string> Y;  // class labels

FeatureVector global_means;
FeatureVector global_stddevs;
bool is_normalized = false;

MultiClassPerceptron global_perceptron;
GaussianNaiveBayes nb_model;

vector<vector<int>> confusion_matrix;

// Interactive canvas for drawing
struct DrawingCanvas {
    Mat canvas;      // 45x45 internal canvas
    Mat display;     // Scaled display canvas
    bool drawing;    // Is mouse button pressed?
    Point lastPoint; // Last drawn point
};

DrawingCanvas g_canvas;
string g_windowName;  // Current window name for drawing

void updateDisplay() {
    resize(g_canvas.canvas, g_canvas.display,
           Size(CANVAS_SIZE * CANVAS_SCALE, CANVAS_SIZE * CANVAS_SCALE),
           0, 0, INTER_NEAREST);
    imshow(g_windowName, g_canvas.display);
}

void mouseCallback(int event, int x, int y, int flags, void* userdata) {
    // Convert display coordinates to canvas coordinates
    int cx = x / CANVAS_SCALE;
    int cy = y / CANVAS_SCALE;

    // Clamp to canvas bounds
    cx = max(0, min(cx, CANVAS_SIZE - 1));
    cy = max(0, min(cy, CANVAS_SIZE - 1));

    if (event == EVENT_LBUTTONDOWN) {
        g_canvas.drawing = true;
        g_canvas.lastPoint = Point(cx, cy);
        g_canvas.canvas.at<uchar>(cy, cx) = 0;  // Draw black pixel
        updateDisplay();
    }
    else if (event == EVENT_MOUSEMOVE && g_canvas.drawing) {
        // Draw line from last point to current point
        line(g_canvas.canvas, g_canvas.lastPoint, Point(cx, cy), Scalar(0), 1);
        g_canvas.lastPoint = Point(cx, cy);
        updateDisplay();
    }
    else if (event == EVENT_LBUTTONUP) {
        g_canvas.drawing = false;
    }
}

string predictWithModel(const Mat& image, ModelType model);

void interactivePrediction(ModelType currentModel) {
    // Initialize white canvas (255 = white, 0 = black for drawn pixels)
    g_canvas.canvas = Mat(CANVAS_SIZE, CANVAS_SIZE, CV_8UC1, Scalar(255));
    g_canvas.display = Mat(CANVAS_SIZE * CANVAS_SCALE, CANVAS_SIZE * CANVAS_SCALE, CV_8UC1);
    g_canvas.drawing = false;

    g_windowName = "Draw Character (Press ENTER to predict, ESC to cancel, 'r' to reset)";
    namedWindow(g_windowName, WINDOW_AUTOSIZE);
    setMouseCallback(g_windowName, mouseCallback);

    updateDisplay();

    cout << "\n--- Interactive Drawing Mode ---" << endl;
    cout << "Draw a character using your mouse." << endl;
    cout << "Press ENTER to predict, 'r' to reset canvas, ESC to cancel." << endl;

    while (true) {
        int key = waitKey(10) & 0xFF;

        if (key == 13 || key == 10) {  // Enter key
            // Predict using the selected model
            string prediction = predictWithModel(g_canvas.canvas, currentModel);
            cout << "\n=========================================" << endl;
            cout << "PREDICTION: " << prediction << endl;
            cout << "=========================================" << endl;
            break;
        }
        else if (key == 27) {  // ESC key
            cout << "Cancelled." << endl;
            break;
        }
        else if (key == 'r' || key == 'R') {  // Reset
            g_canvas.canvas = Mat(CANVAS_SIZE, CANVAS_SIZE, CV_8UC1, Scalar(255));
            updateDisplay();
            cout << "Canvas reset." << endl;
        }
    }

    destroyWindow(g_windowName);
}

// ============== EQUATION MODE ==============

enum DrawAction {
    ACTION_ACCEPT,
    ACTION_RETRY,
    ACTION_SOLVE,
    ACTION_CANCEL
};

// Convert predicted symbol to equation character
string symbolToChar(const string& symbol) {
    if (symbol == "times") return "*";
    return symbol;
}

// Draw character and get user's decision
pair<string, DrawAction> drawAndDecide(ModelType currentModel, const string& currentEquation) {
    g_canvas.canvas = Mat(CANVAS_SIZE, CANVAS_SIZE, CV_8UC1, Scalar(255));
    g_canvas.display = Mat(CANVAS_SIZE * CANVAS_SCALE, CANVAS_SIZE * CANVAS_SCALE, CV_8UC1);
    g_canvas.drawing = false;

    g_windowName = "Equation Mode";
    namedWindow(g_windowName, WINDOW_AUTOSIZE);
    setMouseCallback(g_windowName, mouseCallback);
    updateDisplay();

    string prediction;
    bool predicted = false;

    cout << "\nCurrent equation: " << (currentEquation.empty() ? "(empty)" : currentEquation) << endl;
    cout << "Draw next character. Press ENTER to predict, 's' to solve, 'r' to reset, ESC to cancel." << endl;

    while (true) {
        int key = waitKey(10) & 0xFF;

        // 's' to solve works at any time
        if (key == 's' || key == 'S') {
            destroyWindow(g_windowName);
            return {"", ACTION_SOLVE};
        }

        if (!predicted) {
            if (key == 13 || key == 10) {  // Enter - predict
                prediction = predictWithModel(g_canvas.canvas, currentModel);
                string displayChar = symbolToChar(prediction);
                cout << "\nPredicted: " << displayChar << endl;
                cout << "[a] Accept  [t] Try again  [s] Solve equation  [ESC] Cancel" << endl;
                predicted = true;
            }
            else if (key == 'r' || key == 'R') {
                g_canvas.canvas = Mat(CANVAS_SIZE, CANVAS_SIZE, CV_8UC1, Scalar(255));
                updateDisplay();
                cout << "Canvas reset." << endl;
            }
            else if (key == 27) {  // ESC
                destroyWindow(g_windowName);
                return {"", ACTION_CANCEL};
            }
        }
        else {
            // After prediction, wait for user decision
            if (key == 'a' || key == 'A') {
                destroyWindow(g_windowName);
                return {prediction, ACTION_ACCEPT};
            }
            else if (key == 't' || key == 'T') {
                // Reset and try again
                g_canvas.canvas = Mat(CANVAS_SIZE, CANVAS_SIZE, CV_8UC1, Scalar(255));
                g_canvas.drawing = false;
                updateDisplay();
                predicted = false;
                cout << "\nDraw again. Press ENTER to predict, 'r' to reset." << endl;
            }
            else if (key == 's' || key == 'S') {
                destroyWindow(g_windowName);
                return {"", ACTION_SOLVE};
            }
            else if (key == 27) {  // ESC
                destroyWindow(g_windowName);
                return {"", ACTION_CANCEL};
            }
        }
    }
}

// ============== EXPRESSION EVALUATOR ==============

class ExpressionParser {
private:
    string expr;
    size_t pos;
    char variable;      // 'A', 'b', or 'C' if present
    bool hasVariable;

    char peek() {
        while (pos < expr.size() && expr[pos] == ' ') pos++;
        return pos < expr.size() ? expr[pos] : '\0';
    }

    char get() {
        char c = peek();
        if (pos < expr.size()) pos++;
        return c;
    }

    // Parse a number or variable, returns pair<coefficient, constant>
    // For a number n: returns {0, n}
    // For a variable: returns {1, 0}
    // For coefficient*variable: handled at higher level
    pair<double, double> parseNumber() {
        char c = peek();

        // Check for variable
        if (c == 'A' || c == 'b' || c == 'C') {
            get();
            variable = c;
            hasVariable = true;
            return {1.0, 0.0};  // coefficient=1, constant=0
        }

        // Parse number
        double num = 0;
        bool hasDigit = false;
        while (isdigit(peek())) {
            num = num * 10 + (get() - '0');
            hasDigit = true;
        }

        if (!hasDigit) {
            throw runtime_error("Expected number at position " + to_string(pos));
        }

        return {0.0, num};  // coefficient=0, constant=num
    }

    // Parse factor: number, variable, or (expression)
    pair<double, double> parseFactor() {
        if (peek() == '(') {
            get();  // consume '('
            auto result = parseExpression();
            if (peek() != ')') {
                throw runtime_error("Expected ')'");
            }
            get();  // consume ')'
            return result;
        }
        return parseNumber();
    }

    // Parse term: factor (* factor)*
    pair<double, double> parseTerm() {
        auto left = parseFactor();

        while (peek() == '*') {
            get();  // consume '*'
            auto right = parseFactor();

            // Multiply: (a*x + b) * (c*x + d) = ac*x^2 + (ad+bc)*x + bd
            // We only support linear equations, so either left or right must have coef=0
            if (left.first != 0 && right.first != 0) {
                throw runtime_error("Quadratic equations not supported");
            }

            double newCoef = left.first * right.second + left.second * right.first;
            double newConst = left.second * right.second;
            left = {newCoef, newConst};
        }

        return left;
    }

    // Parse expression: term ((+|-) term)*
    pair<double, double> parseExpression() {
        auto left = parseTerm();

        while (peek() == '+' || peek() == '-') {
            char op = get();
            auto right = parseTerm();

            if (op == '+') {
                left = {left.first + right.first, left.second + right.second};
            } else {
                left = {left.first - right.first, left.second - right.second};
            }
        }

        return left;
    }

public:
    ExpressionParser() : pos(0), variable('\0'), hasVariable(false) {}

    // Evaluate a simple arithmetic expression (no variables)
    double evaluate(const string& expression) {
        expr = expression;
        pos = 0;
        hasVariable = false;
        variable = '\0';

        auto result = parseExpression();

        if (result.first != 0) {
            throw runtime_error("Cannot evaluate expression with variables");
        }

        return result.second;
    }

    // Solve equation: returns the value of the variable
    // Equation format: left = right
    double solveEquation(const string& equation) {
        size_t eqPos = equation.find('=');
        if (eqPos == string::npos) {
            throw runtime_error("No '=' found in equation");
        }

        string leftStr = equation.substr(0, eqPos);
        string rightStr = equation.substr(eqPos + 1);

        // Parse left side
        expr = leftStr;
        pos = 0;
        hasVariable = false;
        auto left = parseExpression();
        char leftVar = variable;
        bool leftHasVar = hasVariable;

        // Parse right side
        expr = rightStr;
        pos = 0;
        hasVariable = false;
        variable = '\0';
        auto right = parseExpression();

        // Combine: left = right  =>  left - right = 0
        // (leftCoef - rightCoef) * x + (leftConst - rightConst) = 0
        double coef = left.first - right.first;
        double constant = left.second - right.second;

        if (abs(coef) < 1e-10) {
            if (abs(constant) < 1e-10) {
                throw runtime_error("Infinite solutions");
            } else {
                throw runtime_error("No solution");
            }
        }

        // coef * x + constant = 0  =>  x = -constant / coef
        return -constant / coef;
    }

    char getVariable() const { return variable; }
    bool foundVariable() const { return hasVariable; }
};

void equationMode(ModelType currentModel) {
    string equation;

    cout << "\n=========================================" << endl;
    cout << "        EQUATION MODE" << endl;
    cout << "=========================================" << endl;
    cout << "Draw symbols to build an equation." << endl;
    cout << "Supported: digits (0-9), operators (+,-,*), parentheses, = and variables (A,b,C)" << endl;

    while (true) {
        auto [prediction, action] = drawAndDecide(currentModel, equation);

        if (action == ACTION_CANCEL) {
            cout << "Equation mode cancelled." << endl;
            return;
        }
        else if (action == ACTION_ACCEPT) {
            string ch = symbolToChar(prediction);
            equation += ch;
            cout << "Added '" << ch << "'. Equation: " << equation << endl;
        }
        else if (action == ACTION_RETRY) {
            // Loop continues, user will draw again
            continue;
        }
        else if (action == ACTION_SOLVE) {
            if (equation.empty()) {
                cout << "No equation to solve!" << endl;
                continue;
            }

            cout << "\n=========================================" << endl;
            cout << "Solving: " << equation << endl;

            try {
                ExpressionParser parser;

                if (equation.find('=') != string::npos) {
                    // Equation with '=' - solve for variable
                    double result = parser.solveEquation(equation);
                    char var = parser.getVariable();
                    cout << "SOLUTION: " << var << " = " << result << endl;
                } else {
                    // Simple arithmetic expression
                    double result = parser.evaluate(equation);
                    cout << "RESULT: " << equation << " = " << result << endl;
                }
            } catch (const exception& e) {
                cout << "ERROR: " << e.what() << endl;
            }

            cout << "=========================================" << endl;
            return;
        }
    }
}

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

    return (float)count/(image.rows * image.cols); //todo: impart la arie in loc de dimensiunea imaginii
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

void computeHuMoments(const Mat& image, FeatureVector& vector, int start_index) {
    Moments m = moments(image, true);

    double hu[7];
    HuMoments(m, hu);

    for (int i = 0; i < 4; i++) {
        if (abs(hu[i]) < 1e-10) {
            vector[start_index + i] = 0.0f;
        } else {
            vector[start_index + i] = (float)(-1.0 * copysign(1.0, hu[i]) * log10(abs(hu[i])));
        }
    }
}

void computeZoningFeatures(const Mat& image, FeatureVector& vector, int start_index) {
    int gridRows = 3;
    int gridCols = 3;

    int cellHeight = image.rows / gridRows;
    int cellWidth = image.cols / gridCols;

    int featureIdx = 0;

    for (int i = 0; i < gridRows; i++) {
        for (int j = 0; j < gridCols; j++) {
            int r_start = i * cellHeight;
            int c_start = j * cellWidth;

            int r_end = (i == gridRows - 1) ? image.rows : (i + 1) * cellHeight;
            int c_end = (j == gridCols - 1) ? image.cols : (j + 1) * cellWidth;

            if (r_end <= r_start || c_end <= c_start) {
                vector[start_index + featureIdx] = 0.0f;
                featureIdx++;
                continue;
            }

            cv::Rect roi(c_start, r_start, c_end - c_start, r_end - r_start);
            cv::Mat cell = image(roi);

            int count = countNonZero(cell < 255);
            float area = (float)(cell.rows * cell.cols);

            if (area > 0) {
                vector[start_index + featureIdx] = (float)count / area;
            } else {
                vector[start_index + featureIdx] = 0.0f;
            }

            featureIdx++;
        }
    }
}

void normalizeVector(FeatureVector& v) {
    if (!is_normalized) return;

    for (int i = 0; i < FEATURE_LENGTH; i++) {
        if (global_stddevs[i] > 1e-6) {
            v[i] = (v[i] - global_means[i]) / global_stddevs[i];
        } else {
            v[i] = 0.0;
        }
    }
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

    computeHuMoments(image, vector, 7);

    computeZoningFeatures(image, vector, 11);

    if (show) cout << vector << endl;

    if (is_normalized) normalizeVector(vector);

    return vector;
}

void performZNormalization() {
    if (feature_matrix.empty()) return;

    int n_samples = feature_matrix.size();

    global_means.fill(0.0f);
    global_stddevs.fill(0.0f);

    // 2. Calculate Means
    for (const auto& v : feature_matrix) {
        for (int i = 0; i < FEATURE_LENGTH; i++) {
            global_means[i] += v[i];
        }
    }
    for (int i = 0; i < FEATURE_LENGTH; i++) {
        global_means[i] /= n_samples;
    }

    // 3. Calculate StdDevs
    for (const auto& v : feature_matrix) {
        for (int i = 0; i < FEATURE_LENGTH; i++) {
            float diff = v[i] - global_means[i];
            global_stddevs[i] += diff * diff;
        }
    }
    for (int i = 0; i < FEATURE_LENGTH; i++) {
        global_stddevs[i] = std::sqrt(global_stddevs[i] / n_samples);
    }

    for (auto& v : feature_matrix) {
        for (int i = 0; i < FEATURE_LENGTH; i++) {
            if (global_stddevs[i] > 1e-6) {
                v[i] = (v[i] - global_means[i]) / global_stddevs[i];
            } else {
                v[i] = 0.0;
            }
        }
    }

    is_normalized = true;
    cout << "Data normalized (Z-Score). Mean/StdDev calculated." << endl;
}

void readImagesFromFolder(const string& class_folder) {
    string folder_path = TRAIN_SYMBOLS_PATH + class_folder + "/";
    if (!fs::exists(folder_path)) {
        std::cerr << "ERROR: Folder does not exist: " << folder_path << std::endl;
        return;
    }
    bool first = true;
    int count = 0;
    cout << "Example from class " << class_folder << " ";
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (count >= TRAIN_LIMIT) {
            break;
        }
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
                count ++;
            }
        }
    }
    cout << "(" << count << " images loaded)" << endl;
}

void readTrainingData() {
    feature_matrix.clear();
    Y.clear();
    is_normalized = false;

    for (const string& class_folder: FOLDER_NAMES) {
        printf("Reading from from %s\ncurrent feature matrix size %d\n", class_folder.c_str(), feature_matrix.size());
        readImagesFromFolder(class_folder);
    }

    performZNormalization();
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

void trainPerceptron() {
    if (feature_matrix.empty()) {
        cerr << "Error: No data to train on!" << endl;
        return;
    }
    global_perceptron.train(feature_matrix, Y);
}

void trainNaiveBayes() {
    if (feature_matrix.empty()) {
        cerr << "Error: No data to train Naive Bayes!" << endl;
        return;
    }
    nb_model.train(feature_matrix, Y);
}

string predictPerceptron(const Mat& image) {
    FeatureVector fv = getFeaturesFromImage(image);
    return global_perceptron.predict(fv);
}

string predictNaiveBayes(const Mat& image) {
    FeatureVector fv = getFeaturesFromImage(image);
    return nb_model.predict(fv);
}

string predictWithModel(const Mat& image, ModelType model) {
    switch (model) {
        case MODEL_KNN:
            return knnForImage(image);
        case MODEL_PERCEPTRON:
            return predictPerceptron(image);
        case MODEL_NAIVE_BAYES:
            return predictNaiveBayes(image);
        default:
            return "Unknown";
    }
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

int getClassIndex(const string& label) {
    auto it = find(FOLDER_NAMES.begin(), FOLDER_NAMES.end(), label);
    if (it != FOLDER_NAMES.end()) {
        return distance(FOLDER_NAMES.begin(), it);
    }
    return -1; // should not be the case
}

void processTestFolder(const string& class_folder, ModelType type) {
    string folder_path = TEST_SYMBOLS_PATH + class_folder + "/";
    int actual_index = getClassIndex(class_folder);

    if (actual_index == -1) {
        cerr << "Skip unknown class: " << class_folder << endl;
        return;
    }

    if (!fs::exists(folder_path)) {
        cerr << "Folder missing: " << folder_path << endl;
        return;
    }

    cout << "Processing " << class_folder << " ";
    int count = 0;

    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (count >= TEST_LIMIT) {
            break;
        }

        if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
            string image_path = entry.path().string();
            Mat image = imread(image_path, IMREAD_GRAYSCALE);

            if (image.empty()) continue;

            string predicted_label;

            switch (type) {
                case MODEL_KNN:
                    predicted_label = knnForImage(image);
                    break;
                case MODEL_PERCEPTRON:
                    predicted_label = predictPerceptron(image);
                    break;
                case MODEL_NAIVE_BAYES:
                    predicted_label = predictNaiveBayes(image);
                    break;
            }

            int predicted_index = getClassIndex(predicted_label);

            if (predicted_index != -1) {
                confusion_matrix[actual_index][predicted_index]++;
            }

            if (++count % 10 == 0) {
                cout << "." << flush;
            }
        }
    }
    cout << " Done (" << count << " images)" << endl;
}

void generateConfusionMatrix(ModelType type) {
    if (feature_matrix.empty()) {
        cerr << "CRITICAL ERROR: Training data is empty! KNN will crash." << endl;
        cerr << "Check your TRAIN_SYMBOLS_PATH and ensure readTrainingData() ran successfully." << endl;
        return;
    }

    int n = FOLDER_NAMES.size();
    confusion_matrix.assign(n, vector<int>(n, 0));

    cout << "Starting confusion matrix generation..." << endl;
    cout << "Training Size: " << feature_matrix.size() << " samples." << endl;

    for (const string& class_folder : FOLDER_NAMES) {
        processTestFolder(class_folder, type);
    }
    cout << "Confusion matrix generation complete." << endl;
}

void printConfusionMatrix() {
    int n = confusion_matrix.size();

    cout << "\n--- Confusion Matrix ---\n" << endl;

    cout << setw(12) << "Act\\Pred";
    for (const auto& className : FOLDER_NAMES) {
        cout << setw(8) << className;
    }
    cout << endl;
    cout << string(12 + 8 * n, '-') << endl;

    for (int i = 0; i < n; i++) {
        cout << setw(12) << FOLDER_NAMES[i];
        for (int j = 0; j < n; j++) {
            cout << setw(8) << confusion_matrix[i][j];
        }
        cout << endl;
    }
    cout << endl;

    cout << "\n--- Accuracy Statistics ---\n" << endl;

    int totalCorrect = 0;
    int totalSamples = 0;

    for (int i = 0; i < n; i++) {
        int classCorrect = confusion_matrix[i][i];
        int classTotal = 0;

        for (int j = 0; j < n; j++) {
            classTotal += confusion_matrix[i][j];
        }

        double accuracy = 0.0;
        if (classTotal > 0) {
            accuracy = (double)classCorrect / classTotal * 100.0;
        }

        totalCorrect += classCorrect;
        totalSamples += classTotal;

        cout << "Class '" << setw(6) << left << FOLDER_NAMES[i] << "': "
             << right << setw(6) << classCorrect << "/" << setw(6) << classTotal
             << " = " << fixed << setprecision(2) << setw(6) << accuracy << "%" << endl;
    }

    double totalAccuracy = 0.0;
    if (totalSamples > 0) {
        totalAccuracy = (double)totalCorrect / totalSamples * 100.0;
    }

    cout << "\nTotal Correct: " << totalCorrect << "/" << totalSamples << endl;
    cout << "OVERALL ACCURACY: " << fixed << setprecision(2) << totalAccuracy << "%" << endl;
    cout << "=========================================" << endl;
}

void printLegend() {
    cout << "\n     INSTRUCTIONS      " << endl;
    cout << "=========================================" << endl;
    cout << " [k] - Confusion Matrix for KNN" << endl;
    cout << " [p] - Confusion Matrix for Perceptron" << endl;
    cout << " [n] - Confusion Matrix for Naive Bayes" << endl;
    cout << " [K] - Select model: KNN" << endl;
    cout << " [P] - Select model: Perceptron" << endl;
    cout << " [N] - Select model: Naive Bayes" << endl;
    cout << " [c] - Predict character class" << endl;
    cout << " [e] - Equation mode (build & solve)" << endl;
    cout << " [q] - Quit" << endl;
    cout << "=========================================" << endl;
}

int main () {
    cout << "Loading Data..." << endl;
    readTrainingData();

    trainPerceptron();
    trainNaiveBayes();

    printLegend();

    ModelType currentModel = MODEL_KNN;
    auto modelName = [](ModelType t) -> string {
        switch(t) {
            case MODEL_KNN: return "KNN";
            case MODEL_PERCEPTRON: return "Perceptron";
            case MODEL_NAIVE_BAYES: return "Naive Bayes";
            default: return "Unknown";
        }
    };

    cout << "Current Model: " << modelName(currentModel) << endl;

    char choice;
    while (true) {
        cout << "\n>> Enter command: ";
        cin >> choice;

        switch (choice) {
            case 'k':
                cout << "\nRunning KNN Evaluation..." << endl;
                generateConfusionMatrix(MODEL_KNN);
                printConfusionMatrix();
                break;
            case 'p':
                cout << "\nRunning Perceptron Evaluation..." << endl;
                generateConfusionMatrix(MODEL_PERCEPTRON);
                printConfusionMatrix();
                break;
            case 'n':
                cout << "\nRunning Naive Bayes Evaluation..." << endl;
                generateConfusionMatrix(MODEL_NAIVE_BAYES);
                printConfusionMatrix();
                break;
            case 'K':
                currentModel = MODEL_KNN;
                cout << "Active Model changed to: " << modelName(currentModel) << endl;
                break;
            case 'P':
                currentModel = MODEL_PERCEPTRON;
                cout << "Active Model changed to: " << modelName(currentModel) << endl;
                break;
            case 'N':
                currentModel = MODEL_NAIVE_BAYES;
                cout << "Active Model changed to: " << modelName(currentModel) << endl;
                break;
            case 'c':
                cout << "Using model: " << modelName(currentModel) << endl;
                interactivePrediction(currentModel);
                break;
            case 'e':
                cout << "Using model: " << modelName(currentModel) << endl;
                equationMode(currentModel);
                break;
            case 'q':
                cout << "Exiting..." << endl;
                return 0;
            default:
                cout << "Invalid command." << endl;
        }
    }
}

