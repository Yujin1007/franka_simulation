#include <fstream> // ifstream header
#include <iostream>
#include <string> // getline header
#include <math.h>
#include <eigen3/Eigen/Dense>
#include <vector>
using namespace Eigen;
using namespace std;


int main() {
    Eigen::MatrixXd eigenMatrix(6, 7); // Example Eigen matrix of size 6x7
    eigenMatrix << 1, 2, 3, 4, 5, 6, 7,
                   8, 9, 10, 11, 12, 13, 14,
                   15, 16, 17, 18, 19, 20, 21,
                   22, 23, 24, 25, 26, 27, 28,
                   29, 30, 31, 32, 33, 34, 35,
                   36, 37, 38, 39, 40, 41, 42;

    std::vector<std::vector<double>> vectorMatrix(6,7);
       // std::vector<std::vector<double>> vectorMatrix(eigenMatrix.rows(), std::vector<double>(eigenMatrix.cols()));

    for (int i = 0; i < eigenMatrix.rows(); ++i) {
        for (int j = 0; j < eigenMatrix.cols(); ++j) {
            vectorMatrix[i][j] = eigenMatrix(i, j);
        }
    }

    // Access the values in the vector of vectors
    for (int i = 0; i < vectorMatrix.size(); ++i) {
        for (int j = 0; j < vectorMatrix[i].size(); ++j) {
            double value = vectorMatrix[i][j];
            cout<<value;
        }
    }

    return 0;
}

