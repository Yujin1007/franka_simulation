#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <rbdl/rbdl.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <mujoco/mujoco.h>

using namespace std;
using namespace Eigen;

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
  Vector3d test;
  test << 0,0,0;
  std::cout<<"test :"<<test.transpose()<<std::endl;
  printf("mujoco cpp \n");
}

