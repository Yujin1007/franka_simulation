cmake_minimum_required(VERSION 3.15)
project(franka_emika_panda)
include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-no-as-needed")

find_package (Eigen3 3.3.7 REQUIRED)
find_package(pybind11 REQUIRED) # for pybind11
include_directories(${pybind11_INCLUDE_DIRS})

message(STATUS "Eigen3 : " ${EIGEN3_INCLUDE_DIR}/Eigen)
message(STATUS "current : " ${CMAKE_CURRENT_SOURCE_DIR})

add_definitions(-DMJSIMULATE_STATIC) # use static =,solve the "error LNK2001/LNK2019" in windows
find_package (X11)

link_directories(
    /usr/include/python3.8 # for pybind11
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${GLEW_INCLUDE_DIRS}
    /usr/include/python3.8
)

set(COM_SRC
        simulate/controller.cpp
        simulate/robotmodel.cpp
        simulate/trajectory.cpp
        simulate/main.cpp
    )

add_executable(${PROJECT_NAME} ${COM_SRC})
add_executable(build_test simulate/main.cpp)

target_link_libraries(${PROJECT_NAME} 
	${CMAKE_DL_LIBS} 
	${X11_LIBRARIES} 
  -lpython3.8
	-lrbdl
  -lrbdl_urdfreader
  -lpthread
)

# for pybind11
add_library(franka_emika_lib_py
  simulate/controller.cpp
  simulate/robotmodel.cpp
  simulate/trajectory.cpp
)
target_compile_options(franka_emika_lib_py PRIVATE -fPIC)

# add_subdirectory(pybind11)
pybind11_add_module(controller simulate/controller.cpp)
target_link_libraries(controller
PUBLIC
    Eigen3::Eigen
    franka_emika_lib_py
    -lrbdl
    -lrbdl_urdfreader
    -lpython3.8
    -lpthread
    ${X11_LIBRARIES} 
    pybind11::pybind11
    ${OMPL_LIBRARIES}
)
