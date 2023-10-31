cmake_minimum_required(VERSION 3.15)
project(franka_emika_panda)
include(CTest)

set(CMAKE_CXX_STANDARD 17)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-no-as-needed")


find_package (Eigen3 3.3.7 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(pybind11 REQUIRED) # for pybind11
find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    tf
)
# find_package(Torch REQUIRED PATHS "/home/kist-robot2/libtorch")

message(STATUS "Eigen3 : " ${EIGEN3_INCLUDE_DIR}/Eigen)
# message(STATUS "Torch  : " ${TORCH_INCLUDE_DIR}/include/torch)
message(STATUS "current : " ${CMAKE_CURRENT_SOURCE_DIR})





add_definitions(-DMJSIMULATE_STATIC) # use static =,solve the "error LNK2001/LNK2019" in windows

find_package (X11)
set(MUJOCO_HOME {CMAKE_SOURCE_DIR})
message(STATUS "MUJOCO_HOME: " ${MUJOCO_HOME})

catkin_package(
  # INCLUDE_DIRS include
#   LIBRARIES franka_panda
  # CATKIN_DEPENDS roscpp std_msgs 
#  DEPENDS system_lib
)

#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


link_directories(
    # ${CMAKE_CURRENT_SOURCE_DIR}/lib/
    /usr/include/python3.8 # for pybind11
    
)


include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${GLEW_INCLUDE_DIRS}
    /usr/include/python3.8
)

set(COM_SRC
        simulate/glfw_dispatch.cc
        simulate/simulate.cc
        simulate/uitools.cc
        simulate/main.cc
        
        simulate/controller.cpp
        simulate/robotmodel.cpp
        simulate/trajectory.cpp
        
        )
# # add_library(franka_emika_lib
# #     ${COM_SRC}
# #     ${catkin_LIBRARIES}
# # )

# # # for pybind11
# add_library(franka_emika_lib_py
#     ${COM_SRC}
# )



# set(COM_LIB 
#       libmujoco.so.2.3.1
#       glfw3 
#       -lpthread)

add_executable(${PROJECT_NAME} ${COM_SRC})
# add_executable(${PROJECT_NAME} simulate/example-app.cpp)


target_link_libraries(${PROJECT_NAME} 
	# ${COM_LIB} 
	${CMAKE_DL_LIBS} 
	${X11_LIBRARIES} 
  libmujoco.so.2.3.1
  glfw
  -lpython3.8
	-lrbdl
    	-lrbdl_urdfreader
    	-lpthread
    	OpenGL::OpenGL
	OpenGL::EGL
	GLEW::GLEW
	${GLEW_LIBRARIES}
	${OPENGL_LIBRARIES}
	# ${TORCH_LIBRARIES}
)



target_link_directories(${PROJECT_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/mujoco-2.3.1/lib
)


# add_library(franka_emika_lib
#     ${COM_SRC}
#     ${catkin_LIBRARIES}
# )

# # for pybind11
add_library(franka_emika_lib_py
    ${COM_SRC}
)

# add_subdirectory(pybind11)
pybind11_add_module(controller simulate/controller.cpp)
target_link_libraries(controller
PUBLIC
    # ${TORCH_LIBRARIES}
    ${catkin_LIBRARIES}
    Eigen3::Eigen
    franka_emika_lib_py
    -lrbdl
      -lrbdl_urdfreader
      OpenGL::OpenGL
    OpenGL::EGL
    -lpython3.8
    libmujoco.so.2.3.1
    glfw
    -lpthread
    ${X11_LIBRARIES} 
    GLEW::GLEW
    pybind11::pybind11
    ${OMPL_LIBRARIES}
    ${GLEW_LIBRARIES}
	  ${OPENGL_LIBRARIES}
    
)


# target_include_directories(controller PRIVATE
#   # /home/kist-robot2/libtorch/include
#   /Home/libtorch/include  
# )

# include_directories(controller
#     usr/include/libtorch/include
# )


# ################# For ROS ###########################
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#    FILES_MATCHING PATTERN "*.h"
#    PATTERN ".svn" EXCLUDE
#  )




# cmake_minimum_required(VERSION 3.15)
# project(franka_emika_panda)
# include(CTest)

# set(CMAKE_CXX_STANDARD 17)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-no-as-needed")


# find_package (Eigen3 3.3.7 REQUIRED)
# find_package(OpenGL REQUIRED)
# find_package(GLEW REQUIRED)
# find_package(pybind11 REQUIRED) # for pybind11
# find_package(catkin REQUIRED COMPONENTS
#     roscpp
#     std_msgs
#     tf
# )
# find_package(Torch REQUIRED PATHS "/home/kist-robot2/libtorch")

# message(STATUS "Eigen3 : " ${EIGEN3_INCLUDE_DIR}/Eigen)
# message(STATUS "Torch  : " ${TORCH_LIBRARIES}/include/torch)
# message(STATUS "current : " ${CMAKE_CURRENT_SOURCE_DIR})





# add_definitions(-DMJSIMULATE_STATIC) # use static =,solve the "error LNK2001/LNK2019" in windows

# find_package (X11)
# # set MUJOCO root
# set(MUJOCO_HOME {CMAKE_SOURCE_DIR})
# message(STATUS "MUJOCO_HOME: " ${MUJOCO_HOME})

# catkin_package(
#   INCLUDE_DIRS include
# #   LIBRARIES franka_panda
#   CATKIN_DEPENDS roscpp std_msgs 
# #  DEPENDS system_lib
# )

# #include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


# #link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/)
# link_directories(
#     ${CMAKE_CURRENT_SOURCE_DIR}/lib/
#     /usr/include/python3.8 # for pybind11
# )


# include_directories(
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
#   $ENV{EIGEN3_INCLUDE_DIR}/Eigen/
#   ${GLEW_INCLUDE_DIRS}
#     /usr/local/include/rbdl
#     /usr/include/python3.8
#     /home/kist-robot2/libtorch
# )

# set(COM_SRC
#         simulate/glfw_dispatch.cc
#         simulate/simulate.cc
#         simulate/uitools.cc
#         simulate/main.cc
        
#         simulate/controller.cpp
#         simulate/robotmodel.cpp
#         simulate/trajectory.cpp
        
#         )
# add_library(franka_emika_lib
#     ${COM_SRC}
#     ${catkin_LIBRARIES}
# )

# # for pybind11
# add_library(franka_emika_lib_py
#     ${COM_SRC}
# )



# set(COM_LIB 
#         libmujoco.so.2.3.1
#         glfw
#         -lpthread)

# add_executable(${PROJECT_NAME} ${COM_SRC})



# target_link_libraries(${PROJECT_NAME} 
# 	${COM_LIB} 
# 	${CMAKE_DL_LIBS} 
# 	${X11_LIBRARIES} 
# 	-lpython3.8
# 	-lrbdl
#     	-lrbdl_urdfreader
#     	-lpthread
#     	OpenGL::OpenGL
# 	OpenGL::EGL
# 	GLEW::GLEW
# 	${GLEW_LIBRARIES}
# 	${OPENGL_LIBRARIES}
# 	${TORCH_LIBRARIES}
# )




# # add_subdirectory(pybind11)
# pybind11_add_module(controller simulate/controller.cpp)
# target_link_libraries(controller
# PUBLIC
#     ${catkin_LIBRARIES}
#     Eigen3::Eigen
#     franka_emika_lib_py
#     -lrbdl
#       -lrbdl_urdfreader
#       -lpthread
#       OpenGL::OpenGL
#     OpenGL::EGL
#     -lpython3.8
#     ${COM_LIB} 
#     ${X11_LIBRARIES} 
#     GLEW::GLEW
#     pybind11::pybind11
#     ${OMPL_LIBRARIES}
#     ${GLEW_LIBRARIES}
# 	  ${OPENGL_LIBRARIES}
# )


# ################# For ROS ###########################
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#    FILES_MATCHING PATTERN "*.h"
#    PATTERN ".svn" EXCLUDE
#  )

