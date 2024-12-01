cmake_minimum_required(VERSION 3.10)
project(RMvision)

set(CMAKE_CXX_STANDARD 11)

# 设置OpenCV路径
set(OpenCV_DIR "/home/chengzi/Desktop/opencv/opencv-4.9.0/build")

# 查找OpenCV包
find_package(OpenCV REQUIRED)

# 包含OpenCV头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 输出OpenCV库状态
message(STATUS "OpenCV library status: ${OpenCV_FOUND}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# 查找源文件和头文件
file(GLOB_RECURSE SRC_FILES "src/*.cpp")
file(GLOB_RECURSE HEAD_FILES "include/*.h")
include_directories("include")

# 添加可执行文件
add_executable(RMvision ${SRC_FILES} )

# 链接OpenCV库
target_link_libraries(RMvision ${OpenCV_LIBS})