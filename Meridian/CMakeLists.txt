cmake_minimum_required(VERSION 3.30)
project(Meridian)

set(CMAKE_VERBOSE_MAKEFILE ON)


set(CMAKE_CXX_STANDARD 17)

# Add GLFW as a subdirectory (assumes it has a CMakeLists.txt)
add_subdirectory(vendor/glfw-3.4)

# Add include directories
include_directories(
        ${CMAKE_SOURCE_DIR}/vendor/glfw-3.4/include
        ${CMAKE_SOURCE_DIR}/vendor/glm
        ${CMAKE_SOURCE_DIR}/vendor
)

# Platform-specific Vulkan setup
if (WIN32)
    # Windows
    include_directories($ENV{VULKAN_SDK}/Include)
    link_directories($ENV{VULKAN_SDK}/Lib)
    set(VULKAN_LIB vulkan-1)
elseif (UNIX)
    # Linux
    find_package(Vulkan REQUIRED)
    include_directories(${Vulkan_INCLUDE_DIRS})
    set(VULKAN_LIB ${Vulkan_LIBRARIES})
endif()

# Add the executable
add_executable(Meridian main.cpp)

# Link libraries: GLFW and Vulkan
target_link_libraries(Meridian
        glfw
        ${VULKAN_LIB}
)
