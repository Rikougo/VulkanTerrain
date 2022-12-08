cmake_minimum_required(VERSION 3.1X)
project(VulkanTerrain LANGUAGES CXX C)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(VMA_STATIC_VULKAN_FUNCTIONS OFF CACHE BOOL "" FORCE)

set(Vulkan_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/include" CACHE STRING "" FORCE)
set(VMA_STATIC_VULKAN_FUNCTIONS ON CACHE BOOL "" FORCE)
set(VMA_DYNAMIC_VULKAN_FUNCTIONS OFF CACHE BOOL "" FORCE)
include_directories(${Vulkan_INCLUDE_DIR})
set(VK_BOOTSTRAP_VULKAN_HEADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/include" CACHE STRING "" FORCE)
find_package(Vulkan REQUIRED)

add_subdirectory(libs/)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCES} main.cpp src/VulkanInitializer.cpp include/VulkanInitializer.hpp src/VulkanEngine.cpp include/VulkanEngine.hpp include/VulkanTypes.hpp src/VulkanMesh.cpp include/VulkanMesh.hpp)

if (MSVC)
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE /Zc:__cplusplus)
endif ()

target_compile_features(${CMAKE_PROJECT_NAME} PUBLIC cxx_std_23)
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC Vulkan::Vulkan glfw IMGUI VulkanMemoryAllocator vk-bootstrap::vk-bootstrap tinyobjloader)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE include/ libs/glfw/include/ libs/glm/ libs/include/ libs/VMA/include libs/tinyobj)