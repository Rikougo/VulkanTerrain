//
// Created by Profanateur on 12/27/2022.
//

#ifndef VULKAN_TERRAIN_VULKAN_TEXTURES_HPP
#define VULKAN_TERRAIN_VULKAN_TEXTURES_HPP

#include <filesystem>

#include <VulkanTypes.hpp>

class VulkanEngine;

namespace VulkanUtil {
    bool LoadImageFromFile(VulkanEngine &p_engine, std::filesystem::path p_path, AllocatedImage& p_outImage);
}

#endif //VULKAN_TERRAIN_VULKAN_TEXTURES_HPP
