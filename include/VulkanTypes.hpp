//
// Created by Profanateur on 12/4/2022.
//

#ifndef VULKAN_TERRAIN_VULKAN_TYPES_HPP
#define VULKAN_TERRAIN_VULKAN_TYPES_HPP

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

struct AllocatedBuffer {
    VkBuffer m_buffer;
    VmaAllocation m_allocation;
};

struct AllocatedImage {
    VkImage m_image;
    VmaAllocation m_allocation;
};

#endif //VULKAN_TERRAIN_VULKAN_TYPES_HPP
