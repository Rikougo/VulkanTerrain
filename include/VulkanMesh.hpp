//
// Created by Profanateur on 12/6/2022.
//

#ifndef VULKAN_TERRAIN_VULKAN_MESH_HPP
#define VULKAN_TERRAIN_VULKAN_MESH_HPP

#include <vector>
#include <filesystem>
#include <iostream>

#include <VulkanTypes.hpp>

#include <tiny_obj_loader.h>
#include <glm/glm.hpp>

struct VertexInputDescription {

	std::vector<VkVertexInputBindingDescription> bindings;
	std::vector<VkVertexInputAttributeDescription> attributes;

	VkPipelineVertexInputStateCreateFlags flags = 0;
};

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 color;
    glm::vec2 uv;

    static VertexInputDescription GetVertexDescription();
};

struct Mesh {
	std::vector<Vertex> m_vertices;
    std::vector<uint32_t> m_indices;

	AllocatedBuffer m_vertexBuffer;

    bool LoadFromObj(const std::filesystem::path &p_path);
};

namespace VulkanUtil {
    Mesh CreateQuad(float p_size = 1, uint8_t p_resolution = 1);
}

#endif //VULKAN_TERRAIN_VULKAN_MESH_HPP
