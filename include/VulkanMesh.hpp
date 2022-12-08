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
#include <glm/vec3.hpp>

struct VertexInputDescription {

	std::vector<VkVertexInputBindingDescription> bindings;
	std::vector<VkVertexInputAttributeDescription> attributes;

	VkPipelineVertexInputStateCreateFlags flags = 0;
};

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 color;

    static VertexInputDescription GetVertexDescription();
};

struct Mesh {
	std::vector<Vertex> m_vertices;

	AllocatedBuffer m_vertexBuffer;

    bool LoadFromObj(const std::filesystem::path &p_path);
};

#endif //VULKAN_TERRAIN_VULKAN_MESH_HPP
