//
// Created by Profanateur on 12/4/2022.
//

#ifndef VULKAN_TERRAIN_VULKAN_INITIALIZER_HPP
#define VULKAN_TERRAIN_VULKAN_INITIALIZER_HPP

#include <VulkanTypes.hpp>

namespace VulkanInit {
    VkCommandPoolCreateInfo CommandPoolCreateInfo(uint32_t p_queueFamilyIndex, VkCommandPoolCreateFlags p_flags = 0);
	VkCommandBufferAllocateInfo CommandBufferAllocateInfo(VkCommandPool p_pool, uint32_t p_count = 1, VkCommandBufferLevel p_level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

    VkPipelineShaderStageCreateInfo PipelineShaderStageCreateInfo(VkShaderStageFlagBits p_stage, VkShaderModule p_shaderModule);
    VkPipelineVertexInputStateCreateInfo VertexInputStateCreateInfo();
    VkPipelineInputAssemblyStateCreateInfo InputAssemblyStateCreateInfo(VkPrimitiveTopology p_topology);
    VkPipelineRasterizationStateCreateInfo RasterizationStateCreateInfo(VkPolygonMode p_polygonMode);
    VkPipelineMultisampleStateCreateInfo MultisampleStateCreateInfo();
    VkPipelineColorBlendAttachmentState ColorBlendAttachmentState();

    VkPipelineLayoutCreateInfo PipelineLayoutCreateInfo();

    VkImageCreateInfo ImageCreateInfo(VkFormat p_format, VkImageUsageFlags p_usageFlags, VkExtent3D p_extent);
    VkImageViewCreateInfo ImageViewCreateInfo(VkFormat p_format, VkImage p_image, VkImageAspectFlags p_aspectFlags);

    VkPipelineDepthStencilStateCreateInfo DepthStencilCreateInfo(bool p_bDepthTest, bool p_bDepthWrite, VkCompareOp p_compareOp);


    VkDescriptorSetLayoutBinding DescriptorSetLayoutBinding(VkDescriptorType p_type, VkShaderStageFlags p_stageFlags, uint32_t p_binding);
    VkWriteDescriptorSet WriteDescriptorSet(VkDescriptorType p_type, VkDescriptorSet p_dstSet, VkDescriptorBufferInfo* p_bufferInfo , uint32_t p_binding);

    VkCommandBufferBeginInfo CommandBufferBeginInfo(VkCommandBufferUsageFlags p_flags = 0);
    VkSubmitInfo SubmitInfo(VkCommandBuffer *p_cmd);
}

#endif //VULKAN_TERRAIN_VULKAN_INITIALIZER_HPP
