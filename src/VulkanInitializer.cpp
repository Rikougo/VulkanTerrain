//
// Created by Profanateur on 12/4/2022.
//

#include "VulkanInitializer.hpp"


VkCommandPoolCreateInfo VulkanInit::CommandPoolCreateInfo(uint32_t p_queueFamilyIndex, VkCommandPoolCreateFlags p_flags /*= 0*/) {
	VkCommandPoolCreateInfo l_info = {};
    l_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    l_info.pNext = nullptr;

    l_info.queueFamilyIndex = p_queueFamilyIndex;
    l_info.flags = p_flags;
	return l_info;
}

VkCommandBufferAllocateInfo VulkanInit::CommandBufferAllocateInfo(VkCommandPool p_pool, uint32_t p_count /*= 1*/, VkCommandBufferLevel p_level /*= VK_COMMAND_BUFFER_LEVEL_PRIMARY*/) {
	VkCommandBufferAllocateInfo l_info = {};
    l_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    l_info.pNext = nullptr;

    l_info.commandPool = p_pool;
    l_info.commandBufferCount = p_count;
    l_info.level = p_level;
	return l_info;
}

VkPipelineShaderStageCreateInfo VulkanInit::PipelineShaderStageCreateInfo(VkShaderStageFlagBits p_stage, VkShaderModule p_shaderModule) {
    VkPipelineShaderStageCreateInfo l_info{};
    l_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    l_info.pNext = nullptr;

    //shader stage
    l_info.stage = p_stage;
    //module containing the code for this shader stage
    l_info.module = p_shaderModule;
    //the entry point of the shader
    l_info.pName = "main";
    return l_info;
}

VkPipelineVertexInputStateCreateInfo VulkanInit::VertexInputStateCreateInfo() {
    VkPipelineVertexInputStateCreateInfo l_info = {};
    l_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    l_info.pNext = nullptr;

    //no vertex bindings or attributes
    l_info.vertexBindingDescriptionCount = 0;
    l_info.vertexAttributeDescriptionCount = 0;
    return l_info;
}

VkPipelineInputAssemblyStateCreateInfo VulkanInit::InputAssemblyStateCreateInfo(VkPrimitiveTopology p_topology) {
    VkPipelineInputAssemblyStateCreateInfo l_info = {};
    l_info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    l_info.pNext = nullptr;

    l_info.topology = p_topology;
    //we are not going to use primitive restart on the entire tutorial so leave it on false
    l_info.primitiveRestartEnable = VK_FALSE;
    return l_info;
}

VkPipelineRasterizationStateCreateInfo VulkanInit::RasterizationStateCreateInfo(VkPolygonMode p_polygonMode){
    VkPipelineRasterizationStateCreateInfo l_info = {};
    l_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    l_info.pNext = nullptr;

    l_info.depthClampEnable = VK_FALSE;
    //discards all primitives before the rasterization stage if enabled which we don't want
    l_info.rasterizerDiscardEnable = VK_FALSE;

    l_info.polygonMode = p_polygonMode;
    l_info.lineWidth = 1.0f;
    //no backface cull
    l_info.cullMode = VK_CULL_MODE_NONE;
    l_info.frontFace = VK_FRONT_FACE_CLOCKWISE;
    //no depth bias
    l_info.depthBiasEnable = VK_FALSE;
    l_info.depthBiasConstantFactor = 0.0f;
    l_info.depthBiasClamp = 0.0f;
    l_info.depthBiasSlopeFactor = 0.0f;

    return l_info;
}

VkPipelineMultisampleStateCreateInfo VulkanInit::MultisampleStateCreateInfo(){
    VkPipelineMultisampleStateCreateInfo l_info = {};
    l_info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    l_info.pNext = nullptr;

    l_info.sampleShadingEnable = VK_FALSE;
    //multisampling defaulted to no multisampling (1 sample per pixel)
    l_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    l_info.minSampleShading = 1.0f;
    l_info.pSampleMask = nullptr;
    l_info.alphaToCoverageEnable = VK_FALSE;
    l_info.alphaToOneEnable = VK_FALSE;
    return l_info;
}

VkPipelineColorBlendAttachmentState VulkanInit::ColorBlendAttachmentState() {
    VkPipelineColorBlendAttachmentState l_colorBlendAttachment = {};
    l_colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    l_colorBlendAttachment.blendEnable = VK_FALSE;
    return l_colorBlendAttachment;
}

VkPipelineTessellationStateCreateInfo VulkanInit::TessellationStateCreateInfo(uint32_t p_patchControlPoints) {
    VkPipelineTessellationStateCreateInfo l_info = {};
    l_info.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
    l_info.pNext = nullptr;

    l_info.patchControlPoints = p_patchControlPoints;
    return l_info;
}

VkPipelineLayoutCreateInfo VulkanInit::PipelineLayoutCreateInfo() {
    VkPipelineLayoutCreateInfo l_info{};
    l_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    l_info.pNext = nullptr;

    //empty defaults
    l_info.flags = 0;
    l_info.setLayoutCount = 0;
    l_info.pSetLayouts = nullptr;
    l_info.pushConstantRangeCount = 0;
    l_info.pPushConstantRanges = nullptr;
    return l_info;
}


VkImageCreateInfo VulkanInit::ImageCreateInfo(VkFormat p_format, VkImageUsageFlags p_usageFlags, VkExtent3D p_extent) {
    VkImageCreateInfo l_info = { };
    l_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    l_info.pNext = nullptr;

    l_info.imageType = VK_IMAGE_TYPE_2D;

    l_info.format = p_format;
    l_info.extent = p_extent;

    l_info.mipLevels = 1;
    l_info.arrayLayers = 1;
    l_info.samples = VK_SAMPLE_COUNT_1_BIT;
    l_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    l_info.usage = p_usageFlags;

    return l_info;
}

VkImageViewCreateInfo VulkanInit::ImageViewCreateInfo(VkFormat p_format, VkImage p_image, VkImageAspectFlags p_aspectFlags) {
    //build a image-view for the depth image to use for rendering
	VkImageViewCreateInfo l_info = {};
    l_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    l_info.pNext = nullptr;

    l_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    l_info.image = p_image;
    l_info.format = p_format;
    l_info.subresourceRange.baseMipLevel = 0;
    l_info.subresourceRange.levelCount = 1;
    l_info.subresourceRange.baseArrayLayer = 0;
    l_info.subresourceRange.layerCount = 1;
    l_info.subresourceRange.aspectMask = p_aspectFlags;

	return l_info;
}

VkPipelineDepthStencilStateCreateInfo VulkanInit::DepthStencilCreateInfo(bool p_bDepthTest, bool p_bDepthWrite, VkCompareOp p_compareOp)
{
    VkPipelineDepthStencilStateCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    info.pNext = nullptr;

    info.depthTestEnable = p_bDepthTest ? VK_TRUE : VK_FALSE;
    info.depthWriteEnable = p_bDepthWrite ? VK_TRUE : VK_FALSE;
    info.depthCompareOp = p_bDepthTest ? p_compareOp : VK_COMPARE_OP_ALWAYS;
    info.depthBoundsTestEnable = VK_FALSE;
    info.minDepthBounds = 0.0f; // Optional
    info.maxDepthBounds = 1.0f; // Optional
    info.stencilTestEnable = VK_FALSE;

    return info;
}

VkDescriptorSetLayoutBinding VulkanInit::DescriptorSetLayoutBinding(VkDescriptorType p_type, VkShaderStageFlags p_stageFlags, uint32_t p_binding)
{
	VkDescriptorSetLayoutBinding l_setbind = {};
    l_setbind.binding = p_binding;
    l_setbind.descriptorCount = 1;
    l_setbind.descriptorType = p_type;
    l_setbind.pImmutableSamplers = nullptr;
    l_setbind.stageFlags = p_stageFlags;

	return l_setbind;
}

VkWriteDescriptorSet VulkanInit::WriteDescriptorSet(VkDescriptorType p_type, VkDescriptorSet p_dstSet, VkDescriptorBufferInfo* p_bufferInfo , uint32_t p_binding)
{
	VkWriteDescriptorSet l_write = {};
    l_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    l_write.pNext = nullptr;

    l_write.dstBinding = p_binding;
    l_write.dstSet = p_dstSet;
    l_write.descriptorCount = 1;
    l_write.descriptorType = p_type;
    l_write.pBufferInfo = p_bufferInfo;

	return l_write;
}

VkCommandBufferBeginInfo VulkanInit::CommandBufferBeginInfo(VkCommandBufferUsageFlags p_flags /*= 0*/) {
    VkCommandBufferBeginInfo l_info = {};
    l_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    l_info.pNext = nullptr;

    l_info.pInheritanceInfo = nullptr;
    l_info.flags = p_flags;
	return l_info;
};

VkSubmitInfo VulkanInit::SubmitInfo(VkCommandBuffer *p_cmd) {
    VkSubmitInfo l_info = {};
    l_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    l_info.pNext = nullptr;

    l_info.waitSemaphoreCount = 0;
    l_info.pWaitSemaphores = nullptr;
    l_info.pWaitDstStageMask = nullptr;
    l_info.commandBufferCount = 1;
    l_info.pCommandBuffers = p_cmd;
    l_info.signalSemaphoreCount = 0;
    l_info.pSignalSemaphores = nullptr;

	return l_info;
};

VkSamplerCreateInfo VulkanInit::SamplerCreateInfo(VkFilter p_filters, VkSamplerAddressMode p_samplerAddressMode /*= VK_SAMPLER_ADDRESS_MODE_REPEAT*/){
    VkSamplerCreateInfo l_info = {};
    l_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    l_info.pNext = nullptr;

    l_info.magFilter = p_filters;
    l_info.minFilter = p_filters;
    l_info.addressModeU = p_samplerAddressMode;
    l_info.addressModeV = p_samplerAddressMode;
    l_info.addressModeW = p_samplerAddressMode;

	return l_info;
};

VkWriteDescriptorSet VulkanInit::WriteDescriptorImage(VkDescriptorType p_type, VkDescriptorSet p_dstSet, VkDescriptorImageInfo* p_imageInfo, uint32_t p_binding){
    VkWriteDescriptorSet l_write = {};
    l_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    l_write.pNext = nullptr;

    l_write.dstBinding = p_binding;
    l_write.dstSet = p_dstSet;
    l_write.descriptorCount = 1;
    l_write.descriptorType = p_type;
    l_write.pImageInfo = p_imageInfo;

    return l_write;
};