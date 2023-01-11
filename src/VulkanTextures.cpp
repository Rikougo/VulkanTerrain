//
// Created by Profanateur on 12/27/2022.
//

#include "VulkanTextures.hpp"
#include <iostream>

#include <VulkanEngine.hpp>
#include <VulkanInitializer.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

bool VulkanUtil::LoadImageFromFile(VulkanEngine &p_engine, std::filesystem::path p_path, AllocatedImage &p_outImage, VkFormat p_imageFormat) {
    int l_texWidth, l_texHeight, l_texChannels;

	unsigned char* l_pixels = stbi_load(p_path.string().c_str(), &l_texWidth, &l_texHeight, &l_texChannels, STBI_rgb_alpha);

	if (!l_pixels) {
		std::cout << "Failed to load texture file " << p_path << std::endl;
		return false;
	}

    bool l_result = VulkanUtil::LoadImageFromData(p_engine, l_pixels, l_texWidth, l_texHeight, p_outImage, p_imageFormat);

    stbi_image_free(l_pixels);

    return l_result;
}

bool VulkanUtil::LoadImageFromData(VulkanEngine &p_engine, unsigned char* p_imageData, int p_texWidth, int p_texHeight, AllocatedImage &p_outImage, VkFormat p_imageFormat) {
    void* l_pxlPtr = p_imageData;
	VkDeviceSize l_imageSize = p_texWidth * p_texHeight * 4;

	//the format R8G8B8A8 matches exactly with the pixels loaded from stb_image lib
	VkFormat l_imageFormat = p_imageFormat;

	//allocate temporary buffer for holding texture data to upload
	AllocatedBuffer stagingBuffer = p_engine.CreateBuffer(l_imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

	//copy data to buffer
	void* data;
	vmaMapMemory(p_engine.m_allocator, stagingBuffer.m_allocation, &data);

	memcpy(data, l_pxlPtr, static_cast<size_t>(l_imageSize));

	vmaUnmapMemory(p_engine.m_allocator, stagingBuffer.m_allocation);

    VkExtent3D imageExtent;
	imageExtent.width = static_cast<uint32_t>(p_texWidth);
	imageExtent.height = static_cast<uint32_t>(p_texHeight);
	imageExtent.depth = 1;

	VkImageCreateInfo dimg_info = VulkanInit::ImageCreateInfo(l_imageFormat, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, imageExtent);

	AllocatedImage l_newImage;

	VmaAllocationCreateInfo dimageAllocInfo = {};
	dimageAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	//allocate and create the image
	vmaCreateImage(p_engine.m_allocator, &dimg_info,
                   &dimageAllocInfo, &l_newImage.m_image, &l_newImage.m_allocation,
                   nullptr);

    p_engine.ImmediateSubmit([&](VkCommandBuffer cmd) {
		VkImageSubresourceRange l_range;
		l_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		l_range.baseMipLevel = 0;
		l_range.levelCount = 1;
		l_range.baseArrayLayer = 0;
		l_range.layerCount = 1;

		VkImageMemoryBarrier imageBarrier_toTransfer = {};
		imageBarrier_toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;

		imageBarrier_toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageBarrier_toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		imageBarrier_toTransfer.image = l_newImage.m_image;
		imageBarrier_toTransfer.subresourceRange = l_range;

		imageBarrier_toTransfer.srcAccessMask = 0;
		imageBarrier_toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		//barrier the image into the transfer-receive layout
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1,
                             &imageBarrier_toTransfer);

        VkBufferImageCopy copyRegion = {};
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;

        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageExtent = imageExtent;

        //copy the buffer into the image
        vkCmdCopyBufferToImage(cmd, stagingBuffer.m_buffer, l_newImage.m_image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

        VkImageMemoryBarrier imageBarrier_toReadable = imageBarrier_toTransfer;

        imageBarrier_toReadable.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        imageBarrier_toReadable.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        imageBarrier_toReadable.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        imageBarrier_toReadable.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        //barrier the image into the shader readable layout
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1,
                             &imageBarrier_toReadable);
    });


    /*p_engine.m_mainDeletionQueue.PushFunction([=]() {
		vmaDestroyImage(p_engine.m_allocator, l_newImage.m_image, l_newImage.m_allocation);
	});*/

	vmaDestroyBuffer(p_engine.m_allocator, stagingBuffer.m_buffer, stagingBuffer.m_allocation);

	p_outImage = l_newImage;
	return true;
}