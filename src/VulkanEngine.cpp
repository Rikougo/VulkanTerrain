//
// Created by Profanateur on 12/4/2022.
//

#include "VulkanEngine.hpp"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
#include "imgui_impl_glfw.h"

#define VK_CHECK(x)                                                 \
	do                                                              \
	{                                                               \
		VkResult err = x;                                           \
		if (err)                                                    \
		{                                                           \
			std::cout <<"Detected Vulkan error: " << err << std::endl; \
			abort();                                                \
		}                                                           \
	} while (0)



void VulkanEngine::Init() {
    m_showDemoWindow = new bool(false);
    m_showInspectorWindow = new bool(true);
    m_showHeightMapWindow = new bool(false);

    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    m_window = glfwCreateWindow(
            static_cast<int>(m_windowExtent.width),
            static_cast<int>(m_windowExtent.height),
            "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(m_window, this);

    glfwSetFramebufferSizeCallback(m_window, FramebufferSizeCallback);

    glfwSetKeyCallback(m_window, [](GLFWwindow* p_window, int p_key, int p_scancode, int p_action, int p_mods) {
        auto* engine = reinterpret_cast<VulkanEngine*>(glfwGetWindowUserPointer(p_window));
        engine->OnKeyPressed(p_key, p_scancode, p_action, p_mods);
    });

    InitVulkan();
    InitSwapchain();
    InitCommands();
    InitDefaultRenderpass();
    InitFramebuffers();
    InitSyncStructures();
    InitDescriptors();
    InitPipelines();
    InitIMGUI();

    LoadImages();
    LoadMeshes();

    InitScene();

    m_initialized = true;
}

void VulkanEngine::Run() {
    auto l_now = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(m_window)) {
        auto l_then = std::chrono::high_resolution_clock::now();
        float l_deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(l_then - l_now).count();
        l_now = l_then;

        glfwPollEvents();

        ProcessInputs(l_deltaTime);

        DrawUI();
        Draw();
    }

    vkDeviceWaitIdle(m_device);
}

void VulkanEngine::Cleanup() {
    if (m_initialized) {
        std::vector<VkFence> l_fences(FRAME_AMOUNT);
        for(uint8_t i = 0; i < FRAME_AMOUNT; i++) l_fences[i] = m_frames[i].renderFence;
        vkWaitForFences(m_device, FRAME_AMOUNT, l_fences.data(), true, 1000000000);

        m_mainDeletionQueue.Flush();

        CleanupSwapchain();

        vmaDestroyAllocator(m_allocator);

		vkDestroyDevice(m_device, nullptr);
		vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
		vkb::destroy_debug_utils_messenger(m_instance, m_debugMessenger);
		vkDestroyInstance(m_instance, nullptr);

        glfwDestroyWindow(m_window);
    }
}

AllocatedBuffer VulkanEngine::CreateBuffer(size_t p_allocSize, VkBufferUsageFlags p_usage, VmaMemoryUsage p_memoryUsage) {
    //allocate vertex buffer
	VkBufferCreateInfo l_bufferInfo = {};
    l_bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    l_bufferInfo.pNext = nullptr;

    l_bufferInfo.size = p_allocSize;
    l_bufferInfo.usage = p_usage;

	VmaAllocationCreateInfo l_vmaAllocInfo = {};
    l_vmaAllocInfo.usage = p_memoryUsage;

	AllocatedBuffer l_newBuffer{};

	//allocate the buffer
	VK_CHECK(vmaCreateBuffer(m_allocator, &l_bufferInfo, &l_vmaAllocInfo,
                             &l_newBuffer.m_buffer,
                             &l_newBuffer.m_allocation,
                             nullptr));

	return l_newBuffer;
}

size_t VulkanEngine::PadUniformBufferSize(size_t p_originalSize) const {
	// Calculate required alignment based on minimum device offset alignment
	size_t l_minUboAlignment = m_physicalProperties.limits.minUniformBufferOffsetAlignment;
	size_t l_alignedSize = p_originalSize;
	if (l_minUboAlignment > 0) {
        l_alignedSize = (l_alignedSize + l_minUboAlignment - 1) & ~(l_minUboAlignment - 1);
	}
	return l_alignedSize;
}

Material* VulkanEngine::CreateMaterial(VkPipeline p_pipeline, VkPipelineLayout p_layout, const std::string &p_name) {
    Material l_mat{};
    l_mat.pipeline = p_pipeline;
    l_mat.pipelineLayout = p_layout;
    m_materials[p_name] = l_mat;
    return &m_materials[p_name];
}

Material* VulkanEngine::GetMaterial(const std::string &p_name) {
    auto it = m_materials.find(p_name);
    if (it == m_materials.end()) {
        return nullptr;
    } else {
        return &(*it).second;
    }
}

Mesh* VulkanEngine::GetMesh(const std::string& p_name) {
    auto it = m_mesh.find(p_name);
    if (it == m_mesh.end()) {
        return nullptr;
    } else {
        return &(*it).second;
    }
}

void VulkanEngine::LoadImages() {
    Texture l_june{}, l_heightmap{};

	VulkanUtil::LoadImageFromFile(*this, "assets/june.png", l_june.image);

	VkImageViewCreateInfo l_imageInfo = VulkanInit::ImageViewCreateInfo(VK_FORMAT_R8G8B8A8_SRGB,
                                                                      l_june.image.m_image, VK_IMAGE_ASPECT_COLOR_BIT);
	vkCreateImageView(m_device, &l_imageInfo, nullptr, &l_june.imageView);

	m_loadedTextures["june"] = l_june;

    VulkanUtil::LoadImageFromFile(*this, "assets/heightmap.png", l_heightmap.image);

	VkImageViewCreateInfo l_heightmapInfo = VulkanInit::ImageViewCreateInfo(VK_FORMAT_R8G8B8A8_SRGB,
                                                                      l_heightmap.image.m_image, VK_IMAGE_ASPECT_COLOR_BIT);
	vkCreateImageView(m_device, &l_heightmapInfo, nullptr, &l_heightmap.imageView);

    m_loadedTextures["heightmap"] = l_heightmap;

    m_mainDeletionQueue.PushFunction([=]() {
        vkDestroyImageView(m_device, l_june.imageView, nullptr);
    });
}

void VulkanEngine::LoadMeshes() {
    const std::filesystem::path l_meshPath = std::filesystem::current_path() / "assets" / "mesh";

    for(auto const& l_directoryEntry : std::filesystem::directory_iterator(l_meshPath)) {
        if (l_directoryEntry.is_regular_file()) {
            std::string l_fileName = l_directoryEntry.path().filename().string();
            std::string l_fileExtension = l_directoryEntry.path().extension().string();
            if (l_fileExtension == ".obj") {
                std::string l_meshName = l_fileName.substr(0, l_fileName.size() - 4);
                Mesh l_mesh{};
                l_mesh.LoadFromObj(l_directoryEntry.path());
                UploadMesh(l_mesh);

                m_mesh[l_meshName] = l_mesh;
            }
        }
    }

    m_terrainMesh = VulkanUtil::CreateQuad(200.0f, 16);
    UploadMesh(m_terrainMesh);
}

void VulkanEngine::UploadMesh(Mesh &p_mesh) {
    const size_t bufferSize= p_mesh.m_vertices.size() * sizeof(Vertex);
	//allocate staging buffer
	VkBufferCreateInfo stagingBufferInfo = {};
	stagingBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	stagingBufferInfo.pNext = nullptr;

	stagingBufferInfo.size = bufferSize;
	stagingBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	//let the VMA library know that this data should be on CPU RAM
	VmaAllocationCreateInfo l_vmaAllocInfo = {};
    l_vmaAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

	AllocatedBuffer stagingBuffer{};

	//allocate the buffer
	VK_CHECK(vmaCreateBuffer(m_allocator, &stagingBufferInfo, &l_vmaAllocInfo,
		&stagingBuffer.m_buffer,
		&stagingBuffer.m_allocation,
		nullptr));

    void* data;
	vmaMapMemory(m_allocator, stagingBuffer.m_allocation, &data);

	memcpy(data, p_mesh.m_vertices.data(), p_mesh.m_vertices.size() * sizeof(Vertex));

	vmaUnmapMemory(m_allocator, stagingBuffer.m_allocation);

    //allocate vertex buffer
	VkBufferCreateInfo vertexBufferInfo = {};
	vertexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	//this is the total size, in bytes, of the buffer we are allocating
	vertexBufferInfo.size = bufferSize;
	//this buffer is going to be used as a Vertex Buffer
	vertexBufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;


	//let the VMA library know that this data should be writeable by CPU, but also readable by GPU
	l_vmaAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	//allocate the buffer
	VK_CHECK(vmaCreateBuffer(m_allocator, &vertexBufferInfo, &l_vmaAllocInfo,
		&p_mesh.m_vertexBuffer.m_buffer,
		&p_mesh.m_vertexBuffer.m_allocation,
		nullptr));

    ImmediateSubmit([=](VkCommandBuffer p_cmd) {
        VkBufferCopy copy;
		copy.dstOffset = 0;
		copy.srcOffset = 0;
		copy.size = bufferSize;
		vkCmdCopyBuffer(p_cmd, stagingBuffer.m_buffer, p_mesh.m_vertexBuffer.m_buffer, 1, &copy);
    });

	//add the destruction of triangle mesh buffer to the deletion queue
	m_mainDeletionQueue.PushFunction([=]() {
        vmaDestroyBuffer(m_allocator, p_mesh.m_vertexBuffer.m_buffer, p_mesh.m_vertexBuffer.m_allocation);
    });

    vmaDestroyBuffer(m_allocator, stagingBuffer.m_buffer, stagingBuffer.m_allocation);
}

void VulkanEngine::ImmediateSubmit(std::function<void(VkCommandBuffer p_cmd)>&& function) {
    vkResetFences(m_device, 1, &m_uploadContext.m_uploadFence);
    VkCommandBuffer cmd = m_uploadContext.m_commandBuffer;

	//begin the command buffer recording. We will use this command buffer exactly once before resetting, so we tell vulkan that
	VkCommandBufferBeginInfo cmdBeginInfo = VulkanInit::CommandBufferBeginInfo(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

	//execute the function
	function(cmd);

	VK_CHECK(vkEndCommandBuffer(cmd));

	VkSubmitInfo submit = VulkanInit::SubmitInfo(&cmd);

	//submit command buffer to the queue and execute it.
	// _uploadFence will now block until the graphic commands finish execution
	VK_CHECK(vkQueueSubmit(m_graphicsQueue, 1, &submit, m_uploadContext.m_uploadFence));

	vkWaitForFences(m_device, 1, &m_uploadContext.m_uploadFence, true, 9999999999);
	vkResetFences(m_device, 1, &m_uploadContext.m_uploadFence);

	// reset the command buffers inside the command pool
	vkResetCommandPool(m_device, m_uploadContext.m_commandPool, 0);
}

void VulkanEngine::InitScene() {
    if (m_mesh.empty()) {
        std::cerr << "No meshes loaded, scene is empty" << std::endl;
        return;
    }

    //create a sampler for the texture
	VkSamplerCreateInfo samplerInfo = VulkanInit::SamplerCreateInfo(VK_FILTER_NEAREST);

	VkSampler juneSampler, heightMapSampler;
	vkCreateSampler(m_device, &samplerInfo, nullptr, &juneSampler);
    vkCreateSampler(m_device, &samplerInfo, nullptr, &heightMapSampler);

	Material* texturedMat =	GetMaterial("textured");

	//allocate the descriptor set for single-texture to use on the material
	VkDescriptorSetAllocateInfo l_texAllocInfo = {};
	l_texAllocInfo.pNext = nullptr;
	l_texAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	l_texAllocInfo.descriptorPool = m_descriptorPool;
	l_texAllocInfo.descriptorSetCount = 1;
	l_texAllocInfo.pSetLayouts = &m_singleTextureSetLayout;

    VkDescriptorSetAllocateInfo l_heightmapAllocInfo = {};
	l_heightmapAllocInfo.pNext = nullptr;
	l_heightmapAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	l_heightmapAllocInfo.descriptorPool = m_descriptorPool;
	l_heightmapAllocInfo.descriptorSetCount = 1;
	l_heightmapAllocInfo.pSetLayouts = &m_heightmapSetLayout;

	vkAllocateDescriptorSets(m_device, &l_texAllocInfo, &texturedMat->textureSet);
    vkAllocateDescriptorSets(m_device, &l_heightmapAllocInfo, &m_terrainMaterial.heightmapSet);

    m_mainDeletionQueue.PushFunction([=]() {
        vkDestroySampler(m_device, juneSampler, nullptr);
        vkDestroySampler(m_device, heightMapSampler, nullptr);
    });

	//write to the descriptor set so that it points to our empire_diffuse texture
	VkDescriptorImageInfo imageBufferInfo{};
	imageBufferInfo.sampler = juneSampler;
	imageBufferInfo.imageView = m_loadedTextures["june"].imageView;
	imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	VkWriteDescriptorSet texture1 = VulkanInit::WriteDescriptorImage(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texturedMat->textureSet,
            &imageBufferInfo, 0);
	vkUpdateDescriptorSets(m_device, 1, &texture1, 0, nullptr);

    VkDescriptorImageInfo heightMapBufferInfo{};
    heightMapBufferInfo.sampler = heightMapSampler;
	heightMapBufferInfo.imageView = m_loadedTextures["heightmap"].imageView;
	heightMapBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet heightmap1 = VulkanInit::WriteDescriptorImage(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, m_terrainMaterial.heightmapSet,
            &heightMapBufferInfo, 0);
    vkUpdateDescriptorSets(m_device, 1, &heightmap1, 0, nullptr);

    m_camera.position = glm::vec3{0.0f,0.0f,-10.0f};
    m_camera.forward = glm::vec3{0.0f,0.0f,1.0f};
    m_camera.right = glm::cross(m_camera.forward, glm::vec3{0.0f,1.0f,0.0f});

    m_sceneParameters.sunlightDirection = glm::vec4{1.0f, 0.0f, 0.0f, 0.0f};
    m_sceneParameters.sunlightColor = glm::vec4{1.0f, 1.0f, 1.0f, 1.0f};

    std::string l_defaultMesh = m_mesh.begin()->first;

    m_terrain.mesh = &m_terrainMesh;
    m_terrain.material = &m_terrainMaterial;
}

void VulkanEngine::InitVulkan() {
    vkb::InstanceBuilder l_builder;

    auto l_instRet = l_builder.set_app_name("VulkanTerrain")
            .request_validation_layers(true)
            .require_api_version(1, 2, 0)
            .use_default_debug_messenger()
            .build();

    vkb::Instance l_vkbInstance = l_instRet.value();

    m_instance = l_vkbInstance.instance;
    m_debugMessenger = l_vkbInstance.debug_messenger;

    if (glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create surface.");
    }

    //use vkbootstrap to select a GPU.
	//We want a GPU that can write to the SDL surface and supports Vulkan 1.1
	vkb::PhysicalDeviceSelector l_selector{l_vkbInstance };
	vkb::PhysicalDevice l_physicalDevice = l_selector
		.set_minimum_version(1, 2)
		.set_surface(m_surface)
        .set_required_features({.tessellationShader = true, .fillModeNonSolid = true})
		.select()
		.value();

    if (l_physicalDevice.features.tessellationShader == VK_FALSE) {
        throw std::runtime_error("GPU does not support tessellation shaders");
    }

    if (l_physicalDevice.features.fillModeNonSolid == VK_FALSE) {
        throw std::runtime_error("GPU does not support tessellation shaders");
    }

	//create the final Vulkan device
	vkb::DeviceBuilder l_deviceBuilder{l_physicalDevice };

    VkPhysicalDeviceShaderDrawParametersFeatures l_shaderDrawParametersFeatures = {};
    l_shaderDrawParametersFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES;
    l_shaderDrawParametersFeatures.pNext = nullptr;
    l_shaderDrawParametersFeatures.shaderDrawParameters = VK_TRUE;

	vkb::Device l_vkbDevice = l_deviceBuilder.add_pNext(&l_shaderDrawParametersFeatures).build().value();

	// Get the VkDevice handle used in the rest of a Vulkan application
	m_device = l_vkbDevice.device;
	m_physicalDevice = l_physicalDevice.physical_device;
    m_physicalProperties = l_vkbDevice.physical_device.properties;

	std::cout << "The GPU has a minimum buffer alignment of " << m_physicalProperties.limits.minUniformBufferOffsetAlignment << std::endl;

    m_graphicsQueue = l_vkbDevice.get_queue(vkb::QueueType::graphics).value();
	m_graphicsQueueFamily = l_vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    VmaVulkanFunctions vulkanFunctions = {};
    vulkanFunctions.vkGetInstanceProcAddr = &vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr = &vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo l_allocatorInfo = {};
    l_allocatorInfo.physicalDevice = m_physicalDevice;
    l_allocatorInfo.device = m_device;
    l_allocatorInfo.instance = m_instance;
    l_allocatorInfo.pVulkanFunctions = &vulkanFunctions;
    vmaCreateAllocator(&l_allocatorInfo, &m_allocator);
}

void VulkanEngine::InitSwapchain() {
    vkb::SwapchainBuilder l_swapchainBuilder{m_physicalDevice, m_device, m_surface };

	vkb::Swapchain l_vkbSwapchain = l_swapchainBuilder
		.use_default_format_selection()
		//use vsync present mode
		.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
		.set_desired_extent(m_windowExtent.width, m_windowExtent.height)
		.build()
		.value();

	//store swapchain and its related images
	m_swapchain = l_vkbSwapchain.swapchain;
	m_swapchainImages = l_vkbSwapchain.get_images().value();
	m_swapchainImageViews = l_vkbSwapchain.get_image_views().value();

	m_swapchainImageFormat = l_vkbSwapchain.image_format;

    //depth image size will match the window
	VkExtent3D depthImageExtent = {
        m_windowExtent.width,
        m_windowExtent.height,
        1
    };

	//hardcoding the depth format to 32 bit float
	m_depthFormat = VK_FORMAT_D32_SFLOAT;

	//the depth image will be an image with the format we selected and Depth Attachment usage flag
	VkImageCreateInfo l_depthImgInfo = VulkanInit::ImageCreateInfo(
            m_depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depthImageExtent);

	//for the depth image, we want to allocate it from GPU local memory
	VmaAllocationCreateInfo l_depthImgAllocinfo = {};
    l_depthImgAllocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    l_depthImgAllocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	//allocate and create the image
	vmaCreateImage(
            m_allocator,
            &l_depthImgInfo,
            &l_depthImgAllocinfo,
            &m_depthImage.m_image,
            &m_depthImage.m_allocation,
            nullptr);

	//build an image-view for the depth image to use for rendering
	VkImageViewCreateInfo l_depthViewInfo = VulkanInit::ImageViewCreateInfo(m_depthFormat, m_depthImage.m_image, VK_IMAGE_ASPECT_DEPTH_BIT);

	VK_CHECK(vkCreateImageView(m_device, &l_depthViewInfo, nullptr, &m_depthImageView));

    /*m_mainDeletionQueue.PushFunction([=]() {
        vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
        vkDestroyImageView(m_device, m_depthImageView, nullptr);
        vmaDestroyImage(m_allocator, m_depthImage.m_image, m_depthImage.m_allocation);
    });*/

    std::cout << "Create Swapchain, extent " << m_windowExtent.width << "x" << m_windowExtent.height << std::endl;
}

void VulkanEngine::InitCommands() {
    VkCommandPoolCreateInfo l_uploadCommandPoolInfo = VulkanInit::CommandPoolCreateInfo(m_graphicsQueueFamily);
	//create pool for upload context
	VK_CHECK(vkCreateCommandPool(m_device, &l_uploadCommandPoolInfo, nullptr, &m_uploadContext.m_commandPool));

	m_mainDeletionQueue.PushFunction([=]() {
		vkDestroyCommandPool(m_device, m_uploadContext.m_commandPool, nullptr);
	});

	//allocate the default command buffer that we will use for the instant commands
	VkCommandBufferAllocateInfo l_cmdAllocInfo = VulkanInit::CommandBufferAllocateInfo(m_uploadContext.m_commandPool, 1);

	// VkCommandBuffer cmd;
	VK_CHECK(vkAllocateCommandBuffers(m_device, &l_cmdAllocInfo, &m_uploadContext.m_commandBuffer));

    VkCommandPoolCreateInfo commandPoolInfo = VulkanInit::CommandPoolCreateInfo(
            m_graphicsQueueFamily,
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	for (int i = 0; i < FRAME_AMOUNT; i++) {
		VK_CHECK(vkCreateCommandPool(m_device, &commandPoolInfo, nullptr, &m_frames[i].commandPool));

		//allocate the default command buffer that we will use for rendering
		VkCommandBufferAllocateInfo cmdAllocInfo = VulkanInit::CommandBufferAllocateInfo(m_frames[i].commandPool, 1);

		VK_CHECK(vkAllocateCommandBuffers(m_device, &cmdAllocInfo, &m_frames[i].mainCommandBuffer));

		m_mainDeletionQueue.PushFunction([=]() {
			vkDestroyCommandPool(m_device, m_frames[i].commandPool, nullptr);
		});
	}
}

void VulkanEngine::InitDefaultRenderpass() {
    // the renderpass will use this color attachment.
	VkAttachmentDescription color_attachment = {};
	//the attachment will have the format needed by the swapchain
	color_attachment.format = m_swapchainImageFormat;
	//1 sample, we won't be doing MSAA
	color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	// we Clear when this attachment is loaded
	color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	// we keep the attachment stored when the renderpass ends
	color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	//we don't care about stencil
	color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

	//we don't know or care about the starting layout of the attachment
	color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	//after the renderpass ends, the image has to be on a layout ready for display
	color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_attachment_ref = {};
	//attachment number will index into the pAttachments array in the parent renderpass itself
	color_attachment_ref.attachment = 0;
	color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;


    VkAttachmentDescription depth_attachment = {};
    // Depth attachment
    depth_attachment.flags = 0;
    depth_attachment.format = m_depthFormat;
    depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depth_attachment_ref = {};
    depth_attachment_ref.attachment = 1;
    depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	//we are going to create 1 subpass, which is the minimum you can do
	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &color_attachment_ref;
    subpass.pDepthStencilAttachment = &depth_attachment_ref;

    VkAttachmentDescription attachments[2] = { color_attachment,depth_attachment };

	VkRenderPassCreateInfo render_pass_info = {};
	render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	//2 attachments from said array
	render_pass_info.attachmentCount = 2;
	render_pass_info.pAttachments = &attachments[0];
	render_pass_info.subpassCount = 1;
	render_pass_info.pSubpasses = &subpass;

    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkSubpassDependency depth_dependency = {};
    depth_dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    depth_dependency.dstSubpass = 0;
    depth_dependency.srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    depth_dependency.srcAccessMask = 0;
    depth_dependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    depth_dependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkSubpassDependency dependencies[2] = { dependency, depth_dependency };

    render_pass_info.dependencyCount = 2;
    render_pass_info.pDependencies = &dependencies[0];

	VK_CHECK(vkCreateRenderPass(m_device, &render_pass_info, nullptr, &m_renderPass));

    m_mainDeletionQueue.PushFunction([=]() {
        vkDestroyRenderPass(m_device, m_renderPass, nullptr);
    });
}

void VulkanEngine::InitFramebuffers() {
    //create the framebuffers for the swapchain images. This will connect the render-pass to the images for rendering
	VkFramebufferCreateInfo l_framebufferInfo = {};
    l_framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    l_framebufferInfo.pNext = nullptr;

    l_framebufferInfo.renderPass = m_renderPass;
    l_framebufferInfo.width = m_windowExtent.width;
    l_framebufferInfo.height = m_windowExtent.height;
    l_framebufferInfo.layers = 1;

	//grab how many images we have in the swapchain
	const size_t l_swapchainImageCount = m_swapchainImages.size();
	m_framebuffers = std::vector<VkFramebuffer>(l_swapchainImageCount);

	//create framebuffers for each of the swapchain image views
	for (size_t i = 0; i < l_swapchainImageCount; i++) {
        VkImageView l_attachments[2];
        l_attachments[0] = m_swapchainImageViews[i];
        l_attachments[1] = m_depthImageView;

        l_framebufferInfo.pAttachments = l_attachments;
        l_framebufferInfo.attachmentCount = 2;
		VK_CHECK(vkCreateFramebuffer(m_device, &l_framebufferInfo, nullptr, &m_framebuffers[i]));

        /*m_mainDeletionQueue.PushFunction([=]() {
			vkDestroyFramebuffer(m_device, m_framebuffers[i], nullptr);
			vkDestroyImageView(m_device, m_swapchainImageViews[i], nullptr);
    	});*/
	}
}

void VulkanEngine::InitSyncStructures() {
    //create synchronization structures

	VkFenceCreateInfo fenceCreateInfo = {};
	fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceCreateInfo.pNext = nullptr;
	//we want to create the fence with the Create Signaled flag, so we can wait on it before using it on a GPU command (for the first frame)
	fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	//for the semaphores we don't need any flags
	VkSemaphoreCreateInfo semaphoreCreateInfo = {};
	semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	semaphoreCreateInfo.pNext = nullptr;
	semaphoreCreateInfo.flags = 0;

    VkFenceCreateInfo l_uploadFenceCreateInfo{};
    l_uploadFenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	l_uploadFenceCreateInfo.pNext = nullptr;
	//we want to create the fence with the Create Signaled flag, so we can wait on it before using it on a GPU command (for the first frame)
	l_uploadFenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VK_CHECK(vkCreateFence(m_device, &l_uploadFenceCreateInfo, nullptr, &m_uploadContext.m_uploadFence));

    m_mainDeletionQueue.PushFunction([=]() {
        vkDestroyFence(m_device, m_uploadContext.m_uploadFence, nullptr);
    });

    for (int i = 0; i < FRAME_AMOUNT; i++) {
        VK_CHECK(vkCreateFence(m_device, &fenceCreateInfo, nullptr, &m_frames[i].renderFence));

        //enqueue the destruction of the fence
        m_mainDeletionQueue.PushFunction([=]() {
            vkDestroyFence(m_device, m_frames[i].renderFence, nullptr);
            });


        VK_CHECK(vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr, &m_frames[i].presentSemaphore));
        VK_CHECK(vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr, &m_frames[i].renderSemaphore));

        //enqueue the destruction of semaphores
        m_mainDeletionQueue.PushFunction([=]() {
            vkDestroySemaphore(m_device, m_frames[i].presentSemaphore, nullptr);
            vkDestroySemaphore(m_device, m_frames[i].renderSemaphore, nullptr);
            });
	}
}

void VulkanEngine::InitDescriptors() {
    const size_t sceneParamBufferSize = FRAME_AMOUNT * PadUniformBufferSize(sizeof(GPUSceneData));

    m_sceneParameterBuffer = CreateBuffer(
            sceneParamBufferSize,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU);

    std::vector<VkDescriptorPoolSize> sizes =
	{
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 10 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10 },
	};

	VkDescriptorPoolCreateInfo pool_info = {};
	pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.flags = 0;
	pool_info.maxSets = 10;
	pool_info.poolSizeCount = (uint32_t)sizes.size();
	pool_info.pPoolSizes = sizes.data();

	vkCreateDescriptorPool(m_device, &pool_info, nullptr, &m_descriptorPool);

    VkDescriptorSetLayoutBinding l_cameraBufferBinding = VulkanInit::DescriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,0);

    VkDescriptorSetLayoutBinding l_sceneBufferBinding = VulkanInit::DescriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, 1);

    VkDescriptorSetLayoutBinding l_bindings[] = { l_cameraBufferBinding, l_sceneBufferBinding };

	VkDescriptorSetLayoutCreateInfo l_setinfo = {};
    l_setinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    l_setinfo.pNext = nullptr;
	//no flags
	l_setinfo.flags = 0;
	l_setinfo.bindingCount = 2;
	l_setinfo.pBindings = l_bindings;

	vkCreateDescriptorSetLayout(m_device, &l_setinfo, nullptr, &m_globalSetLayout);

    VkDescriptorSetLayoutBinding objectBind = VulkanInit::DescriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_SHADER_STAGE_VERTEX_BIT, 0);

	VkDescriptorSetLayoutCreateInfo set2info = {};
	set2info.bindingCount = 1;
	set2info.flags = 0;
	set2info.pNext = nullptr;
	set2info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	set2info.pBindings = &objectBind;

	vkCreateDescriptorSetLayout(m_device, &set2info, nullptr, &m_objectSetLayout);

    VkDescriptorSetLayoutBinding textureBind = VulkanInit::DescriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT, 0);

	VkDescriptorSetLayoutCreateInfo set3info = {};
	set3info.bindingCount = 1;
	set3info.flags = 0;
	set3info.pNext = nullptr;
	set3info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	set3info.pBindings = &textureBind;

	vkCreateDescriptorSetLayout(m_device, &set3info, nullptr, &m_singleTextureSetLayout);

    VkDescriptorSetLayoutBinding l_heightmapBind = VulkanInit::DescriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, 0);

	VkDescriptorSetLayoutCreateInfo set4info = {};
	set4info.bindingCount = 1;
	set4info.flags = 0;
	set4info.pNext = nullptr;
	set4info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	set4info.pBindings = &l_heightmapBind;

	vkCreateDescriptorSetLayout(m_device, &set4info, nullptr, &m_heightmapSetLayout);

    for (int i = 0; i < FRAME_AMOUNT; i++)
	{
        constexpr size_t MAX_OBJECTS = 10'000;
        m_frames[i].objectBuffer = CreateBuffer(
                sizeof(GPUObjectData) * MAX_OBJECTS,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_CPU_TO_GPU);

		m_frames[i].cameraBuffer = CreateBuffer(sizeof(GPUCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

        VkDescriptorSetAllocateInfo allocInfo ={};
		allocInfo.pNext = nullptr;
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		//using the pool we just set
		allocInfo.descriptorPool = m_descriptorPool;
		//only 1 descriptor
		allocInfo.descriptorSetCount = 1;
		//using the global data layout
		allocInfo.pSetLayouts = &m_globalSetLayout;

		vkAllocateDescriptorSets(m_device, &allocInfo, &m_frames[i].globalDescriptor);

        VkDescriptorSetAllocateInfo objectSetAlloc = {};
		objectSetAlloc.pNext = nullptr;
		objectSetAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		objectSetAlloc.descriptorPool = m_descriptorPool;
		objectSetAlloc.descriptorSetCount = 1;
		objectSetAlloc.pSetLayouts = &m_objectSetLayout;

		vkAllocateDescriptorSets(m_device, &objectSetAlloc, &m_frames[i].objectDescriptor);

        VkDescriptorBufferInfo cameraInfo;
		cameraInfo.buffer = m_frames[i].cameraBuffer.m_buffer;
		cameraInfo.offset = 0;
		cameraInfo.range = sizeof(GPUCameraData);

		VkDescriptorBufferInfo sceneInfo;
		sceneInfo.buffer = m_sceneParameterBuffer.m_buffer;
		sceneInfo.offset = 0;
		sceneInfo.range = sizeof(GPUSceneData);

        VkDescriptorBufferInfo objectBufferInfo;
		objectBufferInfo.buffer = m_frames[i].objectBuffer.m_buffer;
		objectBufferInfo.offset = 0;
		objectBufferInfo.range = sizeof(GPUObjectData) * MAX_OBJECTS;

		VkWriteDescriptorSet cameraWrite = VulkanInit::WriteDescriptorSet(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, m_frames[i].globalDescriptor,&cameraInfo,0);
		VkWriteDescriptorSet sceneWrite = VulkanInit::WriteDescriptorSet(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, m_frames[i].globalDescriptor, &sceneInfo, 1);
        VkWriteDescriptorSet objectWrite = VulkanInit::WriteDescriptorSet(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, m_frames[i].objectDescriptor, &objectBufferInfo, 0);

		VkWriteDescriptorSet setWrites[] = { cameraWrite,sceneWrite, objectWrite };

		vkUpdateDescriptorSets(m_device, 3, setWrites, 0, nullptr);
    }

	// add buffers to deletion queues
	for (int i = 0; i < FRAME_AMOUNT; i++)
	{
		m_mainDeletionQueue.PushFunction([=]() {
            vmaDestroyBuffer(m_allocator, m_frames[i].objectBuffer.m_buffer, m_frames[i].objectBuffer.m_allocation);
			vmaDestroyBuffer(m_allocator, m_frames[i].cameraBuffer.m_buffer, m_frames[i].cameraBuffer.m_allocation);
		});
	}

    m_mainDeletionQueue.PushFunction([=]() {
		vkDestroyDescriptorSetLayout(m_device, m_globalSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(m_device, m_objectSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(m_device, m_singleTextureSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(m_device, m_heightmapSetLayout, nullptr);
		vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);

        vmaDestroyBuffer(m_allocator, m_sceneParameterBuffer.m_buffer, m_sceneParameterBuffer.m_allocation);
	});
}

void VulkanEngine::InitPipelines() {
    VkShaderModule l_litFragShader, l_litTexFragShader, l_meshVertShader;
	if (!LoadShaderModule("./shaders/bin/lit_main.spv", &l_litFragShader))
		throw std::runtime_error("Error when building the main fragment shader module");
	else
        std::cout << "Main fragment shader successfully loaded" << std::endl;

    if (!LoadShaderModule("./shaders/bin/lit_texture.spv", &l_litTexFragShader))
		throw std::runtime_error("Error when building the texture fragment shader module");
	else
        std::cout << "Texture fragment shader successfully loaded" << std::endl;

	if (!LoadShaderModule("./shaders/bin/mesh.spv", &l_meshVertShader))
		throw std::runtime_error("Error when building the mesh vertex shader module");
	else
        std::cout << "Mesh vertex shader successfully loaded" << std::endl;

    VkPipelineLayoutCreateInfo l_meshPipelineLayoutInfo = VulkanInit::PipelineLayoutCreateInfo();
    l_meshPipelineLayoutInfo.pushConstantRangeCount = 0;

    VkDescriptorSetLayout setLayouts[] = { m_globalSetLayout, m_objectSetLayout };
    l_meshPipelineLayoutInfo.setLayoutCount = 2;
    l_meshPipelineLayoutInfo.pSetLayouts = setLayouts;

    VkPipelineLayout l_meshPipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(m_device, &l_meshPipelineLayoutInfo, nullptr, &l_meshPipelineLayout));

    //create pipeline layout for the textured mesh, which has 3 descriptor sets
	//we start from  the normal mesh layout
	VkPipelineLayoutCreateInfo l_texturePipelineLayoutInfo = l_meshPipelineLayoutInfo;

	VkDescriptorSetLayout l_texturedSetLayouts[] = {
            m_globalSetLayout, m_objectSetLayout, m_singleTextureSetLayout
    };

	l_texturePipelineLayoutInfo.setLayoutCount = 3;
	l_texturePipelineLayoutInfo.pSetLayouts = l_texturedSetLayouts;

	VkPipelineLayout l_texturedPipeLayout;
	VK_CHECK(vkCreatePipelineLayout(m_device, &l_texturePipelineLayoutInfo, nullptr, &l_texturedPipeLayout));

    //build the stage-create-info for both vertex and fragment stages. This lets the pipeline know the shader modules per stage
	PipelineBuilder pipelineBuilder;

	pipelineBuilder.m_shaderStages.push_back(VulkanInit::PipelineShaderStageCreateInfo(
            VK_SHADER_STAGE_FRAGMENT_BIT, l_litFragShader));

	//vertex input controls how to read vertices from vertex buffers. We aren't using it yet
	pipelineBuilder.m_vertexInputInfo = VulkanInit::VertexInputStateCreateInfo();

	//input assembly is the configuration for drawing triangle lists, strips, or individual points.
	//we are just going to draw triangle list
	pipelineBuilder.m_inputAssembly = VulkanInit::InputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

	//build viewport and scissor from the swapchain extents
	pipelineBuilder.m_viewport.x = 0.0f;
	pipelineBuilder.m_viewport.y = 0.0f;
	pipelineBuilder.m_viewport.width = (float)m_windowExtent.width;
	pipelineBuilder.m_viewport.height = (float)m_windowExtent.height;
	pipelineBuilder.m_viewport.minDepth = 0.0f;
	pipelineBuilder.m_viewport.maxDepth = 1.0f;

	pipelineBuilder.m_scissor.offset = { 0, 0 };
	pipelineBuilder.m_scissor.extent = m_windowExtent;

	//configure the rasterizer to draw filled triangles
	pipelineBuilder.m_rasterizer = VulkanInit::RasterizationStateCreateInfo(VK_POLYGON_MODE_FILL);
	//we don't use multisampling, so just run the default one
	pipelineBuilder.m_multisampling = VulkanInit::MultisampleStateCreateInfo();
	//a single blend attachment with no blending and writing to RGBA
	pipelineBuilder.m_colorBlendAttachment = VulkanInit::ColorBlendAttachmentState();
    pipelineBuilder.m_depthStencil = VulkanInit::DepthStencilCreateInfo(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);

    VertexInputDescription vertexDescription = Vertex::GetVertexDescription();

	// Vertex input attributes
	pipelineBuilder.m_vertexInputInfo.pVertexAttributeDescriptions = vertexDescription.attributes.data();
	pipelineBuilder.m_vertexInputInfo.vertexAttributeDescriptionCount = vertexDescription.attributes.size();

    // Vertex input bindings
	pipelineBuilder.m_vertexInputInfo.pVertexBindingDescriptions = vertexDescription.bindings.data();
	pipelineBuilder.m_vertexInputInfo.vertexBindingDescriptionCount = vertexDescription.bindings.size();

	//clear the shader stages for the builder
	pipelineBuilder.m_shaderStages.clear();

	//add the other shaders
	pipelineBuilder.m_shaderStages.push_back(
            VulkanInit::PipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, l_meshVertShader));

	//make sure that l_litFragShader is holding the compiled colored_triangle.frag
	pipelineBuilder.m_shaderStages.push_back(
            VulkanInit::PipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, l_litFragShader));
    pipelineBuilder.m_pipelineLayout = l_meshPipelineLayout;

    // --- DEFAULT PIPELINE ---
	VkPipeline l_meshPipeline = pipelineBuilder.BuildPipeline(m_device, m_renderPass);

    CreateMaterial(l_meshPipeline, l_meshPipelineLayout, "defaultmesh");

    // --- WIREFRAME PIPELINE ---
    pipelineBuilder.m_rasterizer.polygonMode = VK_POLYGON_MODE_LINE;
    VkPipeline l_meshWirePipeline = pipelineBuilder.BuildPipeline(m_device, m_renderPass);
    CreateMaterial(l_meshWirePipeline, l_meshPipelineLayout, "wiremesh");

    // --- TEXTURE PIPELINE ---
    pipelineBuilder.m_rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	pipelineBuilder.m_shaderStages.clear();
	pipelineBuilder.m_shaderStages.push_back(
		VulkanInit::PipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, l_meshVertShader));

	pipelineBuilder.m_shaderStages.push_back(
		VulkanInit::PipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, l_litTexFragShader));

	pipelineBuilder.m_pipelineLayout = l_texturedPipeLayout;
	VkPipeline texPipeline = pipelineBuilder.BuildPipeline(m_device, m_renderPass);
	CreateMaterial(texPipeline, l_texturedPipeLayout, "textured");

    // --- TERRAIN PIPELINE ---
    VkPipelineLayoutCreateInfo l_terrainPipelineLayoutInfo = VulkanInit::PipelineLayoutCreateInfo();
    l_terrainPipelineLayoutInfo.pushConstantRangeCount = 0;
    VkDescriptorSetLayout l_terrainSetLayouts[] = {
            m_globalSetLayout, m_objectSetLayout, m_heightmapSetLayout
    };

	l_terrainPipelineLayoutInfo.setLayoutCount = 3;
	l_terrainPipelineLayoutInfo.pSetLayouts = l_terrainSetLayouts;

	VkPipelineLayout l_terrainPipelineLayout;
	VK_CHECK(vkCreatePipelineLayout(m_device, &l_terrainPipelineLayoutInfo, nullptr, &l_terrainPipelineLayout));

    VkShaderModule l_terrainVert, l_terrainTessControlShader, l_terrainTessEvalShader, l_terrainFrag;
    if (!LoadShaderModule("./shaders/bin/terrain_vert.spv", &l_terrainVert))
        throw std::runtime_error("Error when building the terrain vertex shader module");
    else
        std::cout << "Vertex shader successfully loaded" << std::endl;

    if (!LoadShaderModule("./shaders/bin/terrain_tess_control.spv", &l_terrainTessControlShader))
        throw std::runtime_error("Error when building the terrain tessellation control shader module");
    else
        std::cout << "Terrain tessellation control shader successfully loaded" << std::endl;

    if (!LoadShaderModule("./shaders/bin/terrain_tess_eval.spv", &l_terrainTessEvalShader))
        throw std::runtime_error("Error when building the terrain tessellation evaluation shader module");
    else
        std::cout << "Terrain tessellation evaluation shader successfully loaded" << std::endl;

    if (!LoadShaderModule("./shaders/bin/terrain_frag.spv", &l_terrainFrag))
        throw std::runtime_error("Error when building the terrain fragment shader module");
    else
        std::cout << "Fragment shader successfully loaded" << std::endl;

    pipelineBuilder.m_pipelineLayout = l_terrainPipelineLayout;
    pipelineBuilder.m_inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    pipelineBuilder.m_tessellationState = VulkanInit::TessellationStateCreateInfo(4);

    pipelineBuilder.m_shaderStages.clear();
    pipelineBuilder.m_shaderStages.push_back(
		VulkanInit::PipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, l_terrainVert));

	pipelineBuilder.m_shaderStages.push_back(
		VulkanInit::PipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, l_terrainFrag));

    pipelineBuilder.m_shaderStages.push_back(
		VulkanInit::PipelineShaderStageCreateInfo(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, l_terrainTessControlShader));

    pipelineBuilder.m_shaderStages.push_back(
		VulkanInit::PipelineShaderStageCreateInfo(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, l_terrainTessEvalShader));

    pipelineBuilder.m_rasterizer.polygonMode = VK_POLYGON_MODE_LINE;
    m_terrainWiredPipeline = pipelineBuilder.BuildPipeline(m_device, m_renderPass);
    pipelineBuilder.m_rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    m_terrainPipeline = pipelineBuilder.BuildPipeline(m_device, m_renderPass);

	m_terrainMaterial.pipeline = m_drawTerrainWire ? m_terrainWiredPipeline : m_terrainPipeline;
    m_terrainMaterial.pipelineLayout = l_terrainPipelineLayout;


    m_mainDeletionQueue.PushFunction([=]() {
        vkDestroyShaderModule(m_device, l_litTexFragShader, nullptr);
        vkDestroyShaderModule(m_device, l_litFragShader, nullptr);
        vkDestroyShaderModule(m_device, l_meshVertShader, nullptr);

        vkDestroyShaderModule(m_device, l_terrainVert, nullptr);
        vkDestroyShaderModule(m_device, l_terrainTessControlShader, nullptr);
        vkDestroyShaderModule(m_device, l_terrainTessEvalShader, nullptr);
        vkDestroyShaderModule(m_device, l_terrainFrag, nullptr);

        vkDestroyPipeline(m_device, m_terrainPipeline, nullptr);
        vkDestroyPipeline(m_device, m_terrainWiredPipeline, nullptr);

        vkDestroyPipeline(m_device, texPipeline, nullptr);
        vkDestroyPipeline(m_device, l_meshPipeline, nullptr);
        vkDestroyPipeline(m_device, l_meshWirePipeline, nullptr);

		vkDestroyPipelineLayout(m_device, l_meshPipelineLayout, nullptr);
        vkDestroyPipelineLayout(m_device, l_texturedPipeLayout, nullptr);
    });
}

void VulkanEngine::InitIMGUI() {
    //1: create descriptor pool for IMGUI
	// the size of the pool is very oversize, but it's copied from imgui demo itself.
	VkDescriptorPoolSize pool_sizes[] =
	{
		{ VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
	};

	VkDescriptorPoolCreateInfo pool_info = {};
	pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	pool_info.maxSets = 1000;
	pool_info.poolSizeCount = std::size(pool_sizes);
	pool_info.pPoolSizes = pool_sizes;

	VkDescriptorPool imguiPool;
	VK_CHECK(vkCreateDescriptorPool(m_device, &pool_info, nullptr, &imguiPool));


	// 2: initialize imgui library
    IMGUI_CHECKVERSION();
	//this initializes the core structures of imgui
	ImGui::CreateContext();

	//this initializes imgui for GLFW
	ImGui_ImplGlfw_InitForVulkan(m_window, true);

	//this initializes imgui for Vulkan
	ImGui_ImplVulkan_InitInfo init_info = {};
	init_info.Instance = m_instance;
	init_info.PhysicalDevice = m_physicalDevice;
	init_info.Device = m_device;
	init_info.Queue = m_graphicsQueue;
	init_info.DescriptorPool = imguiPool;
	init_info.MinImageCount = 3;
	init_info.ImageCount = 3;
	init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

	ImGui_ImplVulkan_Init(&init_info, m_renderPass);

    m_imguiVulkanWindow.Surface = m_surface;

	//execute a gpu command to upload imgui font textures
	ImmediateSubmit([&](VkCommandBuffer cmd) {
        ImGui_ImplVulkan_CreateFontsTexture(cmd);
    });

    vkDeviceWaitIdle(m_device);

	//clear font textures from cpu data
	ImGui_ImplVulkan_DestroyFontUploadObjects();

	//add the destroy the imgui created structures
	m_mainDeletionQueue.PushFunction([=]() {
        vkDestroyDescriptorPool(m_device, imguiPool, nullptr);
        ImGui_ImplVulkan_Shutdown();
    });
}

void VulkanEngine::CleanupSwapchain() {
    vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
    vkDestroyImageView(m_device, m_depthImageView, nullptr);
    vmaDestroyImage(m_allocator, m_depthImage.m_image, m_depthImage.m_allocation);

    const size_t l_swapchainImageCount = m_swapchainImages.size();
    for (size_t i = 0; i < l_swapchainImageCount; i++) {
        vkDestroyFramebuffer(m_device, m_framebuffers[i], nullptr);
        vkDestroyImageView(m_device, m_swapchainImageViews[i], nullptr);
	}
}

void VulkanEngine::RecreateSwapchain() {
    vkDeviceWaitIdle(m_device);

    m_windowExtent = m_desiredWindowExtent;

    // cleanup swapchain
    CleanupSwapchain();

    // create swapchain
    InitSwapchain();
    // create framebuffers (& imageviews)
    InitFramebuffers();
}

void VulkanEngine::DrawUI() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::BeginMainMenuBar();
    if (ImGui::MenuItem("DemoWindow", "CTRL + D", *m_showDemoWindow, true)) *m_showDemoWindow = !*m_showDemoWindow;
    if (ImGui::MenuItem("InspectorWindow", "CTRL + I", *m_showInspectorWindow, true)) *m_showInspectorWindow = !*m_showInspectorWindow;
    if (ImGui::MenuItem("HeightMapWindow", "CTRL + H", false, false)) *m_showHeightMapWindow = !*m_showHeightMapWindow;
    ImGui::EndMainMenuBar();

    if (*m_showDemoWindow) ImGui::ShowDemoWindow(m_showDemoWindow);

    if (*m_showInspectorWindow) {
        if (ImGui::Begin("Inspector", m_showInspectorWindow)) {
            ImGui::BeginGroup();
            ImGui::Text("Scene");
            ImGui::SliderFloat3("LightDirection", &m_sceneParameters.sunlightDirection.x, -1.0f, 1.0f);
            ImGui::ColorEdit3("LightColor", &m_sceneParameters.sunlightColor.x);
            ImGui::SliderFloat("Terrain subdivision", &m_sceneParameters.terrainSubdivision, 1.0f, 64.0f, "%.0f");
            ImGui::SliderFloat("Terrain height", &m_sceneParameters.displacementFactor, 1.0f, 20.0f, "%.0f");
            ImGui::EndGroup();
            constexpr ImGuiTreeNodeFlags BASE_NODE_FLAGS = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;

            uint32_t l_id{0};
            for(auto &l_renderObject : m_renderables) {

                bool node_open = ImGui::TreeNodeEx(std::to_string(l_id).c_str(), BASE_NODE_FLAGS, "%s(%d)",
                                               l_renderObject.meshName.c_str(), l_id);

                if (node_open) {
                    if (ImGui::BeginCombo("##Mesh", l_renderObject.meshName.c_str(), ImGuiComboFlags_NoArrowButton)) {
                        for(auto &l_mesh : m_mesh) {
                            bool is_selected = (l_renderObject.meshName == l_mesh.first);
                            if (ImGui::Selectable(l_mesh.first.c_str(), is_selected)) {
                                l_renderObject.meshName = l_mesh.first;
                                l_renderObject.mesh = &l_mesh.second;
                            }
                            if (is_selected) {
                                ImGui::SetItemDefaultFocus();
                            }
                        }
                        ImGui::EndCombo();
                    }
                    if (ImGui::BeginCombo("##Material", l_renderObject.materialName.c_str(), ImGuiComboFlags_NoArrowButton)) {
                        for(auto &l_material : m_materials) {
                            bool is_selected = (l_renderObject.materialName == l_material.first);
                            if (ImGui::Selectable(l_material.first.c_str(), is_selected)) {
                                l_renderObject.materialName = l_material.first;
                                l_renderObject.material = &l_material.second;
                            }
                            if (is_selected) {
                                ImGui::SetItemDefaultFocus();
                            }
                        }
                        ImGui::EndCombo();
                    }
                    ImGui::SliderFloat3(std::format("Position##{}", l_id).c_str(), &l_renderObject.position.x, -10.0f, 10.0f);
                    ImGui::SliderFloat3(std::format("Scale##{}", l_id).c_str(), &l_renderObject.scale.x, 1.0f, 10.0f);
                    ImGui::TreePop();
                }

                l_id++;
            }

        }
        ImGui::End();
    }

    ImGui::Render();
}

void VulkanEngine::Draw() {
    VK_CHECK(vkWaitForFences(m_device, 1, &GetCurrentFrame().renderFence, VK_TRUE, 1000000000));

    //request image from the swapchain, one second timeout
	uint32_t swapchainImageIndex;
	VkResult l_acquireResult = vkAcquireNextImageKHR(m_device, m_swapchain, 1000000000,
                                                     GetCurrentFrame().presentSemaphore, nullptr,
                                                     &swapchainImageIndex);

    if (l_acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        RecreateSwapchain();
        return;
    } else if (l_acquireResult != VK_SUCCESS && l_acquireResult != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to acquire swapchain image.");
    }

	VK_CHECK(vkResetFences(m_device, 1, &GetCurrentFrame().renderFence));

    //now that we are sure that the commands finished executing, we can safely reset the command buffer to begin recording again.
	VK_CHECK(vkResetCommandBuffer(GetCurrentFrame().mainCommandBuffer, 0));

    //naming it cmd for shorter writing
	VkCommandBuffer cmd = GetCurrentFrame().mainCommandBuffer;

	//begin the command buffer recording. We will use this command buffer exactly once, so we want to let Vulkan know that
	VkCommandBufferBeginInfo cmdBeginInfo = {};
	cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	cmdBeginInfo.pNext = nullptr;

	cmdBeginInfo.pInheritanceInfo = nullptr;
	cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    //make a clear-color from frame number. This will flash with a 120*pi frame period.
	VkClearValue clearValue;
	clearValue.color = { { 0.0f, 0.0f, 0.0f, 1.0f } };

	//clear depth at 1
	VkClearValue depthClear;
	depthClear.depthStencil.depth = 1.f;

	//start the main renderpass.
	//We will use the clear color from above, and the framebuffer of the index the swapchain gave us
	VkRenderPassBeginInfo rpInfo = {};
	rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	rpInfo.pNext = nullptr;

	rpInfo.renderPass = m_renderPass;
	rpInfo.renderArea.offset.x = 0;
	rpInfo.renderArea.offset.y = 0;
	rpInfo.renderArea.extent = m_windowExtent;
	rpInfo.framebuffer = m_framebuffers[swapchainImageIndex];

	//connect clear values
	rpInfo.clearValueCount = 2;
	VkClearValue clearValues[] = { clearValue, depthClear };
	rpInfo.pClearValues = &clearValues[0];

	vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

    DrawObjects(cmd, m_renderables.data(), m_renderables.size());

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    //finalize the render pass
	vkCmdEndRenderPass(cmd);
	//finalize the command buffer (we can no longer add commands, but it can now be executed)
	VK_CHECK(vkEndCommandBuffer(cmd));

    //prepare the submission to the queue.
	//we want to wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready
	//we will signal the _renderSemaphore, to signal that rendering has finished

	VkSubmitInfo submit = {};
	submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.pNext = nullptr;

	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

	submit.pWaitDstStageMask = &waitStage;

	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores = &GetCurrentFrame().presentSemaphore;

	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &GetCurrentFrame().renderSemaphore;

	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &cmd;

	//submit command buffer to the queue and execute it.
	// _renderFence will now block until the graphic commands finish execution
	VK_CHECK(vkQueueSubmit(m_graphicsQueue, 1, &submit, GetCurrentFrame().renderFence));

    // this will put the image we just rendered into the visible window.
	// we want to wait on the _renderSemaphore for that,
	// as it's necessary that drawing commands have finished before the image is displayed to the user
	VkPresentInfoKHR presentInfo = {};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.pNext = nullptr;

	presentInfo.pSwapchains = &m_swapchain;
	presentInfo.swapchainCount = 1;

	presentInfo.pWaitSemaphores = &GetCurrentFrame().renderSemaphore;
	presentInfo.waitSemaphoreCount = 1;

	presentInfo.pImageIndices = &swapchainImageIndex;

	VkResult l_presentResult = vkQueuePresentKHR(m_graphicsQueue, &presentInfo);

    if (l_presentResult == VK_ERROR_OUT_OF_DATE_KHR || l_presentResult == VK_SUBOPTIMAL_KHR || m_framebufferResized) {
        RecreateSwapchain();
        m_framebufferResized = false;
    } else if (l_presentResult != VK_SUCCESS) {
        throw std::runtime_error("failed to present swap chain image!");
    }

	//increase the number of frames drawn
	m_frameNumber++;
}

void VulkanEngine::DrawObjects(VkCommandBuffer p_cmd, RenderObject* p_first, uint32_t p_count) {
    glm::mat4 view = glm::lookAt(m_camera.position, m_camera.position + m_camera.forward, glm::vec3{0.0f, 1.0f, 0.0f});
	//camera projection
	glm::mat4 projection = glm::perspective(
            glm::radians(90.f),
            static_cast<float>(m_windowExtent.width) / static_cast<float>(m_windowExtent.height),
            0.1f, 200.0f);
	projection[1][1] *= -1;

	//fill a GPU camera data struct
	GPUCameraData camData{};
	camData.proj = projection;
	camData.view = view;
	camData.viewproj = projection * view;
    camData.cameraPosition = m_camera.position;

	//and copy it to the buffer
	void* data;
	vmaMapMemory(m_allocator, GetCurrentFrame().cameraBuffer.m_allocation, &data);

	memcpy(data, &camData, sizeof(GPUCameraData));

	vmaUnmapMemory(m_allocator, GetCurrentFrame().cameraBuffer.m_allocation);

    float framed = (static_cast<float>(m_frameNumber) / 120.f);

	m_sceneParameters.ambientColor = { sin(framed),0,cos(framed),1 };

	char* sceneData;
	vmaMapMemory(m_allocator, m_sceneParameterBuffer.m_allocation , (void**)&sceneData);

	int frameIndex = m_frameNumber % FRAME_AMOUNT;

	sceneData += PadUniformBufferSize(sizeof(GPUSceneData)) * frameIndex;

	memcpy(sceneData, &m_sceneParameters, sizeof(GPUSceneData));

	vmaUnmapMemory(m_allocator, m_sceneParameterBuffer.m_allocation);

    // fill model matrix SSBO
    void* objectData;
    vmaMapMemory(m_allocator, GetCurrentFrame().objectBuffer.m_allocation, &objectData);

    auto* objectSSBO = (GPUObjectData*)objectData;
    for (int i = 0; i < p_count+1; i++)
    {
        if (i == 0) {
            objectSSBO[i].modelMatrix = glm::mat4{1.0f};
            continue;
        }

        RenderObject& object = p_first[i-1];
        glm::mat4 l_transform{1.0f};
        l_transform = glm::translate(l_transform, object.position);
        l_transform = glm::scale(l_transform, object.scale);
        objectSSBO[i].modelMatrix = l_transform;
    }

    vmaUnmapMemory(m_allocator, GetCurrentFrame().objectBuffer.m_allocation);

    // --- DRAW TERRAIN ---
	Mesh* lastMesh = nullptr;
	Material* lastMaterial = nullptr;
    DrawRenderObject(p_cmd, m_terrain, lastMesh, lastMaterial, 0);

    // --- DRAW OBJECTS ---
	for (int i = 0; i < p_count; i++)
	{
		RenderObject& object = p_first[i];

        DrawRenderObject(p_cmd, object, lastMesh, lastMaterial, i+1);
	}
}

void VulkanEngine::DrawRenderObject(VkCommandBuffer p_cmd, RenderObject& p_object, Mesh* p_lastMesh, Material* p_lastMaterial, uint32_t p_index) {
    int l_frameIndex = m_frameNumber % FRAME_AMOUNT;

    //only bind the pipeline if it doesn't match with the already bound one
    if (p_object.material != p_lastMaterial) {
        vkCmdBindPipeline(p_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p_object.material->pipeline);
        p_lastMaterial = p_object.material;

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float) m_windowExtent.width;
        viewport.height = (float) m_windowExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(p_cmd, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = m_windowExtent;
        vkCmdSetScissor(p_cmd, 0, 1, &scissor);

        uint32_t l_uniformOffset = PadUniformBufferSize(sizeof(GPUSceneData)) * l_frameIndex;
        vkCmdBindDescriptorSets(
                p_cmd,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                p_object.material->pipelineLayout, 0, 1,
                &GetCurrentFrame().globalDescriptor, 1, &l_uniformOffset);

        vkCmdBindDescriptorSets(
                p_cmd,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                p_object.material->pipelineLayout, 1, 1,
                &GetCurrentFrame().objectDescriptor, 0, nullptr);

        if (p_object.material->textureSet != VK_NULL_HANDLE) {
            //texture descriptor
            vkCmdBindDescriptorSets(p_cmd,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    p_object.material->pipelineLayout, 2, 1,
                                    &p_object.material->textureSet, 0, nullptr);
        }

        if (p_object.material->heightmapSet != VK_NULL_HANDLE) {
            //texture descriptor
            vkCmdBindDescriptorSets(p_cmd,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    p_object.material->pipelineLayout, 2, 1,
                                    &p_object.material->heightmapSet, 0, nullptr);
        }
    }

    //only bind the mesh if it's a different one from last bind
    if (p_object.mesh != p_lastMesh) {
        //bind the mesh vertex buffer with offset 0
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(p_cmd, 0, 1, &p_object.mesh->m_vertexBuffer.m_buffer, &offset);
        p_lastMesh = p_object.mesh;
    }
    //we can now draw
    vkCmdDraw(p_cmd, p_object.mesh->m_vertices.size(), 1, 0, p_index);
}

void VulkanEngine::ProcessInputs(float p_deltaTime) {
    static constexpr float DEAD_ZONE = 0.5f;
    static glm::vec<2, double> s_mousePos = {0,0};
    static bool s_firstFrame = true;

    glm::vec<2, double> l_lastMousePos{0,0};
    {
        if (!s_firstFrame) {
            l_lastMousePos = s_mousePos;
            glfwGetCursorPos(m_window, &s_mousePos[0], &s_mousePos[1]);
        } else {
            s_firstFrame = false;
            glfwGetCursorPos(m_window, &s_mousePos[0], &s_mousePos[1]);
            l_lastMousePos = s_mousePos;
        }
    }

    if (glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && glfwGetKey(m_window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
        bool l_cameraRotDirty = false;
        if (abs(l_lastMousePos.x - s_mousePos.x) > DEAD_ZONE) {
            auto l_newRotY = static_cast<float>(m_camera.rotation.y + (s_mousePos.x - l_lastMousePos.x));
            m_camera.rotation.y = l_newRotY;
            l_cameraRotDirty = true;
        }

        if (abs(l_lastMousePos.y - s_mousePos.y) > DEAD_ZONE) {
            auto l_newRotX = static_cast<float>(m_camera.rotation.x + (l_lastMousePos.y - s_mousePos.y));
            l_newRotX = glm::clamp(l_newRotX, -89.0f, 89.0f);
            m_camera.rotation.x = l_newRotX;
            l_cameraRotDirty = true;
        }

        if (l_cameraRotDirty) {
            m_camera.forward = glm::normalize(glm::vec3{
                    cos(glm::radians(m_camera.rotation.x)) * cos(glm::radians(m_camera.rotation.y)),
                    sin(glm::radians(m_camera.rotation.x)),
                    cos(glm::radians(m_camera.rotation.x)) * sin(glm::radians(m_camera.rotation.y))
            });
            m_camera.right = glm::normalize(glm::cross(m_camera.forward, {0.0f, 1.0f, 0.0f}));
        }
    }

    if (glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS)
        m_camera.position += m_camera.forward * (m_camera.speed * p_deltaTime);
    if (glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS)
        m_camera.position -= m_camera.forward * (m_camera.speed * p_deltaTime);
    if (glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS)
        m_camera.position -= m_camera.right * (m_camera.speed * p_deltaTime);
    if (glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS)
        m_camera.position += m_camera.right * (m_camera.speed * p_deltaTime);
    if (glfwGetKey(m_window, GLFW_KEY_Q) == GLFW_PRESS)
        m_camera.position.y -= (m_camera.speed * p_deltaTime);
    if (glfwGetKey(m_window, GLFW_KEY_E) == GLFW_PRESS)
        m_camera.position.y += (m_camera.speed * p_deltaTime);
}

void VulkanEngine::OnKeyPressed(int p_key, int p_scancode, int p_action, int p_mods) {
    if (p_key == GLFW_KEY_ESCAPE && p_action == GLFW_PRESS)
        glfwSetWindowShouldClose(m_window, true);

    if (p_key == GLFW_KEY_Z && p_action == GLFW_PRESS) {
        m_drawTerrainWire = !m_drawTerrainWire;
        m_terrain.material->pipeline = m_drawTerrainWire ? m_terrainWiredPipeline : m_terrainPipeline;
    }
}

bool VulkanEngine::LoadShaderModule(const std::filesystem::path &p_path, VkShaderModule *p_outModule) {
    std::ifstream l_file(p_path, std::ios::ate | std::ios::binary);

    if (!l_file.is_open()) {
        return false;
    }

    size_t l_fileSize = (size_t)l_file.tellg();
    std::vector<uint32_t> l_buffer(l_fileSize / sizeof(uint32_t));
    l_file.seekg(0);

    l_file.read((char*)l_buffer.data(), l_fileSize);

    l_file.close();

    VkShaderModuleCreateInfo l_createInfo{};
    l_createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    l_createInfo.pNext = nullptr;

    l_createInfo.codeSize = l_buffer.size() * sizeof(uint32_t);
    l_createInfo.pCode = l_buffer.data();

    //check that the creation goes well.
    VkShaderModule l_shaderModule;
    if (vkCreateShaderModule(m_device, &l_createInfo, nullptr, &l_shaderModule) != VK_SUCCESS) {
        return false;
    }
    *p_outModule = l_shaderModule;
    return true;
}

FrameData &VulkanEngine::GetCurrentFrame() {
    return m_frames[m_frameNumber % FRAME_AMOUNT];
}

void VulkanEngine::FramebufferSizeCallback(GLFWwindow* p_window, int p_width, int p_height) {
    auto l_engine = reinterpret_cast<VulkanEngine*>(glfwGetWindowUserPointer(p_window));
    auto l_width = static_cast<uint32_t>(p_width), l_height = static_cast<uint32_t>(p_height);
    if (l_width != l_engine->m_windowExtent.width || l_height != l_engine->m_windowExtent.height) {
        l_engine->m_framebufferResized = true;
    }

    l_engine->m_desiredWindowExtent.width = l_width;
    l_engine->m_desiredWindowExtent.height = l_height;
}

VkPipeline PipelineBuilder::BuildPipeline(VkDevice device, VkRenderPass pass) {
    //make viewport state from our stored viewport and scissor.
    //at the moment we won't support multiple viewports or scissors
    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.pNext = nullptr;

    viewportState.viewportCount = 1;
    // viewportState.pViewports = &m_viewport;
    viewportState.scissorCount = 1;
    // viewportState.pScissors = &m_scissor;

    //setup dummy color blending. We aren't using transparent objects yet
    //the blending is just "no blend", but we do write to the color attachment
    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.pNext = nullptr;

    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &m_colorBlendAttachment;

    //build the actual pipeline
	//we now use all of the info structs we have been writing into into this one to create the pipeline
	VkGraphicsPipelineCreateInfo pipelineInfo = {};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.pNext = nullptr;

    std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();


	pipelineInfo.stageCount = m_shaderStages.size();
	pipelineInfo.pStages = m_shaderStages.data();
	pipelineInfo.pVertexInputState = &m_vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &m_inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &m_rasterizer;
	pipelineInfo.pMultisampleState = &m_multisampling;
	pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
	pipelineInfo.layout = m_pipelineLayout;
	pipelineInfo.renderPass = pass;
	pipelineInfo.subpass = 0;
	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.pDepthStencilState = &m_depthStencil;

    if (m_inputAssembly.topology & VK_PRIMITIVE_TOPOLOGY_PATCH_LIST) {
        pipelineInfo.pTessellationState = &m_tessellationState;
    } else {
        pipelineInfo.pTessellationState = VK_NULL_HANDLE;
    }

	//it's easy to error out on create graphics pipeline, so we handle it a bit better than the common VK_CHECK case
	VkPipeline newPipeline;
	if (vkCreateGraphicsPipelines(
		device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS) {
		std::cout << "failed to create pipeline\n";
		return VK_NULL_HANDLE; // failed to create graphics pipeline
	}
	else
	{
		return newPipeline;
	}
}
