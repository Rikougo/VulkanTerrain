//
// Created by Profanateur on 12/4/2022.
//

#include "VulkanEngine.hpp"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

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
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    m_window = glfwCreateWindow(
            static_cast<int>(m_windowExtent.width),
            static_cast<int>(m_windowExtent.height),
            "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(m_window, this);

    InitVulkan();
    InitSwapchain();
    InitCommands();
    InitDefaultRenderpass();
    InitFramebuffers();
    InitSyncStructures();
    // InitDescriptors();
    InitPipelines();

    LoadMeshes();

    InitScene();

    m_initialized = true;
}

void VulkanEngine::Run() {
    while (!glfwWindowShouldClose(m_window)) {
        glfwPollEvents();
        // ProcessEvents();
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

	AllocatedBuffer l_newBuffer;

	//allocate the buffer
	VK_CHECK(vmaCreateBuffer(m_allocator, &l_bufferInfo, &l_vmaAllocInfo,
                             &l_newBuffer.m_buffer,
                             &l_newBuffer.m_allocation,
                             nullptr));

	return l_newBuffer;
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

void VulkanEngine::LoadMeshes() {
    Mesh l_suzanne{};
    //make the array 3 vertices long
    l_suzanne.LoadFromObj("./assets/suzanne.obj");
	//we don't care about the vertex normals

	UploadMesh(l_suzanne);

    m_mesh["suzanne"] = l_suzanne;
}

void VulkanEngine::UploadMesh(Mesh &p_mesh) {
    //allocate vertex buffer
	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	//this is the total size, in bytes, of the buffer we are allocating
	bufferInfo.size = p_mesh.m_vertices.size() * sizeof(Vertex);
	//this buffer is going to be used as a Vertex Buffer
	bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;


	//let the VMA library know that this data should be writeable by CPU, but also readable by GPU
	VmaAllocationCreateInfo vmaallocInfo = {};
	vmaallocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

	//allocate the buffer
	VK_CHECK(vmaCreateBuffer(m_allocator, &bufferInfo, &vmaallocInfo,
		&p_mesh.m_vertexBuffer.m_buffer,
		&p_mesh.m_vertexBuffer.m_allocation,
		nullptr));

	//add the destruction of triangle mesh buffer to the deletion queue
	m_mainDeletionQueue.PushFunction([=]() {
        vmaDestroyBuffer(m_allocator, p_mesh.m_vertexBuffer.m_buffer, p_mesh.m_vertexBuffer.m_allocation);
    });

    void* data;
	vmaMapMemory(m_allocator, p_mesh.m_vertexBuffer.m_allocation, &data);

	memcpy(data, p_mesh.m_vertices.data(), p_mesh.m_vertices.size() * sizeof(Vertex));

	vmaUnmapMemory(m_allocator, p_mesh.m_vertexBuffer.m_allocation);
}

void VulkanEngine::InitScene() {
    RenderObject monkey{};
	monkey.mesh = GetMesh("suzanne");
	monkey.material = GetMaterial("defaultmesh");
	monkey.transformMatrix = glm::mat4{ 1.0f };

	m_renderables.push_back(monkey);

	for (int x = -20; x <= 20; x++) {
		for (int y = -20; y <= 20; y++) {

			RenderObject l_smallMonkey{};
            l_smallMonkey.mesh = GetMesh("suzanne");
            l_smallMonkey.material = GetMaterial("defaultmesh");
			glm::mat4 translation = glm::translate(glm::mat4{ 1.0 }, glm::vec3(x, 0, y));
			glm::mat4 scale = glm::scale(glm::mat4{ 1.0 }, glm::vec3(0.2, 0.2, 0.2));
			l_smallMonkey.transformMatrix = translation * scale;

			m_renderables.push_back(l_smallMonkey);
		}
	}
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
		.select()
		.value();

	//create the final Vulkan device
	vkb::DeviceBuilder l_deviceBuilder{l_physicalDevice };

	vkb::Device l_vkbDevice = l_deviceBuilder.build().value();

	// Get the VkDevice handle used in the rest of a Vulkan application
	m_device = l_vkbDevice.device;
	m_physicalDevice = l_physicalDevice.physical_device;
    // m_physicalProperties = l_vkbDevice.physical_device.properties;

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
	VkImageCreateInfo dimg_info = VulkanInit::ImageCreateInfo(m_depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depthImageExtent);

	//for the depth image, we want to allocate it from GPU local memory
	VmaAllocationCreateInfo dimg_allocinfo = {};
	dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	//allocate and create the image
	vmaCreateImage(
            m_allocator,
            &dimg_info,
            &dimg_allocinfo,
            &m_depthImage.m_image,
            &m_depthImage.m_allocation,
            nullptr);

	//build an image-view for the depth image to use for rendering
	VkImageViewCreateInfo dview_info = VulkanInit::ImageViewCreateInfo(m_depthFormat, m_depthImage.m_image, VK_IMAGE_ASPECT_DEPTH_BIT);

	VK_CHECK(vkCreateImageView(m_device, &dview_info, nullptr, &m_depthImageView));

    m_mainDeletionQueue.PushFunction([=]() {
        vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
        vkDestroyImageView(m_device, m_depthImageView, nullptr);
        vmaDestroyImage(m_allocator, m_depthImage.m_image, m_depthImage.m_allocation);
    });
}

void VulkanEngine::InitCommands() {
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
	VkFramebufferCreateInfo fb_info = {};
	fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	fb_info.pNext = nullptr;

	fb_info.renderPass = m_renderPass;
	fb_info.attachmentCount = 1;
	fb_info.width = m_windowExtent.width;
	fb_info.height = m_windowExtent.height;
	fb_info.layers = 1;

	//grab how many images we have in the swapchain
	const size_t l_swapchainImageCount = m_swapchainImages.size();
	m_framebuffers = std::vector<VkFramebuffer>(l_swapchainImageCount);

	//create framebuffers for each of the swapchain image views
	for (size_t i = 0; i < l_swapchainImageCount; i++) {
        VkImageView l_attachments[2];
        l_attachments[0] = m_swapchainImageViews[i];
        l_attachments[1] = m_depthImageView;

        fb_info.pAttachments = l_attachments;
        fb_info.attachmentCount = 2;
		VK_CHECK(vkCreateFramebuffer(m_device, &fb_info, nullptr, &m_framebuffers[i]));

        m_mainDeletionQueue.PushFunction([=]() {
			vkDestroyFramebuffer(m_device, m_framebuffers[i], nullptr);
			vkDestroyImageView(m_device, m_swapchainImageViews[i], nullptr);
    	});
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

void VulkanEngine::InitPipelines() {
    VkShaderModule triangleFragShader;
	if (!LoadShaderModule("./shaders/bin/main_frag.spv", &triangleFragShader))
	{
		std::cout << "Error when building the triangle fragment shader module" << std::endl;
	}
	else {
		std::cout << "Triangle fragment shader successfully loaded" << std::endl;
	}

	VkShaderModule meshVertShader;
	if (!LoadShaderModule("./shaders/bin/mesh_vert.spv", &meshVertShader))
	{
		std::cout << "Error when building the triangle vertex shader module" << std::endl;
	}
	else {
		std::cout << "Red Triangle vertex shader successfully loaded" << std::endl;
	}

    VkPipelineLayoutCreateInfo l_meshPipelineLayoutInfo = VulkanInit::PipelineLayoutCreateInfo();

    VkPushConstantRange l_pushConstant;
    l_pushConstant.offset = 0;
    l_pushConstant.size = sizeof(MeshPushConstants);
    l_pushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    l_meshPipelineLayoutInfo.pPushConstantRanges = &l_pushConstant;
    l_meshPipelineLayoutInfo.pushConstantRangeCount = 1;

    VK_CHECK(vkCreatePipelineLayout(m_device, &l_meshPipelineLayoutInfo, nullptr, &m_meshPipelineLayout));

    //build the stage-create-info for both vertex and fragment stages. This lets the pipeline know the shader modules per stage
	PipelineBuilder pipelineBuilder;

	pipelineBuilder.m_shaderStages.push_back(VulkanInit::PipelineShaderStageCreateInfo(
            VK_SHADER_STAGE_FRAGMENT_BIT, triangleFragShader));

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

	//connect the pipeline builder vertex input info to the one we get from Vertex
	pipelineBuilder.m_vertexInputInfo.pVertexAttributeDescriptions = vertexDescription.attributes.data();
	pipelineBuilder.m_vertexInputInfo.vertexAttributeDescriptionCount = vertexDescription.attributes.size();

	pipelineBuilder.m_vertexInputInfo.pVertexBindingDescriptions = vertexDescription.bindings.data();
	pipelineBuilder.m_vertexInputInfo.vertexBindingDescriptionCount = vertexDescription.bindings.size();

	//clear the shader stages for the builder
	pipelineBuilder.m_shaderStages.clear();

	//compile mesh vertex shader

	//add the other shaders
	pipelineBuilder.m_shaderStages.push_back(
            VulkanInit::PipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, meshVertShader));

	//make sure that triangleFragShader is holding the compiled colored_triangle.frag
	pipelineBuilder.m_shaderStages.push_back(
            VulkanInit::PipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, triangleFragShader));
    pipelineBuilder.m_pipelineLayout = m_meshPipelineLayout;

	//build the mesh triangle pipeline
	m_meshPipeline = pipelineBuilder.BuildPipeline(m_device, m_renderPass);

    CreateMaterial(m_meshPipeline, m_meshPipelineLayout, "defaultmesh");

    m_mainDeletionQueue.PushFunction([=]() {
        vkDestroyShaderModule(m_device, triangleFragShader, nullptr);
        vkDestroyShaderModule(m_device, meshVertShader, nullptr);
        vkDestroyPipeline(m_device, m_meshPipeline, nullptr);

		vkDestroyPipelineLayout(m_device, m_meshPipelineLayout, nullptr);
    });
}

void VulkanEngine::Draw() {
    VK_CHECK(vkWaitForFences(m_device, 1, &GetCurrentFrame().renderFence, true, 1000000000));
	VK_CHECK(vkResetFences(m_device, 1, &GetCurrentFrame().renderFence));

    //request image from the swapchain, one second timeout
	uint32_t swapchainImageIndex;
	VK_CHECK(vkAcquireNextImageKHR(m_device, m_swapchain, 1000000000, GetCurrentFrame().presentSemaphore, nullptr, &swapchainImageIndex));

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
	float flash = abs(sin(m_frameNumber / 120.f));
	clearValue.color = { { 0.0f, 0.0f, flash, 1.0f } };

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

	VK_CHECK(vkQueuePresentKHR(m_graphicsQueue, &presentInfo));

	//increase the number of frames drawn
	m_frameNumber++;
}

void VulkanEngine::DrawObjects(VkCommandBuffer p_cmd, RenderObject* p_first, uint32_t p_count) {
   //camera view
	glm::vec3 camPos = { 0.f,-6.f,-10.f };

	glm::mat4 view = glm::translate(glm::mat4(1.f), camPos);
	//camera projection
	glm::mat4 projection = glm::perspective(glm::radians(70.f), 1700.f / 900.f, 0.1f, 200.0f);
	projection[1][1] *= -1;

	//fill a GPU camera data struct
	GPUCameraData camData;
	camData.proj = projection;
	camData.view = view;
	camData.viewproj = projection * view;

	//and copy it to the buffer
	/*void* data;
	vmaMapMemory(m_allocator, GetCurrentFrame().cameraBuffer.m_allocation, &data);

	memcpy(data, &camData, sizeof(GPUCameraData));

	vmaUnmapMemory(m_allocator, GetCurrentFrame().cameraBuffer.m_allocation);*/

	Mesh* lastMesh = nullptr;
	Material* lastMaterial = nullptr;
	for (int i = 0; i < p_count; i++)
	{
		RenderObject& object = p_first[i];

		//only bind the pipeline if it doesn't match with the already bound one
		if (object.material != lastMaterial) {
			vkCmdBindPipeline(p_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipeline);
			lastMaterial = object.material;
		}

		glm::mat4 model = object.transformMatrix;
		//final render matrix, that we are calculating on the cpu
		glm::mat4 mesh_matrix = projection * view * model;

		MeshPushConstants constants{};
		constants.renderMatrix = mesh_matrix;

		//upload the mesh to the GPU via push constants
		vkCmdPushConstants(
                p_cmd,
                object.material->pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(MeshPushConstants), &constants);

		//only bind the mesh if it's a different one from last bind
		if (object.mesh != lastMesh) {
			//bind the mesh vertex buffer with offset 0
			VkDeviceSize offset = 0;
			vkCmdBindVertexBuffers(p_cmd, 0, 1, &object.mesh->m_vertexBuffer.m_buffer, &offset);
			lastMesh = object.mesh;
		}
		//we can now draw
		vkCmdDraw(p_cmd, object.mesh->m_vertices.size(), 1, 0, 0);
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

VkPipeline PipelineBuilder::BuildPipeline(VkDevice device, VkRenderPass pass) {
    //make viewport state from our stored viewport and scissor.
    //at the moment we won't support multiple viewports or scissors
    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.pNext = nullptr;

    viewportState.viewportCount = 1;
    viewportState.pViewports = &m_viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &m_scissor;

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

	pipelineInfo.stageCount = m_shaderStages.size();
	pipelineInfo.pStages = m_shaderStages.data();
	pipelineInfo.pVertexInputState = &m_vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &m_inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &m_rasterizer;
	pipelineInfo.pMultisampleState = &m_multisampling;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.layout = m_pipelineLayout;
	pipelineInfo.renderPass = pass;
	pipelineInfo.subpass = 0;
	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.pDepthStencilState = &m_depthStencil;

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
