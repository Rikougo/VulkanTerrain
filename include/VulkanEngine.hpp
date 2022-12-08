//
// Created by Profanateur on 12/4/2022.
//

#ifndef VULKAN_TERRAIN_VULKAN_ENGINE_HPP
#define VULKAN_TERRAIN_VULKAN_ENGINE_HPP

#include <iostream>
#include <filesystem>
#include <fstream>
#include <functional>
#include <deque>

#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <VulkanTypes.hpp>
#include <VulkanInitializer.hpp>
#include <VulkanMesh.hpp>

#include <VkBootstrap.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

constexpr uint8_t FRAME_AMOUNT = 2;

struct DeletionQueue {
    std::deque<std::function<void()>> deletors;

    void PushFunction(std::function<void()>&& p_function) {
        deletors.push_back(p_function);
    }

    void Flush() {
        for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
            (*it)();
        }

        deletors.clear();
    }
};

struct MeshPushConstants {
	glm::vec4 data;
	glm::mat4 renderMatrix;
};

struct FrameData {
    VkSemaphore presentSemaphore, renderSemaphore;
    VkFence renderFence;

    VkCommandPool commandPool;
    VkCommandBuffer mainCommandBuffer;

    AllocatedBuffer cameraBuffer;
    AllocatedBuffer objectBuffer;

	VkDescriptorSet globalDescriptor;
};

struct Material {
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
};

struct RenderObject {
    Mesh* mesh;
    Material* material;
    glm::mat4 transformMatrix;
};

struct GPUCameraData{
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 viewproj;
};

/*struct GPUSceneData {
    glm::vec4 ambientColor;
	glm::vec4 sunlightDirection; //w for sun power
	glm::vec4 sunlightColor;
};

struct GPUObjectData {
    glm::mat4 modelMatrix;
};*/

class PipelineBuilder {
public:
    std::vector<VkPipelineShaderStageCreateInfo> m_shaderStages;
	VkPipelineVertexInputStateCreateInfo m_vertexInputInfo;
	VkPipelineInputAssemblyStateCreateInfo m_inputAssembly;
	VkViewport m_viewport;
	VkRect2D m_scissor;
	VkPipelineRasterizationStateCreateInfo m_rasterizer;
	VkPipelineColorBlendAttachmentState m_colorBlendAttachment;
	VkPipelineMultisampleStateCreateInfo m_multisampling;
	VkPipelineLayout m_pipelineLayout;
    VkPipelineDepthStencilStateCreateInfo m_depthStencil;

    VkPipeline BuildPipeline(VkDevice p_device, VkRenderPass p_pass);
};

class VulkanEngine {
private:
    bool m_initialized = false;
    int m_frameNumber = 0;
    VkExtent2D m_windowExtent{1280, 720};

    // --- CORE OBJECTS ---
    GLFWwindow* m_window = nullptr;
    DeletionQueue m_mainDeletionQueue;
    VkInstance m_instance;
    VkDebugUtilsMessengerEXT m_debugMessenger;
    VkPhysicalDevice m_physicalDevice;
    // VkPhysicalDeviceProperties m_physicalProperties;
    VkDevice m_device;
    VkSurfaceKHR m_surface;
    VkQueue m_graphicsQueue;
    uint32_t m_graphicsQueueFamily;
    VmaAllocator m_allocator;

    /*VkDescriptorSetLayout m_globalSetLayout;
    VkDescriptorPool m_descriptorPool;*/

    // --- RENDER OBJECTS ---
    VkSwapchainKHR m_swapchain; // from other articles
	// image format expected by the windowing system
	VkFormat m_swapchainImageFormat;
	//array of images from the swapchain
	std::vector<VkImage> m_swapchainImages;
	//array of image-views from the swapchain
	std::vector<VkImageView> m_swapchainImageViews;

    VkFormat m_depthFormat;
    VkImageView m_depthImageView;
    AllocatedImage m_depthImage;

    VkRenderPass m_renderPass;
    std::vector<VkFramebuffer> m_framebuffers;

    FrameData m_frames[FRAME_AMOUNT];

    VkPipelineLayout m_meshPipelineLayout;
    VkPipeline m_meshPipeline;

    std::vector<RenderObject> m_renderables;
    std::unordered_map<std::string, Material> m_materials;
    std::unordered_map<std::string, Mesh> m_mesh;

    /*GPUSceneData m_sceneParameters;
    AllocatedBuffer m_sceneParameterBuffer;*/
public:
    // VulkanEngine() = default;

    // public lifecycle
    void Init();
    void Run();
    void Cleanup();

private:
    AllocatedBuffer CreateBuffer(size_t p_allocSize, VkBufferUsageFlags p_usage, VmaMemoryUsage p_memoryUsage);
    // void InitDescriptors();

    Mesh* GetMesh(const std::string &p_name);
    Material* GetMaterial(const std::string &p_name);
    Material* CreateMaterial(VkPipeline p_pipeline, VkPipelineLayout p_layout, const std::string &p_name);

    void LoadMeshes();
    void UploadMesh(Mesh& p_mesh);

    void InitScene();

    void InitVulkan();
    void InitSwapchain();
    void InitCommands();
    void InitDefaultRenderpass();
    void InitFramebuffers();
    void InitSyncStructures();
    void InitPipelines();

    FrameData& GetCurrentFrame();

    bool LoadShaderModule(const std::filesystem::path &p_path, VkShaderModule *p_outModule);

    void Draw();
    void DrawObjects(VkCommandBuffer p_cmd, RenderObject* p_first, uint32_t count);

    // size_t PadUniformBufferSize(size_t p_originalSize);
};

#endif //VULKAN_TERRAIN_VULKAN_ENGINE_HPP
