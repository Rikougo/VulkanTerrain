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
#include <format>

#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

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

struct FrameData {
    VkSemaphore presentSemaphore, renderSemaphore;
    VkFence renderFence;

    VkCommandPool commandPool;
    VkCommandBuffer mainCommandBuffer;

    AllocatedBuffer cameraBuffer;
    AllocatedBuffer objectBuffer;

	VkDescriptorSet globalDescriptor;
    VkDescriptorSet objectDescriptor;
};

struct Material {
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
};

struct RenderObject {
    Mesh* mesh;
    Material* material;
    glm::vec3 position{0.0f};
    glm::vec3 rotation{0.0f};
    glm::vec3 scale{1.0f};
    glm::mat4 transformMatrix;

    std::string meshName;
    std::string materialName;
};

struct GPUCameraData{
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 viewproj;
};

struct GPUSceneData {
    glm::vec4 ambientColor;
	glm::vec4 sunlightDirection; //w for sun power
	glm::vec4 sunlightColor;
};

struct GPUObjectData {
    glm::mat4 modelMatrix;
};

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

struct UploadContext {
    VkFence m_uploadFence;
	VkCommandPool m_commandPool;
	VkCommandBuffer m_commandBuffer;
};

class VulkanEngine {
private:
    bool m_initialized = false;
    int m_frameNumber = 0;
    VkExtent2D m_windowExtent{1280, 720};
    VkExtent2D m_desiredWindowExtent{1280, 720};

    // --- CORE OBJECTS ---
    GLFWwindow* m_window = nullptr;
    DeletionQueue m_mainDeletionQueue;
    VkInstance m_instance;
    VkDebugUtilsMessengerEXT m_debugMessenger;
    VkPhysicalDevice m_physicalDevice;
    VkPhysicalDeviceProperties m_physicalProperties;
    VkDevice m_device;
    VkSurfaceKHR m_surface;
    VkQueue m_graphicsQueue;
    uint32_t m_graphicsQueueFamily;
    VmaAllocator m_allocator;

    ImGui_ImplVulkanH_Window m_imguiVulkanWindow;

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
    bool m_framebufferResized = false;

    VkDescriptorPool m_descriptorPool;
    VkDescriptorSetLayout m_globalSetLayout;
    VkDescriptorSetLayout m_objectSetLayout;

    // --- SCENE STUFF ---
    std::vector<RenderObject> m_renderables;
    std::unordered_map<std::string, Material> m_materials;
    std::unordered_map<std::string, Mesh> m_mesh;

    UploadContext m_uploadContext;

    GPUSceneData m_sceneParameters;
    AllocatedBuffer m_sceneParameterBuffer;

private:
    bool* m_showDemoWindow;
    bool* m_showInspectorWindow;
public:
    // VulkanEngine() = default;

    // public lifecycle
    void Init();
    void Run();
    void Cleanup();

private:
    AllocatedBuffer CreateBuffer(size_t p_allocSize, VkBufferUsageFlags p_usage, VmaMemoryUsage p_memoryUsage);
    size_t PadUniformBufferSize(size_t p_originalSize) const;

    Mesh* GetMesh(const std::string &p_name);
    Material* GetMaterial(const std::string &p_name);
    Material* CreateMaterial(VkPipeline p_pipeline, VkPipelineLayout p_layout, const std::string &p_name);

    void LoadMeshes();
    void UploadMesh(Mesh& p_mesh);

    void ImmediateSubmit(std::function<void(VkCommandBuffer p_cmd)>&& function);

    void InitScene();

    void InitVulkan();
    void InitSwapchain();
    void InitCommands();
    void InitDefaultRenderpass();
    void InitFramebuffers();
    void InitSyncStructures();
    void InitDescriptors();
    void InitPipelines();
    void InitIMGUI();

    void CleanupSwapchain();
    void RecreateSwapchain();

    FrameData& GetCurrentFrame();

    bool LoadShaderModule(const std::filesystem::path &p_path, VkShaderModule *p_outModule);

    void DrawUI();
    void Draw();
    void DrawObjects(VkCommandBuffer p_cmd, RenderObject* p_first, uint32_t count);

    static void FramebufferSizeCallback(GLFWwindow* p_window, int p_width, int p_height);
};

#endif //VULKAN_TERRAIN_VULKAN_ENGINE_HPP
