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
#include <VulkanTextures.hpp>

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
    VkDescriptorSet textureSet{VK_NULL_HANDLE};
    VkDescriptorSet heightmapSet{VK_NULL_HANDLE};
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
};

struct Camera {
    glm::vec3 position;
    glm::vec3 rotation;
    glm::vec3 forward;
    glm::vec3 right;
    float speed = 5.0f;
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

struct Texture {
    AllocatedImage image;
    VkImageView imageView;
};

struct GPUCameraData{
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 viewproj;
    glm::vec3 cameraPosition;
};

struct GPUSceneData {
    glm::vec4 ambientColor;
	glm::vec4 sunlightDirection; //w for sun power
	glm::vec4 sunlightColor;
    float terrainSubdivision;
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
    VkPipelineTessellationStateCreateInfo m_tessellationState;

    VkPipeline BuildPipeline(VkDevice p_device, VkRenderPass p_pass);
};

struct UploadContext {
    VkFence m_uploadFence;
	VkCommandPool m_commandPool;
	VkCommandBuffer m_commandBuffer;
};

class VulkanEngine {
public:
    VmaAllocator m_allocator;
    DeletionQueue m_mainDeletionQueue;
    VkDescriptorSetLayout m_singleTextureSetLayout;
private:
    bool m_initialized = false;
    int m_frameNumber = 0;
    VkExtent2D m_windowExtent{1280, 720};
    VkExtent2D m_desiredWindowExtent{1280, 720};

    // --- CORE OBJECTS ---
    GLFWwindow* m_window = nullptr;
    VkInstance m_instance;
    VkDebugUtilsMessengerEXT m_debugMessenger;
    VkPhysicalDevice m_physicalDevice;
    VkPhysicalDeviceProperties m_physicalProperties;
    VkDevice m_device;
    VkSurfaceKHR m_surface;
    VkQueue m_graphicsQueue;
    uint32_t m_graphicsQueueFamily;

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
    VkDescriptorSetLayout m_heightmapSetLayout;

    // --- SCENE STUFF ---
    std::unordered_map<std::string, Texture> m_loadedTextures;
    std::unordered_map<std::string, Material> m_materials;
    std::unordered_map<std::string, Mesh> m_mesh;
    Camera m_camera{};

    std::vector<RenderObject> m_renderables;

    Mesh m_terrainMesh;
    Material m_terrainMaterial;
    VkPipeline m_terrainPipeline;
    VkPipeline m_terrainWiredPipeline;
    RenderObject m_terrain;
    bool m_drawTerrainWire = false;

    UploadContext m_uploadContext;

    GPUSceneData m_sceneParameters;
    AllocatedBuffer m_sceneParameterBuffer;
private:
    bool* m_showDemoWindow;
    bool* m_showInspectorWindow;
    bool* m_showHeightMapWindow;
public:
    // VulkanEngine() = default;

    // public lifecycle
    void Init();
    void Run();
    void Cleanup();


    AllocatedBuffer CreateBuffer(size_t p_allocSize, VkBufferUsageFlags p_usage, VmaMemoryUsage p_memoryUsage);
    void ImmediateSubmit(std::function<void(VkCommandBuffer p_cmd)>&& function);

    void OnKeyPressed(int p_key, int p_scancode, int p_action, int p_mods);
private:
    [[nodiscard]] size_t PadUniformBufferSize(size_t p_originalSize) const;

    Mesh* GetMesh(const std::string &p_name);
    Material* GetMaterial(const std::string &p_name);
    Material* CreateMaterial(VkPipeline p_pipeline, VkPipelineLayout p_layout, const std::string &p_name);

    void LoadImages();
    void LoadMeshes();
    void UploadMesh(Mesh& p_mesh);

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
    void DrawRenderObject(VkCommandBuffer p_cmd, RenderObject& p_object, Mesh* p_lastMesh, Material* p_lastMaterial, uint32_t p_index);
    void ProcessInputs(float p_deltaTime);

    static void FramebufferSizeCallback(GLFWwindow* p_window, int p_width, int p_height);
};

#endif //VULKAN_TERRAIN_VULKAN_ENGINE_HPP
