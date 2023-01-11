//
// Created by sakeiru on 1/9/2023.
//

#ifndef VULKAN_TERRAIN_VULKAN_ENGINE_STRUCT_H
#define VULKAN_TERRAIN_VULKAN_ENGINE_STRUCT_H

#include <deque>
#include <functional>
#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <VulkanTypes.hpp>
#include <VulkanMesh.hpp>

enum TerrainDisplay {
    LIGHTNING = 0,
    NORMAL    = 1,
    HEIGHT    = 2
};

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
    VkPipeline pipeline{VK_NULL_HANDLE};
    VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
};

struct Camera {
    glm::vec3 position;
    glm::vec3 rotation;

    glm::mat4 view;
    glm::mat4 projection;
    glm::mat4 viewproj;

    glm::vec3 forward;
    glm::vec3 right;
    float speed{5.0f};

    bool dirty{true};

    bool Update(VkExtent2D p_extent) {
        if (!dirty) return false;

        view = glm::lookAt(position, position + forward, glm::vec3{0.0f, 1.0f, 0.0f});
        //camera projection
        projection = glm::perspective(
                glm::radians(90.f),
                static_cast<float>(p_extent.width) / static_cast<float>(p_extent.height),
                0.1f, 200.0f);
        projection[1][1] *= -1;

        viewproj = projection * view;

        forward = glm::normalize(glm::vec3{
                cos(glm::radians(rotation.x)) * cos(glm::radians(rotation.y)),
                sin(glm::radians(rotation.x)),
                cos(glm::radians(rotation.x)) * sin(glm::radians(rotation.y))
        });
        right = glm::normalize(glm::cross(forward, {0.0f, 1.0f, 0.0f}));

        return true;
    }
};

struct RenderObject {
    Mesh* mesh;
    Material* material;
    bool active{true};
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

struct RaycastResult {
    bool found{false};
    float t{std::numeric_limits<float>::max()};
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

// GPU OBJECTS
struct GPUCameraData{
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 viewproj;
    glm::vec3 cameraPosition;
};

struct GPUSceneData {
    glm::vec4 ambientColor;
	glm::vec4 sunlightDirection;        // w for sun power
	glm::vec4 sunlightColor;            // sunlight color
    float terrainSubdivision{1.0f};     // max value of terrain subdivision
    float displacementFactor{1.0f};     // height for normalized value = 1
    float minDistance{5.0f};            // min distance before lowering terrain subdivision
    float maxDistance{10.0f};           // max distance of terrain subdivision
    glm::vec3 clickedPoint{0.0f}; // world space clicked point position
    TerrainDisplay terrainDisplay{HEIGHT};
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

    VkPipeline BuildPipeline(VkDevice p_device, VkRenderPass p_pass) {
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
        pipelineInfo.renderPass = p_pass;
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
            p_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS) {
            std::cout << "failed to create pipeline\n";
            return VK_NULL_HANDLE; // failed to create graphics pipeline
        }
        else
        {
            return newPipeline;
        }
    }
};

struct UploadContext {
    VkFence m_uploadFence;
	VkCommandPool m_commandPool;
	VkCommandBuffer m_commandBuffer;
};

#endif //VULKAN_TERRAIN_VULKAN_ENGINE_STRUCT_H
