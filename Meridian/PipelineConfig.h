//
// Created by TreyH on 7/2/2025.
//

#ifndef PIPELINECONFIG_H
#define PIPELINECONFIG_H

#include <vulkan/vulkan.h>

struct PipelineConfig {
    VkPipelineViewportStateCreateInfo viewportInfo{};
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    VkPipelineMultisampleStateCreateInfo multisampling{};
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    VkViewport viewport{};
    VkRect2D scissor{};
    VkPipelineLayout pipelineLayout = nullptr;
    VkRenderPass renderPass = nullptr;

    static PipelineConfig defaultConfig(VkExtent2D extent);
};

#endif //PIPELINECONFIG_H
