//
// Created by TreyH on 7/2/2025.
//

#include "PipelineConfig.h"

PipelineConfig PipelineConfig::defaultConfig(VkExtent2D extent) {
    PipelineConfig config{};

    config.viewport.x = 0.0f;
    config.viewport.y = 0.0f;
    config.viewport.width = static_cast<float>(extent.width);
    config.viewport.height = static_cast<float>(extent.height);
    config.viewport.minDepth = 0.0f;
    config.viewport.maxDepth = 1.0f;

    config.scissor.offset = {0, 0};
    config.scissor.extent = extent;

    config.viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    config.viewportInfo.viewportCount = 1;
    config.viewportInfo.pViewports = &config.viewport;
    config.viewportInfo.scissorCount = 1;
    config.viewportInfo.pScissors = &config.scissor;

    config.inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    config.inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    config.inputAssembly.primitiveRestartEnable = VK_FALSE;

    config.rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    config.rasterizer.depthClampEnable = VK_FALSE;
    config.rasterizer.rasterizerDiscardEnable = VK_FALSE;
    config.rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    config.rasterizer.lineWidth = 1.0f;
    config.rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    config.rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    config.rasterizer.depthBiasEnable = VK_FALSE;

    config.multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    config.multisampling.sampleShadingEnable = VK_FALSE;
    config.multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    config.colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    config.colorBlendAttachment.blendEnable = VK_FALSE;

    config.colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    config.colorBlending.logicOpEnable = VK_FALSE;
    config.colorBlending.attachmentCount = 1;
    config.colorBlending.pAttachments = &config.colorBlendAttachment;

    config.depthStencil = {};
    config.depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    config.depthStencil.depthTestEnable = VK_TRUE;
    config.depthStencil.depthWriteEnable = VK_TRUE;
    config.depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    config.depthStencil.depthBoundsTestEnable = VK_FALSE;
    config.depthStencil.stencilTestEnable = VK_FALSE;


    return config;
}