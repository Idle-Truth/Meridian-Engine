//
// Created by TreyH on 7/2/2025.
//

#include "GraphicsShader.h"

GraphicsShader::GraphicsShader(VkDevice device, const std::string& vertPath, const std::string& fragPath)
    : device(device),
      vertexShader(loadShader(device, vertPath, VK_SHADER_STAGE_VERTEX_BIT)),
      fragmentShader(loadShader(device, fragPath, VK_SHADER_STAGE_FRAGMENT_BIT)) {

    stages[0] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = vertexShader.stage;
    stages[0].module = vertexShader.module;
    stages[0].pName = "main";

    stages[1] = {};
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = fragmentShader.stage;
    stages[1].module = fragmentShader.module;
    stages[1].pName = "main";
}

GraphicsShader::~GraphicsShader() {
    vkDestroyShaderModule(device, vertexShader.module, nullptr);
    vkDestroyShaderModule(device, fragmentShader.module, nullptr);
}

const VkPipelineShaderStageCreateInfo* GraphicsShader::getStages() const {
    return stages;
}


