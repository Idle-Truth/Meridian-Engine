//
// Created by TreyH on 7/2/2025.
//

#ifndef GRAPHICSSHADER_H
#define GRAPHICSSHADER_H

#include <vulkan/vulkan.h>
#include <string>
#include "Shader.h"

class GraphicsShader {
public:
    GraphicsShader(VkDevice device, const std::string& vertPath, const std::string& fragPath);
    ~GraphicsShader();

    const VkPipelineShaderStageCreateInfo* getStages() const;

private:
    VkDevice device;
    Shader vertexShader;
    Shader fragmentShader;
    VkPipelineShaderStageCreateInfo stages[2];
};


#endif //GRAPHICSSHADER_H
