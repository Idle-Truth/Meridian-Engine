//
// Created by TreyH on 7/2/2025.
//

#ifndef SHADER_H
#define SHADER_H

#include <vulkan/vulkan.h>
#include <string>
#include <vector>

struct Shader {
    VkShaderModule module;
    VkShaderStageFlagBits stage;
    std::string filepath;
};

Shader loadShader(VkDevice device, const std::string& filepath, VkShaderStageFlagBits stage);

#endif //SHADER_H
