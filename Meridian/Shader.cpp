#include "Shader.h"
#include <fstream>
#include <vector>
#include <stdexcept>

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + filename);
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

Shader loadShader(VkDevice device, const std::string& filepath, VkShaderStageFlagBits stage) {
    Shader shader{};
    shader.filepath = filepath;
    shader.stage = stage;

    auto code = readFile(filepath);

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    if (vkCreateShaderModule(device, &createInfo, nullptr, &shader.module) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module");
    }

    return shader;
}
