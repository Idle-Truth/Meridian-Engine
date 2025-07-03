//
// Created by TreyH on 7/3/2025.
//

#include "Mesh.h"
#include <stdexcept>
#include <cstring>

static void createBuffer(VkDevice device, VkPhysicalDevice physicalDevice,
                         VkDeviceSize size, VkBufferUsageFlags usage,
                         VkMemoryPropertyFlags properties,
                         VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memRequirements.memoryTypeBits & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            allocInfo.memoryTypeIndex = i;
            break;
        }
    }

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

Mesh Mesh::createCube(VkDevice device, VkPhysicalDevice physicalDevice,
                      VkCommandPool commandPool, VkQueue graphicsQueue) {
    Mesh mesh{};
    mesh.modelMatrix = glm::mat4(1.0f); // identity

    const std::vector<Vertex> vertices = {
        {{-0.5f, -0.5f,  0.5f}, {1, 0, 0}, {0, 0, 1}},
        {{ 0.5f, -0.5f,  0.5f}, {0, 1, 0}, {0, 0, 1}},
        {{ 0.5f,  0.5f,  0.5f}, {0, 0, 1}, {0, 0, 1}},
        {{-0.5f,  0.5f,  0.5f}, {1, 1, 0}, {0, 0, 1}},
        {{-0.5f, -0.5f, -0.5f}, {1, 0, 1}, {0, 0, -1}},
        {{ 0.5f, -0.5f, -0.5f}, {0, 1, 1}, {0, 0, -1}},
        {{ 0.5f,  0.5f, -0.5f}, {0.5f, 0.5f, 0.5f}, {0, 0, -1}},
        {{-0.5f,  0.5f, -0.5f}, {1, 1, 1}, {0, 0, -1}},
    };

    const std::vector<uint16_t> indices = {
        0, 1, 2, 2, 3, 0,  // Front
        4, 5, 6, 6, 7, 4,  // Back
        1, 5, 6, 6, 2, 1,  // Right
        4, 0, 3, 3, 7, 4,  // Left
        3, 2, 6, 6, 7, 3,  // Top
        4, 5, 1, 1, 0, 4   // Bottom
    };

    VkDeviceSize vbSize = sizeof(vertices[0]) * vertices.size();
    VkDeviceSize ibSize = sizeof(indices[0]) * indices.size();

    createBuffer(device, physicalDevice, vbSize,
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 mesh.vertexBuffer, mesh.vertexBufferMemory);

    void* data;
    vkMapMemory(device, mesh.vertexBufferMemory, 0, vbSize, 0, &data);
    memcpy(data, vertices.data(), (size_t)vbSize);
    vkUnmapMemory(device, mesh.vertexBufferMemory);

    createBuffer(device, physicalDevice, ibSize,
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 mesh.indexBuffer, mesh.indexBufferMemory);

    vkMapMemory(device, mesh.indexBufferMemory, 0, ibSize, 0, &data);
    memcpy(data, indices.data(), (size_t)ibSize);
    vkUnmapMemory(device, mesh.indexBufferMemory);

    mesh.indexCount = static_cast<uint32_t>(indices.size());

    return mesh;
}
