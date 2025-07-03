//
// Created by TreyH on 7/3/2025.
//

#ifndef MESH_H
#define MESH_H

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include "Vertex.h"

struct Mesh {
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;

    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;

    VkDescriptorSet descriptorSet;
    uint32_t indexCount;

    glm::mat4 modelMatrix;

    // Factory-style creator
    static Mesh createCube(VkDevice device, VkPhysicalDevice physicalDevice,
                           VkCommandPool commandPool, VkQueue graphicsQueue);
};



#endif //MESH_H
