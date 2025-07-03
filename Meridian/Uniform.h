//
// Created by TreyH on 7/2/2025.
//

#ifndef UNIFORM_H
#define UNIFORM_H

#include <glm/glm.hpp>

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec3 lightPos;
    glm::vec3 viewPos;
};

#endif //UNIFORM_H
