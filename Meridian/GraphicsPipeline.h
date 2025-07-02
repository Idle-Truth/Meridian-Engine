//
// Created by TreyH on 7/2/2025.
//

#ifndef GRAPHICSPIPELINE_H
#define GRAPHICSPIPELINE_H

#include <string>
#include <vulkan/vulkan.h>
#include "GraphicsShader.h"
#include "PipelineConfig.h"

class GraphicsPipeline {
public:
    GraphicsPipeline(
        VkDevice device,
        const GraphicsShader& shader,
        const PipelineConfig& config,
        VkPipelineLayout pipelineLayout,
        VkRenderPass renderPass);

    ~GraphicsPipeline();

    VkPipeline getPipeline() const { return pipeline; }

private:
    VkDevice device;
    VkPipeline pipeline;
};



#endif //GRAPHICSPIPELINE_H
