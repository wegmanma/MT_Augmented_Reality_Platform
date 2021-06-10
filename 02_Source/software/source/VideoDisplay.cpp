#include <vector>
#include <array>
#include <chrono>
#include <vulkan/vulkan.h>

#include "VideoDisplay.hpp"
#include "VulkanHelper.hpp"
#include "FrameCapture.hpp"



void VideoDisplay::cleanupSwapChain(VkDevice device, size_t numSwapChainImages) {
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    for (size_t i = 0; i < numSwapChainImages; i++) {
        vkDestroyBuffer(device, uniformBuffers[i], nullptr);
        vkFreeMemory(device, uniformBuffersMemory[i], nullptr);

    }
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
}

void VideoDisplay::cleanup(VkDevice device) {


    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

    vkDestroyBuffer(device, indexBuffer, nullptr);
    vkFreeMemory(device, indexBufferMemory, nullptr);
    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkFreeMemory(device, vertexBufferMemory, nullptr);
}

void VideoDisplay::create(VkDevice device, VkPhysicalDevice physicalDevice, VkRenderPass renderPass, VkCommandPool commandPool, VkQueue graphicsQueue, VkExtent2D swapChainExtent, size_t numSwapChainImages, VkImageView combinedImageView, VkSampler combinedSampler) {
    createDescriptorSetLayout(device);

    
    createGraphicsPipeline(device, renderPass,swapChainExtent);
    createVertexBuffer(device,physicalDevice, commandPool, graphicsQueue);
    
    createIndexBuffer(device, physicalDevice, commandPool, graphicsQueue);
    std::cout << "IndexBuffer created" << std::endl;
    createUniformBuffers(device,physicalDevice,numSwapChainImages);
    std::cout << "UniformBuffer created" << std::endl;
    createDescriptorPool(device,numSwapChainImages);
    createDescriptorSets(device, numSwapChainImages, combinedImageView, combinedSampler);
}

void VideoDisplay::recreate(VkDevice device, VkPhysicalDevice physicalDevice, VkRenderPass renderPass, VkExtent2D swapChainExtent, size_t numSwapChainImages, VkImageView combinedImageView, VkSampler combinedSampler) {
    createGraphicsPipeline(device, renderPass, swapChainExtent);
    createUniformBuffers(device, physicalDevice, numSwapChainImages);
    createDescriptorPool(device, numSwapChainImages);
    createDescriptorSets(device, numSwapChainImages, combinedImageView, combinedSampler);
}

void VideoDisplay::update(VkDevice device, VkCommandPool commandPool, VkQueue graphicsQueue, VkExtent2D swapChainExtent, uint32_t currentImage) {
    updateUniformBuffer(device, swapChainExtent, currentImage);
}

void VideoDisplay::createDescriptorSetLayout(VkDevice device) {
    VkDescriptorSetLayoutBinding uboLayoutBinding = {};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.pImmutableSamplers = nullptr;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS;

    VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
}

void VideoDisplay::createGraphicsPipeline(VkDevice device, VkRenderPass renderPass, VkExtent2D swapChainExtent) {
    // video windows
    auto vertShaderCode = vkh::readFile("vulkanShaders/vert.spv");
    auto fragShaderCode = vkh::readFile("vulkanShaders/frag.spv");

    VkShaderModule vertShaderModule = vkh::createShaderModule(device, vertShaderCode);
    VkShaderModule fragShaderModule = vkh::createShaderModule(device, fragShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo,  fragShaderStageInfo };

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    // auto bindingDescription = Vertex::getBindingDescription();
    // auto attributeDescriptions = Vertex::getAttributeDescriptions();

    vertexInputInfo.vertexBindingDescriptionCount = 1;
    // vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    // vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    // vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)swapChainExtent.width;
    viewport.height = (float)swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = { 0, 0 };
    scissor.extent = swapChainExtent;

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}


void VideoDisplay::createVertexBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue) {

    // std::vector<Vertex> vertices = {
    //     // indices
    //     {{-1.0f, 0.4f, 1.0f}, {1.0f, 0.0f, 0.0f},{0.0f, 0.5f}},  // 0
    //     {{-0.1f, 0.4f, 0.0f}, {0.0f, 1.0f, 0.0f},{1.0f, 0.5f} }, // 1
    //     {{-0.1f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f},{1.0f, 0.0f} }, // 2
    //     {{-1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f},{0.0f, 0.0f}},   // 3
    //     
    //     {{0.1f, 0.4f, 0.0f}, {1.0f, 0.0f, 0.0f},{0.0f, 1.0f} },    // 4
    //     {{1.0f, 0.4f, 1.0f}, {0.0f, 1.0f, 0.0f},{1.0f, 1.0f} },    // 5
    //     {{1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 1.0f},{1.0f, 0.5f} },    // 6
    //     {{0.1f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.5f} }   // 7
    // };

    //VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
    //VkBuffer stagingBuffer;
    //VkDeviceMemory stagingBufferMemory;
    //std::cout << "went to the stuff without problems" << std::endl;
    //std::cout << "buffersize = " << bufferSize << " or are the flags silly? it's unreasonably big... vertices[0]= " << sizeof(vertices[0]) << " and vertices.size() = " << vertices.size() << std::endl;
    //vkh::createBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
    //void* data;
    //vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    //memcpy(data, vertices.data(), (size_t)bufferSize);
    //vkUnmapMemory(device, stagingBufferMemory);
//
    //vkh::createBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
    //vkh::copyBuffer(device, commandPool, stagingBuffer, vertexBuffer, bufferSize, graphicsQueue);
//
    //vkDestroyBuffer(device, stagingBuffer, nullptr);
    //vkFreeMemory(device, stagingBufferMemory, nullptr);
    //std::cout << "vertexBuffer created" << std::endl;
}

void VideoDisplay::createIndexBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue) {

    VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
    std::cout << "bufferSize = " << bufferSize << sizeof(indices[0]) <<  indices.size() << std::endl;
    std::cout << "indices[10] = " << indices[10] << std::endl;
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    vkh::createBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
    std::cout << "IndexBuffer 1: createBuffer worked " << std::endl;
    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, indices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    vkh::createBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);
    std::cout << "IndexBuffer 2: createBuffer worked " << std::endl;
    vkh::copyBuffer(device, commandPool, stagingBuffer, indexBuffer, bufferSize, graphicsQueue);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void VideoDisplay::createUniformBuffers(VkDevice device, VkPhysicalDevice physicalDevice, size_t numSwapChainImages) {
    //VkDeviceSize bufferSize = sizeof(vkh::UniformBufferObject);

    uniformBuffers.resize(numSwapChainImages);
    uniformBuffersMemory.resize(numSwapChainImages);

    //for (size_t i = 0; i < numSwapChainImages; i++) {
    //    vkh::createBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
    //}
    std::cout << "Uniformbuffers: createBuffer worked " << std::endl;
}

void VideoDisplay::updateUniformBuffer(VkDevice device, VkExtent2D swapChainExtent, uint32_t currentImage) {
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    //vkh::UniformBufferObject ubo = {};
    //// rotate image for 180 degrees, otherwise it's on the head
    //ubo.model = glm::rotate(glm::mat4(1.0f), 1 * glm::radians(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    //// camera looking from positioned on the Z-Axis at Z=8 towards (0;0;0), while upwards face is Y
    //ubo.view = glm::lookAt(glm::vec3(0.0f, 3.0f, 8.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    //// Field of View and clipping distances.
    //ubo.proj = glm::perspective(glm::radians(15.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 40.0f);
    //ubo.proj[1][1] *= -1;

    void* data;
    //vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
    //memcpy(data, &ubo, sizeof(ubo));
    //vkUnmapMemory(device, uniformBuffersMemory[currentImage]);
}

void VideoDisplay::createDescriptorPool(VkDevice device, size_t numSwapChainImages) {
    std::array<VkDescriptorPoolSize, 2> poolSizes = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(numSwapChainImages);
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(numSwapChainImages);
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(numSwapChainImages);

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

void VideoDisplay::createDescriptorSets(VkDevice device, size_t numSwapChainImages, VkImageView combinedImageView, VkSampler combinedSampler) {
    std::vector<VkDescriptorSetLayout> layouts(numSwapChainImages, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(numSwapChainImages);
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets.resize(numSwapChainImages);
    if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < numSwapChainImages; i++) {
        VkDescriptorBufferInfo bufferInfo = {};
        bufferInfo.buffer = uniformBuffers[i];
        bufferInfo.offset = 0;
        //bufferInfo.range = sizeof(vkh::UniformBufferObject);

        VkDescriptorImageInfo imageInfo = {};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = combinedImageView;
        imageInfo.sampler = combinedSampler;

        std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = descriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = descriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &imageInfo;

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}