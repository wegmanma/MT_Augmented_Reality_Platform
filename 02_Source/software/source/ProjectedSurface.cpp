#include <algorithm>
#include <array>
#include <assert.h>
#include <chrono>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <vulkan/vulkan.h>

#include "PositionEstimate.hpp"
#include "ProjectedSurface.hpp"
#include "VulkanHelper.hpp"

void ProjectedSurface::cleanupSwapChain(VkDevice device, size_t numSwapChainImages)
{
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

    for (size_t i = 0; i < numSwapChainImages; i++)
    {
        vkDestroyBuffer(device, uniformBuffers[i], nullptr);
        vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
    }
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
}

void ProjectedSurface::cleanup(VkDevice device)
{
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

    vkDestroyBuffer(device, indexBuffer, nullptr);
    vkFreeMemory(device, indexBufferMemory, nullptr);
    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkFreeMemory(device, vertexBufferMemory, nullptr);
}

void ProjectedSurface::create(VkDevice device, VkPhysicalDevice physicalDevice, VkRenderPass renderPass, VkCommandPool commandPool, VkQueue graphicsQueue, VkExtent2D swapChainExtent, size_t numSwapChainImages, VkImageView projectedImageView, VkSampler projectedSampler, PositionEstimate *positionEstimatePtr)
{
    positionEstimate = positionEstimatePtr;
    createDescriptorSetLayout(device);
    createGraphicsPipeline(device, renderPass, swapChainExtent);
    createVertexBuffer(device, physicalDevice, commandPool, graphicsQueue);
    createIndexBuffer(device, physicalDevice, commandPool, graphicsQueue);
    createUniformBuffers(device, physicalDevice, numSwapChainImages);
    createDescriptorPool(device, numSwapChainImages);
    createDescriptorSets(device, numSwapChainImages, projectedImageView, projectedSampler);
}

void ProjectedSurface::recreate(VkDevice device, VkPhysicalDevice physicalDevice, VkRenderPass renderPass, VkExtent2D swapChainExtent, size_t numSwapChainImages, VkImageView projectedImageView, VkSampler projectedSampler)
{
    createGraphicsPipeline(device, renderPass, swapChainExtent);
    createUniformBuffers(device, physicalDevice, numSwapChainImages);
    createDescriptorPool(device, numSwapChainImages);
    createDescriptorSets(device, numSwapChainImages, projectedImageView, projectedSampler);
}

void ProjectedSurface::update(VkDevice device, VkCommandPool commandPool, VkQueue graphicsQueue, VkExtent2D swapChainExtent, uint32_t currentImage)
{
    updateUniformBuffer(device, swapChainExtent, currentImage);
}

void ProjectedSurface::createDescriptorSetLayout(VkDevice device)
{
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

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
}

void ProjectedSurface::createGraphicsPipeline(VkDevice device, VkRenderPass renderPass, VkExtent2D swapChainExtent)
{
    auto vertShaderCode = vkh::readFile("shaders/vert.spv");
    auto fragShaderCode = vkh::readFile("shaders/frag.spv");

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

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    auto bindingDescription = vkh::Vertex::getBindingDescription();
    auto attributeDescriptions = vkh::Vertex::getAttributeDescriptions();

    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)swapChainExtent.width;
    viewport.height = (float)swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
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

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
    {
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

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void ProjectedSurface::createVertexBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue)
{

    std::vector<vkh::Vertex> vertices = {
        // indices
        {{0.0f, 1.0f, 0.5625f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},   // 0
        {{0.0f, -1.0f, 0.5625f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},  // 1
        {{0.0f, -1.0f, -0.5625f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}}, // 2
        {{0.0f, 1.0f, -0.5625f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}}   // 3
    };

    VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    vkh::createBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
    void *data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    vkh::createBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
    vkh::copyBuffer(device, commandPool, stagingBuffer, vertexBuffer, bufferSize, graphicsQueue);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void ProjectedSurface::createIndexBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue)
{

    VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    vkh::createBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
    // std::cout << "IndexBuffer 1: createBuffer worked " << std::endl;
    void *data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, indices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    vkh::createBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);
    // std::cout << "IndexBuffer 2: createBuffer worked " << std::endl;
    vkh::copyBuffer(device, commandPool, stagingBuffer, indexBuffer, bufferSize, graphicsQueue);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void ProjectedSurface::createUniformBuffers(VkDevice device, VkPhysicalDevice physicalDevice, size_t numSwapChainImages)
{
    VkDeviceSize bufferSize = sizeof(vkh::UniformBufferObject);

    uniformBuffers.resize(numSwapChainImages);
    uniformBuffersMemory.resize(numSwapChainImages);

    for (size_t i = 0; i < numSwapChainImages; i++)
    {
        vkh::createBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
    }
    // std::cout << "Uniformbuffers: createBuffer worked " << std::endl;
}

void ProjectedSurface::updateUniformBuffer(VkDevice device, VkExtent2D swapChainExtent, uint32_t currentImage)
{
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    // printf("Gyro data: %05.1f %05.1f %05.1f\n", gyroData[0], gyroData[1] ,gyroData[2]);
    vkh::UniformBufferObject ubo = {};

    mat4x4_identity(ubo.model);
    mat4x4 Model;
    mat4x4_dup(Model, ubo.model);
    mat4x4 translation;
    mat4x4 rotation;
    


    mat4x4_rotate(rotation, Model, 0.0f, 1.0f, 1.0f, degreesToRadians(0.0f));
    mat4x4_translate(translation, 10.0f, 0.0f, 0.0f);
    mat4x4_mul(ubo.model, translation, rotation);

    // std::cout << "ubo.model = " << std::endl;
    // std::cout << "|" << ubo.model[0][0] << ", " << ubo.model[1][0] << ", " << ubo.model[2][0] << ", " << ubo.model[3][0] << "|" << std::endl;
    // std::cout << "|" << ubo.model[0][1] << ", " << ubo.model[1][1] << ", " << ubo.model[2][1] << ", " << ubo.model[3][1] << "|" << std::endl;
    // std::cout << "|" << ubo.model[0][2] << ", " << ubo.model[1][2] << ", " << ubo.model[2][2] << ", " << ubo.model[3][2] << "|" << std::endl;
    // std::cout << "|" << ubo.model[0][3] << ", " << ubo.model[1][3] << ", " << ubo.model[2][3] << ", " << ubo.model[3][3] << "|" << std::endl;

    vec3 eye = {0.0f, 0.0f, 0.0f};
    vec4 center_before = {1.0f, 0.0f, 0.0f, 1.0f};
    vec4 center_after; 
    vec4 up_before = {0.0f, 0.0f, 1.0f, 1.0f};
    vec4 up_after;
    // mat4x4_rotate(rotation, Model, 0.0f, 1.0f, 1.0f, degreesToRadians(25));
    // std::cout << "rotation = " << std::endl;
    // std::cout << "|" << rotation[0][0] << ", " << rotation[1][0] << ", " << rotation[2][0] << ", " << rotation[3][0] << "|" << std::endl;
    // std::cout << "|" << rotation[0][1] << ", " << rotation[1][1] << ", " << rotation[2][1] << ", " << rotation[3][1] << "|" << std::endl;
    // std::cout << "|" << rotation[0][2] << ", " << rotation[1][2] << ", " << rotation[2][2] << ", " << rotation[3][2] << "|" << std::endl;
    // std::cout << "|" << rotation[0][3] << ", " << rotation[1][3] << ", " << rotation[2][3] << ", " << rotation[3][3] << "|" << std::endl;
    positionEstimate->get_gyro_matrix(rotation);
    mat4x4_mul_vec4(center_after, rotation, center_before);
    mat4x4_mul_vec4(up_after, rotation, up_before);
    vec3 center = {center_after[0],center_after[1],center_after[2]};
    vec3 up = {up_after[0],up_after[1],up_after[2]};
    mat4x4_look_at(ubo.view, eye, center, up);
    // std::cout << "center: " << center[0] << ", " << center[1] << ", " << center[2] << ", " << center[3] << std::endl;
    // std::cout << "up: " << up[0] << ", " << up[1] << ", " << up[2] << ", " << up[3] << std::endl;
    // std::cout << "ubo.view = " << std::endl;
    // std::cout << "|" << ubo.view[0][0] << " " << ubo.view[1][0] << " " << ubo.view[2][0] << " " << ubo.view[3][0] << "|" << std::endl;
    // std::cout << "|" << ubo.view[0][1] << " " << ubo.view[1][1] << " " << ubo.view[2][1] << " " << ubo.view[3][1] << "|" << std::endl;
    // std::cout << "|" << ubo.view[0][2] << " " << ubo.view[1][2] << " " << ubo.view[2][2] << " " << ubo.view[3][2] << "|" << std::endl;
    // std::cout << "|" << ubo.view[0][3] << " " << ubo.view[1][3] << " " << ubo.view[2][3] << " " << ubo.view[3][3] << "|" << std::endl;

    mat4x4_perspective(ubo.proj, degreesToRadians(62.2f/2.f), swapChainExtent.width / (float)(swapChainExtent.height)*0.9, 0.1f, 20.0f);
    ubo.proj[1][1] *= -1;

    // std::cout << "ubo.proj = " << std::endl;
    // std::cout << "|" << ubo.proj[0][0] << " " << ubo.proj[1][0] << " " << ubo.proj[2][0] << " " << ubo.proj[3][0] << "|" << std::endl;
    // std::cout << "|" << ubo.proj[0][1] << " " << ubo.proj[1][1] << " " << ubo.proj[2][1] << " " << ubo.proj[3][1] << "|" << std::endl;
    // std::cout << "|" << ubo.proj[0][2] << " " << ubo.proj[1][2] << " " << ubo.proj[2][2] << " " << ubo.proj[3][2] << "|" << std::endl;
    // std::cout << "|" << ubo.proj[0][3] << " " << ubo.proj[1][3] << " " << ubo.proj[2][3] << " " << ubo.proj[3][3] << "|" << std::endl;

    // mat4x4 totalMatrix;
    // mat4x4 perspView;
    // vec4 source = {0.0f, 1.0f, 0.5625f, 1.0};
    // vec4 result;
    // mat4x4_mul(perspView,ubo.proj,ubo.view);
    // mat4x4_mul(totalMatrix, perspView,ubo.model);
    // std::cout << "totalMatrix = " << std::endl;
    // std::cout << "|" << totalMatrix[0][0] << ", " << totalMatrix[1][0] << ", " << totalMatrix[2][0] << ", " << totalMatrix[3][0] << "|" << std::endl;
    // std::cout << "|" << totalMatrix[0][1] << ", " << totalMatrix[1][1] << ", " << totalMatrix[2][1] << ", " << totalMatrix[3][1] << "|" << std::endl;
    // std::cout << "|" << totalMatrix[0][2] << ", " << totalMatrix[1][2] << ", " << totalMatrix[2][2] << ", " << totalMatrix[3][2] << "|" << std::endl;
    // std::cout << "|" << totalMatrix[0][3] << ", " << totalMatrix[1][3] << ", " << totalMatrix[2][3] << ", " << totalMatrix[3][3] << "|" << std::endl;
    // mat4x4_mul_vec4(result, totalMatrix, source);
    // std::cout << "(0.0f, 1.0f, 0.5625f) result = " << std::endl;
    // std::cout << "(" << result[0]/result[3] << " " << result[1]/result[3] << " " << result[2]/result[3]  << ")" << std::endl;
    // source[1] = -1.0f;
    // mat4x4_mul_vec4(result, totalMatrix, source);
    // std::cout << "(0.0f, -1.0f, 0.5625f) result = " << std::endl;
    // std::cout << "(" << result[0]/result[3] << " " << result[1]/result[3] << " " << result[2]/result[3]  << ")" << std::endl;
    // source[2] = -0.5625f;
    // mat4x4_mul_vec4(result, totalMatrix, source);
    // std::cout << "(0.0f, -1.0f, -0.5625f) result = " << std::endl;
    // std::cout << "(" << result[0]/result[3] << " " << result[1]/result[3] << " " << result[2]/result[3]  << ")" << std::endl;
    // source[1] = 1.0f;
    // mat4x4_mul_vec4(result, totalMatrix, source);
    // std::cout << "(0.0f, 1.0f, -0.5625f) result = " << std::endl;
    // std::cout << "(" << result[0]/result[3] << " " << result[1]/result[3] << " " << result[2]/result[3]  << ")" << std::endl;
    void *data;

    vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(device, uniformBuffersMemory[currentImage]);
}

void ProjectedSurface::createDescriptorPool(VkDevice device, size_t numSwapChainImages)
{
    std::array<VkDescriptorPoolSize, 2> poolSizes = {};
    // graphics queue needs Uniform Buffer
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(numSwapChainImages);
    // graphics queue needs the regular texture
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(numSwapChainImages);

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    poolInfo.maxSets = static_cast<uint32_t>(numSwapChainImages);

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

void ProjectedSurface::createDescriptorSets(VkDevice device, size_t numSwapChainImages, VkImageView projectedImageView, VkSampler projectedSampler)
{
    std::vector<VkDescriptorSetLayout> layouts(numSwapChainImages, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(numSwapChainImages);
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets.resize(numSwapChainImages);
    if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < numSwapChainImages; i++)
    {
        VkDescriptorBufferInfo bufferInfo = {};
        bufferInfo.buffer = uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(vkh::UniformBufferObject);

        VkDescriptorImageInfo imageInfo = {};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = projectedImageView;
        imageInfo.sampler = projectedSampler;

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
