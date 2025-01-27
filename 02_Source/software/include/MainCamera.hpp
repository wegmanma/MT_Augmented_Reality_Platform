#pragma once
#include <vector>
#include <array>
#include <vulkan/vulkan.h>


class MainCamera  {

public:

    const std::vector<uint32_t> indices = {
    2, 1, 0, 0, 3, 2
    };

    VkPipelineLayout pipelineLayout{};
    VkPipeline graphicsPipeline{};

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;

    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;


    void create(VkDevice device, VkPhysicalDevice physicalDevice, VkRenderPass renderPass, VkCommandPool commandPool, VkQueue graphicsQueue, VkExtent2D swapChainExtent, size_t numSwapChainImages, VkImageView mainImageView, VkSampler mainSampler);

    void recreate(VkDevice device, VkPhysicalDevice physicalDevice, VkRenderPass renderPass, VkExtent2D swapChainExtent, size_t numSwapChainImages, VkImageView mainImageView, VkSampler mainSampler);

    void update(VkDevice device, VkCommandPool commandPool, VkQueue graphicsQueue, VkExtent2D swapChainExtent, uint32_t currentImage);

    void cleanupSwapChain(VkDevice device, size_t numSwapChainImages);

    void cleanup(VkDevice device);

    void createUniformBuffers(VkDevice device, VkPhysicalDevice physicalDevice, size_t numSwapChainImages);

    void updateUniformBuffer(VkDevice device, VkExtent2D swapChainExtent, uint32_t currentImage);

    void createGraphicsPipeline(VkDevice device, VkRenderPass renderPass, VkExtent2D swapChainExtent);

    void createVertexBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue);

    void createIndexBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue);

    void createDescriptorPool(VkDevice device, size_t numSwapChainImages);

    void createDescriptorSets(VkDevice device, size_t numSwapChainImages, VkImageView mainImageView, VkSampler mainSampler);

    void createDescriptorSetLayout(VkDevice device);

private:
};