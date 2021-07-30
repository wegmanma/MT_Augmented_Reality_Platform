#pragma once

#pragma warning(push)
#pragma warning(disable : 26812) // disabling a warning when including a header works normally for most warnings.
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#pragma warning(pop)

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <array>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <optional>
#include <set>
#include <chrono>   
#include "FrameCapture.hpp"
#include "VulkanHelper.hpp"
#include "ProjectedSurface.hpp"
#include "PositionEstimate.hpp"
#include "MainCamera.hpp"
#include "TCPFrameCapture.hpp"

class VulkanFramework {


public:
    void run();

    const std::vector<const char*> validationLayers = {
    "VK_LAYER_LUNARG_standard_validation",
    "VK_LAYER_LUNARG_core_validation",
    "VK_LAYER_LUNARG_parameter_validation",
    "VK_LAYER_GOOGLE_unique_objects",
    "VK_LAYER_LUNARG_object_tracker"
    };

    const std::vector<const char*> deviceExtensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
    };

    MainCamera mainCamera{};
    ProjectedSurface projectedSurface{};

    TCPFrameCapture tcpCapture{};

    PositionEstimate *positionEstimate = {};

    GLFWwindow* window{};

    PFN_vkGetMemoryFdKHR  fpGetMemoryFdKHR;
    PFN_vkGetSemaphoreFdKHR fpGetSemaphoreFdKHR;
    PFN_vkGetPhysicalDeviceProperties2 fpGetPhysicalDeviceProperties2;
    uint8_t  vkDeviceUUID[VK_UUID_SIZE];

    int cudaUpdateVkVertexBufSemaphoreHandle;

    int vkUpdateCudaVertexBufSemaphoreHandle;

private:

    const uint32_t WIDTH = 1600;
    const uint32_t HEIGHT = 1080;

    const int MAX_FRAMES_IN_FLIGHT = 1;
    bool startSubmit = 0;
    


    VkInstance instance{};
    VkDebugReportCallbackEXT callback{};
    VkSurfaceKHR surface{};

    VkPhysicalDevice physicalDevice;
    VkDevice device;

    VkQueue graphicsQueue{};
    VkQueue presentQueue{};

    VkSwapchainKHR swapChain{};
    std::vector<VkImage> swapChainImages{};
    VkFormat swapChainImageFormat{};
    VkExtent2D swapChainExtent{};
    std::vector<VkImageView> swapChainImageViews{};
    std::vector<VkFramebuffer> swapChainFramebuffers{};

    VkRenderPass renderPass{};

    VkCommandPool commandPool{};
    std::vector<VkCommandBuffer> commandBuffers{};

    std::vector<VkSemaphore> imageAvailableSemaphores{};
    std::vector<VkSemaphore> renderFinishedSemaphores{};
    std::vector<VkFence> inFlightFences{};
    std::vector<VkFence> imagesInFlight{};
    size_t currentFrame = 0;

    // VkSemaphore cudaUpdateVkVertexBufSemaphore{};
    // int cudaUpdateVkVertexBufSemaphoreHandle{};
    // VkSemaphore vkUpdateCudaVertexBufSemaphore{};
    // int vkUpdateCudaVertexBufSemaphoreHandle{};

    VkBuffer stagingProjectedBuffer;
    VkDeviceMemory stagingProjectedBufferMemory;

    VkImage projectedImage;
    VkDeviceMemory projectedImageMemory;

    VkImageView projectedImageView;
    VkSampler projectedSampler;

    VkBuffer stagingMainBuffer;
    VkDeviceMemory stagingMainBufferMemory;

    VkImage mainImage;
    VkDeviceMemory mainImageMemory;

    VkImageView mainImageView;
    VkSampler mainSampler;

    bool framebufferResized = false;    

    VkDescriptorPool descriptorPool = {VK_NULL_HANDLE};

    VkSampleCountFlagBits msaaSamples{ VK_SAMPLE_COUNT_1_BIT };

    std::chrono::steady_clock::time_point startTime;
    std::chrono::steady_clock::time_point endTime;
    std::chrono::steady_clock::duration timeSpan;
    double nseconds;

    void initWindow();

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);

    void initVulkan();

    void mainLoop();

    void cleanupSwapChain();

    void cleanup();

    void createInstance();

    void setupDebugCallback();

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objType, uint64_t obj, size_t location, int32_t code, const char* layerPrefix, const char* msg, void* userData);
    
    bool checkValidationLayerSupport();

    void createSurface();

    void pickPhysicalDevice();

    void createLogicalDevice();

    void getKhrExtensionsFn();

    void createSwapChain();

    void recreateSwapChain();

    void createImageViews();

    void createRenderPass();

    void createFramebuffers();

    void createCommandPool();

    void createCommandBuffers();

    void createSyncObjects();

    void drawFrame();

    //void compute();

    void createTextureImage();

    void updateTextureImage();

    void createTextureImageView();

    void createTextureSampler();

    // int getVkSemaphoreHandle(VkExternalSemaphoreHandleTypeFlagBitsKHR externalSemaphoreHandleType, VkSemaphore &semVkCuda);

    // int getVkMemHandle(VkExternalMemoryHandleTypeFlagsKHR externalMemoryHandleType);

};

extern VulkanFramework *globalFrameworkPtr;