#pragma warning(push)
#pragma warning(disable : 26812) // disabling a warning when including a header works normally for most warnings.
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#pragma warning(pop)

// Standard C++17 Includes
#include <iostream>
#include <csignal>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <optional>
#include <set>
#include <chrono>
#include <tuple>

 // Project Includes
#include "VulkanFramework.hpp"
#include "computation.cuh"
#include "FrameCapture.hpp"
#include "VulkanHelper.hpp"
#include "PositionEstimate.hpp"
#include "MainCamera.hpp"
#include "TCPFrameCapture.hpp"


#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif
#pragma warning(push)
#pragma warning(disable : 26812) // disabling a warning when including a header works normally for most warnings.

#define FPS_MAX_INV 2000 // milliseconds

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanFramework::debugCallback(
    VkDebugReportFlagsEXT flags,
    VkDebugReportObjectTypeEXT objType,
    uint64_t obj,
    size_t location,
    int32_t code,
    const char* layerPrefix,
    const char* msg,
    void* userData) {
    std::cerr << "validation layer: " << msg << std::endl;
    return VK_FALSE;
}

VkResult CreateDebugReportCallbackEXT(VkInstance instance, const VkDebugReportCallbackCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugReportCallbackEXT* pCallback) {
    auto func = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pCallback);
    }
    else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}


void DestroyDebugReportCallbackEXT(VkInstance instance, VkDebugReportCallbackEXT callback, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
    if (func != nullptr) {
        func(instance, callback, pAllocator);
    }
}



void signalHandler( int signum) {
   std::cout << "Interrupt signal (" << signum << ") received.\n";

   glfwSetWindowShouldClose(globalFrameworkPtr->window, GLFW_TRUE);
}

void VulkanFramework::run() {
    signal(SIGINT, signalHandler); 
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}


void VulkanFramework::initWindow() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(1920, 1080, "MT Wegr", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

// Static class, compiler knows that from the header file.
void VulkanFramework::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = reinterpret_cast<VulkanFramework*>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
}

void VulkanFramework::initVulkan() {
    computation = new Computation{};
    positionEstimate = new PositionEstimate();
    tcpCapture.start(computation);
    createInstance();
    setupDebugCallback();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    getKhrExtensionsFn();
    createSwapChain();
    
    createImageViews();
    createRenderPass();
    createFramebuffers();
    createCommandPool();
    createTextureImage();
    createTextureImageView();
    createTextureSampler();
    createSyncObjects();
    projectedSurface.create(device, physicalDevice, renderPass, commandPool, graphicsQueue,swapChainExtent, swapChainImages.size(), projectedImageView, projectedSampler, positionEstimate);
    mainCamera.create(device, physicalDevice, renderPass, commandPool, graphicsQueue,swapChainExtent, swapChainImages.size(), mainImageView, mainSampler);

    createCommandBuffers();

}

void VulkanFramework::mainLoop() {
    
    while (!glfwWindowShouldClose(window)) {
        
        startTime = std::chrono::steady_clock::now();

        glfwPollEvents();

        drawFrame();

        endTime = std::chrono::steady_clock::now();

        std::chrono::steady_clock::duration timeSpan = endTime - startTime;

        double nseconds = double(timeSpan.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
        // std::cout << "Calculation and rendering in " << 1/nseconds << "fps"<<  std::endl;
    }

    vkDeviceWaitIdle(device);
}

void VulkanFramework::cleanupSwapChain() {
    for (auto framebuffer : swapChainFramebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }

    vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
    vkDestroyRenderPass(device, renderPass, nullptr);

    for (auto imageView : swapChainImageViews) {
        vkDestroyImageView(device, imageView, nullptr);
    }

    vkDestroySwapchainKHR(device, swapChain, nullptr);
    mainCamera.cleanupSwapChain(device, swapChainImages.size());
    projectedSurface.cleanupSwapChain(device, swapChainImages.size());
    if (descriptorPool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);

}


void VulkanFramework::cleanup() {


    vkDestroyBuffer(device, stagingProjectedBuffer, nullptr);
    vkFreeMemory(device, stagingProjectedBufferMemory, nullptr);

    vkDestroySampler(device, projectedSampler, nullptr);
    vkDestroyImageView(device, projectedImageView, nullptr);

    vkDestroyImage(device, projectedImage, nullptr);
    vkFreeMemory(device, projectedImageMemory, nullptr);

    vkDestroyBuffer(device, stagingMainBuffer, nullptr);
    vkFreeMemory(device, stagingMainBufferMemory, nullptr);

    vkDestroySampler(device, mainSampler, nullptr);
    vkDestroyImageView(device, mainImageView, nullptr);

    vkDestroyImage(device, mainImage, nullptr);
    vkFreeMemory(device, mainImageMemory, nullptr);
    cleanupSwapChain();
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(device, inFlightFences[i], nullptr);
    }
    mainCamera.cleanup(device);
    projectedSurface.cleanup(device);
    vkDestroyCommandPool(device, commandPool, nullptr);

    vkDestroyDevice(device, nullptr);

    if (enableValidationLayers) {
        DestroyDebugReportCallbackEXT(instance, callback, nullptr);
    }
    tcpCapture.cleanup();
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);

    glfwDestroyWindow(window);

    glfwTerminate();
}

void VulkanFramework::createInstance() {

        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Vulkan CUDA Sinewave";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;

        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> enabledExtensionNameList;
        enabledExtensionNameList.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        enabledExtensionNameList.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
        enabledExtensionNameList.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

        for (int i = 0; i < glfwExtensionCount; i++)
        {
            enabledExtensionNameList.push_back(glfwExtensions[i]);
        }
        if (enableValidationLayers) {
            enabledExtensionNameList.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            createInfo.enabledLayerCount = 0;
        }

        createInfo.enabledExtensionCount = enabledExtensionNameList.size();
        createInfo.ppEnabledExtensionNames = enabledExtensionNameList.data();

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }

        // pointCloud.fpGetPhysicalDeviceProperties2 = (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2");
        // if (pointCloud.fpGetPhysicalDeviceProperties2 == NULL) {
        //     throw std::runtime_error("Vulkan: Proc address for \"vkGetPhysicalDeviceProperties2KHR\" not found.\n");
        // }
// 
// 
        // pointCloud.fpGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetInstanceProcAddr(instance, "vkGetMemoryFdKHR");
        // if (pointCloud.fpGetMemoryFdKHR == NULL) {
        //     throw std::runtime_error("Vulkan: Proc address for \"vkGetMemoryFdKHR\" not found.\n");
        // }

}

void VulkanFramework::setupDebugCallback() {
    if (!enableValidationLayers) return;
    VkDebugReportCallbackCreateInfoEXT createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
    createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
    createInfo.pfnCallback = debugCallback;
    if (CreateDebugReportCallbackEXT(instance, &createInfo, nullptr, &callback) != VK_SUCCESS) {
        throw std::runtime_error("failed to set up debug callback!");
    }
}

void VulkanFramework::createSurface() {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
}

void VulkanFramework::pickPhysicalDevice() { 
        uint32_t deviceCount = 0;
 
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices) {
            if (vkh::isDeviceSuitable(device, surface)) {
                physicalDevice = device;
                break;
            }
        }
        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }

        VkPhysicalDeviceIDProperties vkPhysicalDeviceIDProperties = {};
        vkPhysicalDeviceIDProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
        vkPhysicalDeviceIDProperties.pNext = NULL;

        VkPhysicalDeviceProperties2 vkPhysicalDeviceProperties2 = {};
        vkPhysicalDeviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        vkPhysicalDeviceProperties2.pNext = &vkPhysicalDeviceIDProperties;

        // pointCloud.fpGetPhysicalDeviceProperties2(physicalDevice, &vkPhysicalDeviceProperties2);

        // memcpy(pointCloud.computation.vkDeviceUUID, vkPhysicalDeviceIDProperties.deviceUUID, sizeof(pointCloud.computation.vkDeviceUUID));

}

void VulkanFramework::createLogicalDevice() {        
    
        vkh::QueueFamilyIndices indices = vkh::findQueueFamilies(physicalDevice,surface);
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<int> uniqueQueueFamilies = {(int)indices.graphicsFamily, (int)indices.presentFamily};

        float queuePriority = 1.0f;
        for (int queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo = {};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

         
        VkPhysicalDeviceFeatures deviceFeatures = {};
        deviceFeatures.samplerAnisotropy = VK_TRUE;
        VkDeviceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.queueCreateInfoCount = queueCreateInfos.size();

        createInfo.pEnabledFeatures = &deviceFeatures;
        std::vector<const char*> enabledExtensionNameList;

        for (int i = 0; i < deviceExtensions.size(); i++)
        {
            enabledExtensionNameList.push_back(deviceExtensions[i]);
        }
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            createInfo.enabledLayerCount = 0;
        }
        createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensionNameList.size());
        createInfo.ppEnabledExtensionNames = enabledExtensionNameList.data();
        // std::cout << "creating logical device!" << std::endl;
        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }
        vkGetDeviceQueue(device, indices.graphicsFamily, 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily, 0, &presentQueue);
}

void VulkanFramework::getKhrExtensionsFn() {
    // pointCloud.fpGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR");
    // if (pointCloud.fpGetSemaphoreFdKHR == NULL) {
    //     throw std::runtime_error("Vulkan: Proc address for \"vkGetSemaphoreFdKHR\" not found.\n");
    // }
}

void VulkanFramework::createSwapChain() {
    vkh::SwapChainSupportDetails swapChainSupport = vkh::querySwapChainSupport(physicalDevice, surface);

    VkSurfaceFormatKHR surfaceFormat = vkh::chooseSwapSurfaceFormat(swapChainSupport.formats);
#pragma warning(push)
#pragma warning(disable : 26812) // disabling a warning when including a header works normally for most warnings.
    VkPresentModeKHR presentMode = vkh::chooseSwapPresentMode(swapChainSupport.presentModes);
#pragma warning(pop)
    VkExtent2D extent = vkh::chooseSwapExtent(window,swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;

    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    vkh::QueueFamilyIndices indices = vkh::findQueueFamilies(physicalDevice, surface);
    uint32_t queueFamilyIndices[] = { indices.graphicsFamily, indices.presentFamily };

    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
        throw std::runtime_error("failed to create swap chain!");
    }

    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

#pragma warning(push)
#pragma warning(disable : 26812) // disabling a warning when including a header works normally for most warnings.
    swapChainImageFormat = surfaceFormat.format;
#pragma warning(pop)

    swapChainExtent = extent;
}

void VulkanFramework::recreateSwapChain() {

    // what to do if window is minimized -> no need to render
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }
    std::cout << "1" << std::endl;

    vkDeviceWaitIdle(device);
    std::cout << "2" << std::endl;
    cleanupSwapChain();
    std::cout << "3" << std::endl;
    createSwapChain();
    std::cout << "4" << std::endl;
    createImageViews();
    std::cout << "5" << std::endl;
    createRenderPass();  
    std::cout << "6" << std::endl;
    createFramebuffers();
    std::cout << "7" << std::endl;

    projectedSurface.recreate(device, physicalDevice, renderPass, swapChainExtent, swapChainImages.size(),projectedImageView,projectedSampler);
    mainCamera.recreate(device, physicalDevice, renderPass, swapChainExtent, swapChainImages.size(),mainImageView,mainSampler);
    std::cout << "8" << std::endl;
    createCommandBuffers();
    std::cout << "fin" << std::endl;
}

void VulkanFramework::createImageViews() {
    swapChainImageViews.resize(swapChainImages.size());
    for (size_t i = 0; i < swapChainImages.size(); i++) {
        VkImageViewCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = swapChainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = swapChainImageFormat;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image views!");
        }
    }
}

void VulkanFramework::createRenderPass() {
    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    std::array<VkSubpassDescription, 1> subpassDescriptions{};

    subpassDescriptions[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescriptions[0].colorAttachmentCount = 1;
    subpassDescriptions[0].pColorAttachments = &colorAttachmentRef;

    std::array<VkSubpassDependency, 1> dependencies;

    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].dependencyFlags = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].srcAccessMask = 0;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;


    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = static_cast<uint32_t>(subpassDescriptions.size());
    renderPassInfo.pSubpasses = subpassDescriptions.data();
    renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
    renderPassInfo.pDependencies = dependencies.data();



    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }
}


void VulkanFramework::createFramebuffers() {
    swapChainFramebuffers.resize(swapChainImageViews.size());

    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        VkImageView attachments[] = {
            swapChainImageViews[i]
        };

        VkFramebufferCreateInfo framebufferInfo = {};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}

void VulkanFramework::createCommandPool() {
    vkh::QueueFamilyIndices queueFamilyIndices = vkh::findQueueFamilies(physicalDevice,surface);

    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }
}

void VulkanFramework::createCommandBuffers() {
    commandBuffers.resize(swapChainFramebuffers.size());

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();
    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
    }
    for (size_t i = 0; i < commandBuffers.size(); i++) {
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[i];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;

        VkClearValue clearColor = { 0.0f, 0.0f, 0.0f, 0.0f };
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;
        
        vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        VkDeviceSize offsets[] = { 0 };

       VkBuffer vertexBufferMain[] = { mainCamera.vertexBuffer };
       
       vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, mainCamera.graphicsPipeline);
       vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBufferMain, offsets);
       vkCmdBindIndexBuffer(commandBuffers[i], mainCamera.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
       vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, mainCamera.pipelineLayout, 0, 1, &mainCamera.descriptorSets[i], 0, nullptr);
       vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(mainCamera.indices.size()), 1, 0, 0, 0);   


       VkBuffer vertexBufferProjected[] = { projectedSurface.vertexBuffer };
       vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, projectedSurface.graphicsPipeline);
       vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBufferProjected, offsets);
       vkCmdBindIndexBuffer(commandBuffers[i], projectedSurface.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
       vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, projectedSurface.pipelineLayout, 0, 1, &projectedSurface.descriptorSets[i], 0, nullptr);
       // vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(projectedSurface.indices.size()), 1, 0, 0, 0);        


        vkCmdEndRenderPass(commandBuffers[i]);

        if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }
}

void VulkanFramework::createSyncObjects() {
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);

    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
        }
    }
    // std::cout << "Created rendering semaphores" << std::endl;
    VkExportSemaphoreCreateInfoKHR vulkanExportSemaphoreCreateInfo = {};
    vulkanExportSemaphoreCreateInfo.sType       = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
    vulkanExportSemaphoreCreateInfo.pNext       = NULL;      
    vulkanExportSemaphoreCreateInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;    
    semaphoreInfo.pNext = &vulkanExportSemaphoreCreateInfo;

    // if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &cudaUpdateVkVertexBufSemaphore) != VK_SUCCESS ||
    //     vkCreateSemaphore(device, &semaphoreInfo, nullptr, &vkUpdateCudaVertexBufSemaphore) != VK_SUCCESS )
    // {
    //         throw std::runtime_error("failed to create synchronization objects for a CUDA-Vulkan!");
    // }
    // std::cout << "Created CUDA semaphores" << std::endl;
    // pointCloud.computation.vkUpdateCudaVertexBufSemaphoreHandle = getVkSemaphoreHandle(VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT, vkUpdateCudaVertexBufSemaphore);
    // pointCloud.computation.cudaUpdateVkVertexBufSemaphoreHandle = getVkSemaphoreHandle(VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT, cudaUpdateVkVertexBufSemaphore);
    // std::cout << "Got Semaphore Handles" << std::endl;
}

void VulkanFramework::drawFrame() {
    vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapChain();
        return;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }
    mainCamera.update(device,commandPool, graphicsQueue, swapChainExtent, imageIndex);
    projectedSurface.update(device,commandPool, graphicsQueue, swapChainExtent, imageIndex);
    
    if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
        vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
    }
    updateTextureImage();
    imagesInFlight[imageIndex] = inFlightFences[currentFrame];
    if (!startSubmit)
    {
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]}; // , vkUpdateCudaVertexBufSemaphore};

        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;
        // printf("reset fences\n");
        vkResetFences(device, 1, &inFlightFences[currentFrame]); 
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }
        startSubmit = 1;
    }
    else
    {
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]}; // , cudaUpdateVkVertexBufSemaphore};
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]}; // , vkUpdateCudaVertexBufSemaphore};

        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;
        vkResetFences(device, 1, &inFlightFences[currentFrame]);
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }
    }

    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapChains[] = { swapChain };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;

    presentInfo.pImageIndices = &imageIndex;
    result = vkQueuePresentKHR(presentQueue, &presentInfo);

    
    // pointCloud.update(device, swapChainExtent, imageIndex);



    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
        framebufferResized = false;
        std::cout << "recreate Swap Chain 2" << std::endl;
        recreateSwapChain();
        return;
    }
    else if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to present swap chain image!");
    }
    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    std::this_thread::sleep_for(std::chrono::microseconds(5000));
}


void VulkanFramework::createTextureImage() {
    
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load("textures/Cam2.png", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    VkDeviceSize imageSize = texWidth * texHeight * 4;

    if (!pixels) {
        throw std::runtime_error("failed to load texture image!");
    }

    void* data;

    vkh::createBuffer(device, physicalDevice, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingProjectedBuffer, stagingProjectedBufferMemory);
    vkMapMemory(device, stagingProjectedBufferMemory, 0, imageSize, 0, &data);
    memcpy((void*)(((char*)data)), pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(device, stagingProjectedBufferMemory);
    vkh::createImage(device, physicalDevice, texWidth, texHeight, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, projectedImage, projectedImageMemory);
    
    vkh::transitionImageLayout(device, commandPool, projectedImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, graphicsQueue);
    vkh::copyBufferToImage(device, commandPool, stagingProjectedBuffer, projectedImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), graphicsQueue);
    vkh::transitionImageLayout(device, commandPool, projectedImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, graphicsQueue);

 
    texWidth = 265;
    texHeight = 205;
    imageSize = texWidth * texHeight * 4*sizeof(uint16_t);
    int mtx = tcpCapture.lockMutex();
    pixels = (unsigned char*)((void*)tcpCapture.getToFFrame(mtx));
    // memset(pixels,0,imageSize);
    // for (int i = 0; i<352*286; i++) {
    //     pixels[8*i+0] = 0;
    //     pixels[8*i+1] = 0;
    //     pixels[8*i+2] = 0;
    //     pixels[8*i+3] = 0;
    //     pixels[8*i+4] = 255;
    //     pixels[8*i+5] = 255;
    //     pixels[8*i+6] = 0;
    //     pixels[8*i+7] = 0;        
    // }

    if (!pixels) {
        throw std::runtime_error("failed to load texture image!");
    }

    vkh::createBuffer(device, physicalDevice, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingMainBuffer, stagingMainBufferMemory);
    vkMapMemory(device, stagingMainBufferMemory, 0, imageSize, 0, &data);

    memcpy((void*)(((uint16_t*)data)), pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(device, stagingMainBufferMemory);
    tcpCapture.unlockMutex(mtx);
    vkh::createImage(device, physicalDevice, texWidth, texHeight, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mainImage, mainImageMemory);
    
    vkh::transitionImageLayout(device, commandPool, mainImage, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, graphicsQueue);
    vkh::copyBufferToImage(device, commandPool, stagingMainBuffer, mainImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), graphicsQueue);
    vkh::transitionImageLayout(device, commandPool, mainImage, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, graphicsQueue);


}


void VulkanFramework::createTextureImageView() {
    projectedImageView = vkh::createImageView(device, projectedImage, VK_FORMAT_R8G8B8A8_UNORM); 
    mainImageView = vkh::createImageView(device, mainImage, VK_FORMAT_R16G16B16A16_UNORM); 
}

void VulkanFramework::createTextureSampler() {
    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.pNext = nullptr;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = 16;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &projectedSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }
    if (vkCreateSampler(device, &samplerInfo, nullptr, &mainSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }
}

void VulkanFramework::updateTextureImage() {
    

    int texWidth = 64;
    int texHeight = 64;
    int texChannels = 4;
    unsigned char* pixels;
    VkDeviceSize imageSize = texWidth * texHeight * 4;
    void* data;
    // vkMapMemory(device, stagingProjectedBufferMemory, 0, imageSize, 0, &data);
    // memcpy(data, pixels, static_cast<size_t>(imageSize));
    // vkUnmapMemory(device, stagingProjectedBufferMemory);
// 
  // 
    // vkh::transitionImageLayout(device, commandPool, projectedImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, graphicsQueue);
    // vkh::copyBufferToImage(device, commandPool, stagingProjectedBuffer, projectedImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), graphicsQueue);
    // vkh::transitionImageLayout(device, commandPool, projectedImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, graphicsQueue);


    texWidth = 265;
    texHeight = 205;
    imageSize = texWidth * texHeight * 4*sizeof(uint16_t);
    int mtx = tcpCapture.lockMutex();
    pixels = (unsigned char*)((void*)tcpCapture.getToFFrame(mtx));

    if (!pixels) {
        throw std::runtime_error("failed to load texture image!");
    }
    std::cout << "buffers_h at use: "<< (int)pixels[50*265*4+50*4+0] << std::endl;
    vkMapMemory(device, stagingMainBufferMemory, 0, imageSize, 0, &data);
    memcpy((void*)(((uint16_t*)data)), pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(device, stagingMainBufferMemory);
    tcpCapture.unlockMutex(mtx);
    
    vkh::transitionImageLayout(device, commandPool, mainImage, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, graphicsQueue);
    vkh::copyBufferToImage(device, commandPool, stagingMainBuffer, mainImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), graphicsQueue);
    vkh::transitionImageLayout(device, commandPool, mainImage, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, graphicsQueue);



}

// int VulkanFramework::getVkSemaphoreHandle(VkExternalSemaphoreHandleTypeFlagBitsKHR externalSemaphoreHandleType, VkSemaphore &semVkCuda)
// {
//     if (externalSemaphoreHandleType == VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
//         int fd;
//         VkSemaphoreGetFdInfoKHR vulkanSemaphoreGetFdInfoKHR = {};
//         vulkanSemaphoreGetFdInfoKHR.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
//         vulkanSemaphoreGetFdInfoKHR.pNext      = NULL;
//         vulkanSemaphoreGetFdInfoKHR.semaphore  = semVkCuda;
//         vulkanSemaphoreGetFdInfoKHR.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
//         // pointCloud.fpGetSemaphoreFdKHR(device, &vulkanSemaphoreGetFdInfoKHR, &fd);
//         return fd;
//     }
//     return -1;
// }

bool VulkanFramework::checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers) {
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            return false;
        }
    }

    return true;
}

