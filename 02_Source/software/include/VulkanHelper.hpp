#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <optional>
#include <fstream>
#include "linmath.h"


namespace vkh {

	struct SwapChainSupportDetails {
		VkSurfaceCapabilitiesKHR capabilities{};
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;
	};

	std::vector<char> readFile(const std::string& filename);


	
	struct QueueFamilyIndices {
		uint32_t graphicsFamily;
		uint32_t presentFamily;
		uint32_t computeFamily;
		bool isComplete() {
			return graphicsFamily >= 0 && presentFamily >= 0 && computeFamily>= 0;
		}
	};


	void copyBuffer(VkDevice device, VkCommandPool commandPool, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, VkQueue graphicsQueue);

	void copyBufferToImage(VkDevice device, VkCommandPool commandPool, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, VkQueue graphicsQueue);

	void createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);

	void createBufferExtMem(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, 
                            VkExternalMemoryHandleTypeFlagsKHR extMemHandleType,
                            VkBuffer& buffer, VkDeviceMemory& bufferMemory);

	void createImage(VkDevice device, VkPhysicalDevice physicalDevice, uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);

	VkImageView createImageView(VkDevice device, VkImage image, VkFormat format);

	VkCommandBuffer beginSingleTimeCommands(VkDevice device, VkCommandPool commandPool);

	void endSingleTimeCommands(VkDevice device, VkCommandPool commandPool, VkCommandBuffer commandBuffer, VkQueue graphicsQueue);

	VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code);

	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);

	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);

	VkExtent2D chooseSwapExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& capabilities);

	void transitionImageLayout(VkDevice device, VkCommandPool commandPool, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, VkQueue graphicsQueue);

	bool isDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR surface);

	uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);

	bool checkDeviceExtensionSupport(VkPhysicalDevice device);

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface);

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface);

	std::vector<const char*> getRequiredExtensions(bool enableValidationLayers);

}


    