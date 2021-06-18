#pragma once

#define GLFW_INCLUDE_VULKAN
#include "linmath.h"
#include <GLFW/glfw3.h>
#include <array>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

namespace vkh
{

	struct Vertex
	{
		vec4 pos;
		vec3 color;
		vec3 texcoords;

		static VkVertexInputBindingDescription getBindingDescription()
		{
			VkVertexInputBindingDescription bindingDescription = {};

			bindingDescription.binding = 0;
			bindingDescription.stride = sizeof(Vertex);
			bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

			return bindingDescription;
		}

		static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions()
		{
			std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {};
			attributeDescriptions[0].binding = 0;
			attributeDescriptions[0].location = 0;
			attributeDescriptions[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
			attributeDescriptions[0].offset = offsetof(Vertex, pos);

			attributeDescriptions[1].binding = 0;
			attributeDescriptions[1].location = 1;
			attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
			attributeDescriptions[1].offset = offsetof(Vertex, color);

			attributeDescriptions[2].binding = 0;
			attributeDescriptions[2].location = 2;
			attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
			attributeDescriptions[2].offset = offsetof(Vertex, texcoords);
			return attributeDescriptions;
		}
	};

	struct UniformBufferObject
	{
    	mat4x4 model;
    	mat4x4 view;
    	mat4x4 proj;
	};

	struct SwapChainSupportDetails
	{
		VkSurfaceCapabilitiesKHR capabilities{};
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;
	};

	std::vector<char> readFile(const std::string &filename);

	struct QueueFamilyIndices
	{
		uint32_t graphicsFamily;
		uint32_t presentFamily;
		uint32_t computeFamily;
		bool isComplete()
		{
			return graphicsFamily >= 0 && presentFamily >= 0 && computeFamily >= 0;
		}
	};

	void copyBuffer(VkDevice device, VkCommandPool commandPool, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, VkQueue graphicsQueue);

	void copyBufferToImage(VkDevice device, VkCommandPool commandPool, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, VkQueue graphicsQueue);

	void createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer &buffer, VkDeviceMemory &bufferMemory);

	void createBufferExtMem(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
							VkExternalMemoryHandleTypeFlagsKHR extMemHandleType,
							VkBuffer &buffer, VkDeviceMemory &bufferMemory);

	void createImage(VkDevice device, VkPhysicalDevice physicalDevice, uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage &image, VkDeviceMemory &imageMemory);

	VkImageView createImageView(VkDevice device, VkImage image, VkFormat format);

	VkCommandBuffer beginSingleTimeCommands(VkDevice device, VkCommandPool commandPool);

	void endSingleTimeCommands(VkDevice device, VkCommandPool commandPool, VkCommandBuffer commandBuffer, VkQueue graphicsQueue);

	VkShaderModule createShaderModule(VkDevice device, const std::vector<char> &code);

	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats);

	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes);

	VkExtent2D chooseSwapExtent(GLFWwindow *window, const VkSurfaceCapabilitiesKHR &capabilities);

	void transitionImageLayout(VkDevice device, VkCommandPool commandPool, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, VkQueue graphicsQueue);

	bool isDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR surface);

	uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);

	bool checkDeviceExtensionSupport(VkPhysicalDevice device);

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface);

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface);

	std::vector<const char *> getRequiredExtensions(bool enableValidationLayers);

}
