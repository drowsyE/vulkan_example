#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>

static void framebufferSizeCallback(GLFWwindow* window, int width, int height);

struct Vertex {
    glm::vec2 pos;

    static VkVertexInputBindingDescription getBindingDesc() {
        VkVertexInputBindingDescription bindDesc{};
        bindDesc.binding = 0;
        bindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        bindDesc.stride = sizeof(Vertex);

        return bindDesc;
    }

    static std::array<VkVertexInputAttributeDescription, 1> getAttributeDesc() {
        std::array<VkVertexInputAttributeDescription, 1> attrDescs;
        attrDescs[0] =
            (VkVertexInputAttributeDescription){.location = 0,
                                                .binding = 0,
                                                .format = VK_FORMAT_R32G32_SFLOAT,
                                                .offset = offsetof(Vertex, pos)};

        return attrDescs;
    };
};

struct Particle {
    glm::vec2 position;
    glm::vec2 velocity;
    glm::vec4 color;

    static VkVertexInputBindingDescription getBindingDesc() {
        VkVertexInputBindingDescription bindDesc{};
        bindDesc.binding = 0;
        bindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        bindDesc.stride = sizeof(Particle);

        return bindDesc;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        // velocity는 ssbo만 사용하고 그리는 작업에는 사용하지 않아서 입력 x
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Particle, position);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Particle, color);

        return attributeDescriptions;
    }
};

struct UniformBufferObject {
    float dt = 1.0f;
};

class Renderer {
    
public:

    GLFWwindow *window;
    bool framebufferResized = false;

    Renderer();
    ~Renderer();
    void run();

private:

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;

    VkPhysicalDevice physDev;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue computeQueue;
    VkQueue presentQueue;
    uint32_t graphicsAndComputeFamilyIndex;
    uint32_t presentFamilyIndex;

    VkSwapchainKHR swapchain;
    VkFormat swapchainImageFormat;
    VkExtent2D swapchainImageExtent;
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;

    VkRenderPass renderpass;
    VkPipelineLayout graphicsPipelineLayout;
    VkPipeline graphicsPipeline;
    VkPipelineLayout computePipelineLayout;
    VkPipeline computePipeline;
    std::vector<VkFramebuffer> framebuffers;
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkCommandBuffer> computeCommandBuffers;
    
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkSemaphore> computeFinishedSemaphores;
    std::vector<VkFence> computeInFlightFences;
    
    // std::vector<Vertex> vertices;
    // std::vector<uint16_t> indices;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;
    std::vector<VkBuffer> shaderStorageBuffers;
    std::vector<VkDeviceMemory> shaderStorageBuffersMemory;

    VkDescriptorPool descriptorPool;
    VkDescriptorSetLayout computeDescriptorSetLayout;
    std::vector<VkDescriptorSet> computeDesciptorSets;

    void drawFrame();

    void initWindow();
    void createInstance();
    void setupDebugMessenger();
    void createSurface();
    void selectPhysicalDevice();
    void createLogicalDevice();
    void createSwapchain();
    void createImageViews();
    void createRenderpass();
    void createGraphicsPipeline();
    void createComputePipeline();
    void createFramebuffers();
    void createCommandPool();
    void createCommandBuffers();
    void createComputeCommandBuffers();
    void createSyncObjects();
    void recordCommandbuffer(VkCommandBuffer &commandBuffer, uint32_t imageIndex);
    void recordComputeCommandbuffer(VkCommandBuffer &commandbuffer);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memProps, VkBuffer &buffer, VkDeviceMemory &bufferMemory);
    void createVertexBuffer(std::vector<Vertex> &vertices);
    void createIndexBuffer(std::vector<uint16_t> &indices);
    void createUniformBuffers();
    void updateUniformBuffer(uint32_t currentImage);
    void createShaderStorageBuffers();
    void createDescriptorPool();
    void createDescriptorSetLayout();
    void createDescriptorSets();
    
    void cleanupSwapchain();
    void recreateSwapchain(uint32_t imageIndex);
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);
    void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size);
    VkShaderModule createShader(const char* filename);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
};