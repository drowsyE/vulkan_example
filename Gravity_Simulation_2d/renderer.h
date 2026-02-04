#pragma once

#define GLFW_INCLUDE_VULKAN
#include <glfw/glfw3.h>
#include <glm/glm.hpp>
#include <vector>

void chk(VkResult res, const char* msg);
const char* getDebugSeverityStr(VkDebugUtilsMessageSeverityFlagBitsEXT Severity);
const char* getDebugType(VkDebugUtilsMessageTypeFlagsEXT Type);
void populateDebugMessenger(VkDebugUtilsMessengerCreateInfoEXT& debugMessengerCreateInfo);
VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT Severity, VkDebugUtilsMessageTypeFlagsEXT Type, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* userData);
VkResult createDebugUtilsMessenger(VkInstance& instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);
std::vector<char> readFile(const char* filename);
static void framebufferSizeCallback(GLFWwindow *window, int width, int height);

struct Vertex {
    glm::vec2 pos;
    // glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDesc();
    static std::array<VkVertexInputAttributeDescription, 1> getAttributeDesc();
};

struct Mesh {
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;

    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;

    uint32_t indexCount;
}; 

struct CircleObject{

    glm::vec2 position;
    glm::vec2 velocity;
    glm::vec2 net_force;
    glm::vec3 color;

    float mass;
    float radius;
};

struct RectObject{

    glm::vec2 position;
    glm::vec2 net_force;

    // float mass = 1.0f;
};

struct RenderObject {
    Mesh* mesh;
    glm::mat4 model;
};

struct Scene
{
    std::vector<CircleObject> circles;
    std::vector<RectObject> rects;
    Mesh circleMesh;
    Mesh rectMesh;
};

struct PushConstantData
{
    glm::mat4 mvp;
    glm::vec3 color;
};


class Renderer {
public:

    Renderer(int width, int height);
    ~Renderer();

    void drawFrame(const Scene& scene);
    void createVertexBuffer(const std::vector<Vertex> *vertices, VkBuffer &vertexBuffer, VkDeviceMemory &vertexBufferMemory);
    void createIndexBuffer(const std::vector<uint16_t> *indices, VkBuffer &indexBuffer, VkDeviceMemory &indexBufferMemory);

    bool framebufferResized = false;

    GLFWwindow* window;
    VkExtent2D imageExtent;
    VkDevice device;

private:

    int currentFrame = 0;
    
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;

    VkPhysicalDevice phys_dev;
    uint32_t graphicsFamilyIndex = -1;
    uint32_t presentFamilyIndex = -1;
    VkQueue graphicsQueue;
    VkQueue presentQueue;

    VkSwapchainKHR swapchain;
    VkFormat imageFormat;
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    std::vector<VkFramebuffer> framebuffers;
    
    VkRenderPass renderpass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    // VkBuffer vertexBuffer;
    // VkBuffer indexBuffer;
    // VkDeviceMemory vertexBufferMemory;
    // VkDeviceMemory indexBufferMemory;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;

    void initWindow(int width, int height);
    void createInstance();
    void setupDebugMessenger();
    void destroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);
    void createSurface();
    void selectPhysicalDevice();
    void createLogicalDevice();
    void createSwapchain();
    void createImageViews();
    void createRenderpass();
    VkShaderModule createShader(const char* filename);
    void createGraphicsPipeline();
    void createFramebuffer();
    void createCommandPool();
    void createCommandBuffers();
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags properties, VkBuffer &buffer, VkDeviceMemory &memory);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void copyBuffer(VkBuffer &srcBuffer, VkBuffer &dstBuffer, VkDeviceSize size);
    void createSyncObjects();
    void cleanupSwapchain();
    void recreateSwapchain(uint32_t imageIndex);
    void recordCommandBuffer(VkCommandBuffer commandbuffer, uint32_t imageIndex, const Scene &scene);
  
};