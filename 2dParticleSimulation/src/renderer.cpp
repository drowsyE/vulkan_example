#define NDEBUG

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <stdio.h>
#include <vector>
#include <array>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <set>
#include <random>
#include <cstring>
#include "2dParticleSimulation/include/renderer.h"
#include "2dParticleSimulation/include/utils.h"

#define DEFAULT_WIDTH 800
#define DEFAULT_HEIGHT 600
#define WINDOW_TITLE "Black hole simulation"

#define MAX_FRAME_IN_FLIGHT 2
#define PARTICLE_COUNT 1024

int currentFrame = 0;
float lastFrameTime = 0.0f;
double lastTime = 0.0f;

#ifdef NDEBUG
bool enableValidationLayers = false;
#else
bool enableValidationLayers = true;
#endif

std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};

std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,

    #ifdef __APPLE__
    "VK_KHR_portability_subset"
    #endif
};

Renderer::Renderer() {
    initWindow();
    createInstance();
    setupDebugMessenger();
    createSurface();
    selectPhysicalDevice();
    createLogicalDevice();
    createSwapchain();
    createImageViews();
    createRenderpass();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createComputePipeline();
    createFramebuffers();
    createCommandPool();
    createCommandBuffers();
    createComputeCommandBuffers();
    createSyncObjects();
    createUniformBuffers();
    createShaderStorageBuffers();
    createDescriptorPool();
    createDescriptorSets();
}

Renderer::~Renderer() {

    vkDeviceWaitIdle(device);

    // destroy shader storage buffer
    for (VkDeviceMemory &ssbMem : shaderStorageBuffersMemory) {
        vkFreeMemory(device, ssbMem, nullptr);
    }
    for (VkBuffer &ssb : shaderStorageBuffers) {
        vkDestroyBuffer(device, ssb, nullptr);
    }

    // destory uniform buffer
    for (VkDeviceMemory &ubMem : uniformBuffersMemory) {
        vkFreeMemory(device, ubMem, nullptr);
    }
    for (VkBuffer &ub : uniformBuffers) {
        vkDestroyBuffer(device, ub, nullptr);
    }

    for (int i = 0; i < MAX_FRAME_IN_FLIGHT; i++) {
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(device, inFlightFences[i], nullptr);
        vkDestroySemaphore(device, computeFinishedSemaphores[i], nullptr);
        vkDestroyFence(device, computeInFlightFences[i], nullptr);
    }
    for (int i = 0; i < swapchainImages.size(); i++) {
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
    }
    vkDestroyCommandPool(device, commandPool, nullptr);
    for (VkFramebuffer &framebuffer : framebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }
    vkDestroyPipeline(device, computePipeline, nullptr);
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);
    vkDestroyPipelineLayout(device, graphicsPipelineLayout, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, computeDescriptorSetLayout, nullptr);
    vkDestroyRenderPass(device, renderpass, nullptr);
    for (VkImageView &view : swapchainImageViews) {
        vkDestroyImageView(device, view, nullptr);
    }
    vkDestroySwapchainKHR(device, swapchain, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    if (enableValidationLayers) {
        destroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    }
    vkDestroyInstance(instance, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();
}

 void Renderer::run() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        drawFrame();
        // We want to animate the particle system using the last frames time to get smooth, frame-rate independent animation
        double currentTime = glfwGetTime();
        lastFrameTime = (currentTime - lastTime) * 1000.0;
        lastTime = currentTime;
    }

    vkDeviceWaitIdle(device);
}

void Renderer::drawFrame() {
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    // compute submission
    vkWaitForFences(device, 1, &computeInFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    updateUniformBuffer(currentFrame);

    vkResetFences(device, 1, &computeInFlightFences[currentFrame]);
    vkResetCommandBuffer(computeCommandBuffers[currentFrame], 0);

    recordComputeCommandbuffer(computeCommandBuffers[currentFrame]);

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &computeCommandBuffers[currentFrame];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &computeFinishedSemaphores[currentFrame];
    vkQueueSubmit(computeQueue, 1, &submitInfo, computeInFlightFences[currentFrame]);

    // graphics submission
    vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    VkResult res = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
    if (res == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain(-1);
        return;
    } else if (res != VK_SUBOPTIMAL_KHR && res != VK_SUCCESS) {
        throw std::runtime_error("Failed to acquired swapchain image!");
    }

    vkResetFences(device, 1, &inFlightFences[currentFrame]);

    vkResetCommandBuffer(commandBuffers[currentFrame], 0);
    recordCommandbuffer(commandBuffers[currentFrame], imageIndex);

    // **각 세마포어는 같은 인덱스의 stage에서 대기함!!**
    VkSemaphore waitSemaphores[] = {computeFinishedSemaphores[currentFrame], imageAvailableSemaphores[currentFrame]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
    submitInfo.waitSemaphoreCount = 2;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &renderFinishedSemaphores[imageIndex];

    res = vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]);
    if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR || framebufferResized) {
        framebufferResized = false;

        recreateSwapchain(imageIndex);
        return;
    } else if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed at vkQueueSubmit!");
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinishedSemaphores[imageIndex];
    vkQueuePresentKHR(presentQueue, &presentInfo);

    currentFrame = (currentFrame + 1) % MAX_FRAME_IN_FLIGHT;
}

void Renderer::initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(DEFAULT_WIDTH, DEFAULT_HEIGHT, WINDOW_TITLE, nullptr, nullptr);

    glfwSetWindowUserPointer(window, this);
    glfwSetWindowSizeCallback(window, framebufferSizeCallback);

    lastTime = glfwGetTime();
}

void Renderer::createInstance() {

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.apiVersion = VK_API_VERSION_1_2;
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pApplicationName = WINDOW_TITLE;
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No engine";

    uint32_t count;
    const char** exts;
    exts = glfwGetRequiredInstanceExtensions(&count);
    std::vector<const char*> instanceExtensions(exts, exts + count);

    #ifdef __APPLE__
    instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    instanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    #endif

    if (enableValidationLayers) {
        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkDebugUtilsMessengerCreateInfoEXT messengerCI{};
    populateDebugMessenger(messengerCI);

    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;
    instanceInfo.enabledExtensionCount = instanceExtensions.size();
    instanceInfo.ppEnabledExtensionNames = instanceExtensions.data();

    if (enableValidationLayers) {
        instanceInfo.enabledLayerCount = validationLayers.size();
        instanceInfo.ppEnabledLayerNames = validationLayers.data();
        instanceInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &messengerCI;
    } else {
        instanceInfo.enabledLayerCount = 0;
        instanceInfo.ppEnabledLayerNames = nullptr;
    }

    #ifdef __APPLE__
    instanceInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    #endif

    chk(vkCreateInstance(&instanceInfo, nullptr, &instance), "vkCreateInstance");
}

void Renderer::setupDebugMessenger() {
    if (!enableValidationLayers) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    populateDebugMessenger(createInfo);

    chk(createDebugUtilsMessenger(instance, &createInfo, nullptr, &debugMessenger), "createDebugUtilsMessenger");
}

void Renderer::createSurface() {
    chk(glfwCreateWindowSurface(instance, window,  nullptr, &surface), "glfwCreateWindowSurface");
}

void Renderer::selectPhysicalDevice() {
    uint32_t devCnt;
    vkEnumeratePhysicalDevices(instance, &devCnt, nullptr);
    std::vector<VkPhysicalDevice> devices(devCnt);
    vkEnumeratePhysicalDevices(instance, &devCnt, devices.data());

    for (const VkPhysicalDevice &device : devices) {
        VkPhysicalDeviceProperties devProps{};
        vkGetPhysicalDeviceProperties(device, &devProps);

        uint32_t qPropsCnt;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &qPropsCnt, nullptr);
        std::vector<VkQueueFamilyProperties> qFamilyProps(qPropsCnt);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &qPropsCnt, qFamilyProps.data());

        graphicsAndComputeFamilyIndex = -1;
        presentFamilyIndex = -1;
        int i = 0;
        for (const VkQueueFamilyProperties &props : qFamilyProps) {
            if (qFamilyProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT && qFamilyProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                graphicsAndComputeFamilyIndex = i;
            }

            VkBool32 presentSupport = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
            if (presentSupport) {
                presentFamilyIndex = i;
            }

            if (graphicsAndComputeFamilyIndex != -1 && presentFamilyIndex != -1) {
                physDev = device;
                printf("\n[Info] | Device selected : %s\n", devProps.deviceName);
                printf("[Info] | Graphics Family : %d, Present Family : %d\n", graphicsAndComputeFamilyIndex, presentFamilyIndex);
                return;
            }

            ++i;
        }
    }
    throw std::runtime_error("There is no available physical device supporting vulkan!");
}

void Renderer::createLogicalDevice() {

    std::vector<VkDeviceQueueCreateInfo> qCIs;
    float priorities = 1.0f;

    VkDeviceQueueCreateInfo qInfo{};
    qInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qInfo.queueFamilyIndex = graphicsAndComputeFamilyIndex;
    qInfo.queueCount = 1;
    qInfo.pQueuePriorities = &priorities;
    qCIs.push_back(qInfo);

    if (graphicsAndComputeFamilyIndex != presentFamilyIndex) {
        qInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qInfo.queueFamilyIndex = presentFamilyIndex;
        qInfo.queueCount = 1;
        qInfo.pQueuePriorities = &priorities;
        qCIs.push_back(qInfo);
    }

    VkPhysicalDeviceFeatures devFeats{};

    VkDeviceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.queueCreateInfoCount = qCIs.size();
    info.pQueueCreateInfos = qCIs.data();
    info.enabledExtensionCount = deviceExtensions.size();
    info.ppEnabledExtensionNames = deviceExtensions.data();
    info.pEnabledFeatures = &devFeats;
    info.enabledExtensionCount = deviceExtensions.size();
    info.ppEnabledExtensionNames = deviceExtensions.data();
    if (enableValidationLayers) {
        info.enabledLayerCount = validationLayers.size();
        info.ppEnabledLayerNames = validationLayers.data();
    } else {
        info.enabledLayerCount = 0;
        info.ppEnabledLayerNames = nullptr;
    }
   
    chk(vkCreateDevice(physDev, &info, nullptr, &device), "vkCreateDevice");

    vkGetDeviceQueue(device, graphicsAndComputeFamilyIndex, 0, &graphicsQueue);
    vkGetDeviceQueue(device, graphicsAndComputeFamilyIndex, 0, &computeQueue);
    vkGetDeviceQueue(device, presentFamilyIndex, 0, &presentQueue);
}

void Renderer::createSwapchain() {

    // min image count
    VkSurfaceCapabilitiesKHR surfaceCaps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physDev, surface, &surfaceCaps);
    uint32_t minImgs;
    if (surfaceCaps.minImageCount + 1 > surfaceCaps.maxImageCount) {
        minImgs = surfaceCaps.maxImageCount;
    } else {
        minImgs = surfaceCaps.minImageCount + 1;
    }

    uint32_t fmtCnt;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physDev, surface, &fmtCnt, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fmtCnt);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physDev, surface, &fmtCnt, formats.data());

    // image format & colorspace
    VkFormat imageFormat;
    VkColorSpaceKHR imageColorSpace;
    for (const VkSurfaceFormatKHR &fmt : formats) {
        if (fmt.format == VK_FORMAT_R8G8B8_SRGB && fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            imageFormat = fmt.format;
            imageColorSpace = fmt.colorSpace;
            break;
        }
    }
    if (imageFormat != VK_FORMAT_R8G8B8_SRGB) {
        imageFormat = formats[0].format;
        imageColorSpace = formats[0].colorSpace;
    }
    swapchainImageFormat = imageFormat;

    // image extent
    VkExtent2D imageExtent;
    if (surfaceCaps.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        imageExtent = surfaceCaps.currentExtent;
    } else {
        int width, height;
        glfwGetWindowSize(window, &width, &height);

        imageExtent.width = std::clamp<int>(width, surfaceCaps.minImageExtent.width, surfaceCaps.maxImageExtent.width);
        imageExtent.height = std::clamp<int>(height, surfaceCaps.minImageExtent.height, surfaceCaps.maxImageExtent.height);
    }
    swapchainImageExtent = imageExtent;

    // image sharing mode
    VkSharingMode imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    std::vector<uint32_t> queueFamilyIndices = {graphicsAndComputeFamilyIndex};
    if (graphicsAndComputeFamilyIndex != presentFamilyIndex) {
        imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        queueFamilyIndices.push_back(presentFamilyIndex);
    }

    // present mode
    VkPresentModeKHR presentMode;
    uint32_t modeCnt;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physDev, surface, &modeCnt, nullptr);
    std::vector<VkPresentModeKHR> presentModes(modeCnt);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physDev, surface, &modeCnt, presentModes.data());
    for (const VkPresentModeKHR &mode : presentModes) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
            presentMode = mode;
        }
        break;
    }
    if (presentMode != VK_PRESENT_MODE_MAILBOX_KHR) {
        presentMode = VK_PRESENT_MODE_FIFO_KHR;
    }

    VkSwapchainCreateInfoKHR swapInfo{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = nullptr,
        .flags = 0,
        .surface = surface,
        .minImageCount = minImgs,
        .imageFormat = imageFormat,
        .imageColorSpace = imageColorSpace,
        .imageExtent = imageExtent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .imageSharingMode = imageSharingMode,
        .queueFamilyIndexCount = static_cast<uint32_t> (queueFamilyIndices.size()),
        .pQueueFamilyIndices = queueFamilyIndices.data(),
        .preTransform = surfaceCaps.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = presentMode,
        .clipped = VK_TRUE,
        .oldSwapchain = VK_NULL_HANDLE
    };

    chk(vkCreateSwapchainKHR(device, &swapInfo, nullptr, &swapchain), "vkCreateSwapchainKHR");

    vkGetSwapchainImagesKHR(device, swapchain, &minImgs, nullptr);
    swapchainImages.resize(minImgs);
    vkGetSwapchainImagesKHR(device, swapchain, &minImgs, swapchainImages.data());
}

void Renderer::createImageViews() {
    swapchainImageViews.resize(swapchainImages.size());

    for (int i = 0; i < swapchainImages.size(); i++) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = swapchainImages[i];
        viewInfo.format = swapchainImageFormat;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.components = {
            .r = VK_COMPONENT_SWIZZLE_IDENTITY,
            .g = VK_COMPONENT_SWIZZLE_IDENTITY,
            .b = VK_COMPONENT_SWIZZLE_IDENTITY,
            .a = VK_COMPONENT_SWIZZLE_IDENTITY
        };
        viewInfo.subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        };

        chk(vkCreateImageView(device, &viewInfo, nullptr, &swapchainImageViews[i]), "vkCreateImageView");
    }
}

void Renderer::createRenderpass() {

    // attachments
    std::vector<VkAttachmentDescription> attachments;
    VkAttachmentDescription colorAtt{};
    colorAtt.format = swapchainImageFormat;
    colorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAtt.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    colorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments.push_back(colorAtt);
    
    VkAttachmentReference ref{};
    ref.attachment = 0;
    ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    // end of attachments

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &ref;

    // external <-> subpass 0 dependency
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    
    VkRenderPassCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    createInfo.attachmentCount = attachments.size();
    createInfo.pAttachments = attachments.data();
    createInfo.subpassCount = 1;
    createInfo.pSubpasses = &subpass;
    createInfo.dependencyCount = 1;
    createInfo.pDependencies = &dependency;
    
    chk(vkCreateRenderPass(device, &createInfo, nullptr, &renderpass), "vkCreateRenderPass");
}

VkShaderModule Renderer::createShader(const char* filename) {
    auto shaderCode = readFile(filename);
    VkShaderModule shader;
    VkShaderModuleCreateInfo shaderInfo{};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = shaderCode.size();
    shaderInfo.pCode = reinterpret_cast<const uint32_t*> (shaderCode.data());
    vkCreateShaderModule(device, &shaderInfo, nullptr, &shader);
    return shader;
}

void Renderer::createGraphicsPipeline() {

    // ----- vertex input -------
    auto bindingDesc = Particle::getBindingDesc();
    auto attributeDesc = Particle::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo inputInfo{};
    inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    inputInfo.vertexBindingDescriptionCount = 1;
    inputInfo.pVertexBindingDescriptions = &bindingDesc;
    inputInfo.vertexAttributeDescriptionCount = attributeDesc.size();
    inputInfo.pVertexAttributeDescriptions = attributeDesc.data();
    // ------ end of vertex input -----

    // -------- Input assembly ------------
    VkPipelineInputAssemblyStateCreateInfo assemInfo{};
    assemInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assemInfo.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    assemInfo.primitiveRestartEnable = VK_FALSE;
    // ------- end of input assembly -------

    // ----------- shader -----------
    VkShaderModule vertexShaderModule = createShader("2dParticleSimulation/shaders/spv/vert.spv");
    VkShaderModule fragShaderModule = createShader("2dParticleSimulation/shaders/spv/frag.spv");

    VkPipelineShaderStageCreateInfo vertexShaderCI{};
    vertexShaderCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertexShaderCI.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertexShaderCI.module = vertexShaderModule;
    vertexShaderCI.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderCI{};
    fragShaderCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderCI.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderCI.module = fragShaderModule;
    fragShaderCI.pName = "main";

    VkPipelineShaderStageCreateInfo stageInfos[] = {vertexShaderCI, fragShaderCI};
    // ------- end of shader -------

    // skipping tessellation / geometry shader

    // ---------- Rasterizer ----------
    VkPipelineRasterizationStateCreateInfo rasterInfo{};
    rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterInfo.depthClampEnable = VK_FALSE;
    rasterInfo.rasterizerDiscardEnable = VK_FALSE;
    rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
    rasterInfo.lineWidth = 1.0f;
    rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterInfo.depthBiasEnable = VK_FALSE;
    rasterInfo.depthBiasConstantFactor = 0.0f;
    rasterInfo.depthBiasClamp = 0.0f;
    rasterInfo.depthBiasSlopeFactor = 0.0f;
    // ------- end of rasterizer -------

    // --------- Color blending ---------
    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.blendEnable = VK_FALSE;
    blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    blendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blendInfo{};
    blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blendInfo.logicOpEnable = VK_FALSE;
    blendInfo.logicOp = VK_LOGIC_OP_COPY;
    blendInfo.attachmentCount = 1;
    blendInfo.pAttachments = &blendAttachment;
    blendInfo.blendConstants[0] = 0.0f;
    blendInfo.blendConstants[1] = 0.0f;
    blendInfo.blendConstants[2] = 0.0f;
    blendInfo.blendConstants[3] = 0.0f;
    // ----- end of color blending -----

    // ---------- pipeline layout --------- // 
    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 0;
    layoutInfo.pSetLayouts = nullptr;
    layoutInfo.pushConstantRangeCount = 0;
    layoutInfo.pPushConstantRanges = nullptr;
    vkCreatePipelineLayout(device, &layoutInfo, nullptr, &graphicsPipelineLayout);
    // --------- end of pipeline layout ------

    // --------- viewport / scissor ---------
    VkPipelineViewportStateCreateInfo viewportInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .viewportCount = 1,
        .pViewports = nullptr, // for dynamic state
        .scissorCount = 1,
        .pScissors = nullptr, // for dynamic state
    };
    // ------- end of viewport / scissors ----

    // ----- dynamic state -----
    VkDynamicState dynamicStates[2] = {VK_DYNAMIC_STATE_VIEWPORT,
                                        VK_DYNAMIC_STATE_SCISSOR};

    VkPipelineDynamicStateCreateInfo dynamicStateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .dynamicStateCount = 2,
        .pDynamicStates = dynamicStates};
    // ----- end of dynamic state -----

    // ----- multisampling -----
    VkPipelineMultisampleStateCreateInfo sampleInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 1.0f,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE};
    // ----- end of multisampling ----

    VkGraphicsPipelineCreateInfo graphicsPipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stageCount = 2,
        .pStages = stageInfos,
        .pVertexInputState = &inputInfo,
        .pInputAssemblyState = &assemInfo,
        .pTessellationState = VK_NULL_HANDLE,
        .pViewportState = &viewportInfo,
        .pRasterizationState = &rasterInfo,
        .pMultisampleState = &sampleInfo,
        .pDepthStencilState = VK_NULL_HANDLE,
        .pColorBlendState = &blendInfo,
        .pDynamicState = &dynamicStateInfo,
        .layout = graphicsPipelineLayout,
        .renderPass = renderpass,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1};

    chk(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &graphicsPipelineCreateInfo, nullptr, &graphicsPipeline), "vkCreateGraphicsPipelines");

    vkDestroyShaderModule(device, vertexShaderModule, nullptr);
    vkDestroyShaderModule(device, fragShaderModule, nullptr);
}

void Renderer::createComputePipeline() {

    VkShaderModule computeShaderModule = createShader("2dParticleSimulation/shaders/spv/comp.spv");

    VkPipelineShaderStageCreateInfo computeShaderCI{};
    computeShaderCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeShaderCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeShaderCI.module = computeShaderModule;
    computeShaderCI.pName = "main";

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &computeDescriptorSetLayout; // descriptor set layout here
    vkCreatePipelineLayout(device, &layoutInfo, nullptr, &computePipelineLayout);

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = computePipelineLayout;
    pipelineInfo.stage = computeShaderCI;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline);

    vkDestroyShaderModule(device, computeShaderModule, nullptr);
}

void Renderer::createFramebuffers() {
    framebuffers.resize(swapchainImages.size());

    for (int i = 0; i < swapchainImages.size(); i++) {
        VkFramebufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        info.width = swapchainImageExtent.width;
        info.height = swapchainImageExtent.height;
        info.attachmentCount = 1;
        info.pAttachments = &swapchainImageViews[i];
        info.renderPass = renderpass;
        info.layers = 1;

        chk(vkCreateFramebuffer(device, &info, nullptr, &framebuffers[i]), "vkCreateFramebuffer");
    }
}

void Renderer::createCommandPool() {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = graphicsAndComputeFamilyIndex;
    chk(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool), "vkCreateCommandPool");
}

void Renderer::createCommandBuffers() {
    commandBuffers.resize(MAX_FRAME_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = commandBuffers.size();
    allocInfo.commandPool = commandPool;

    chk(vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()), "vkAllocateCommandBuffers");
}

void Renderer::createComputeCommandBuffers() {
    computeCommandBuffers.resize(MAX_FRAME_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = computeCommandBuffers.size();
    allocInfo.commandPool = commandPool;

    chk(vkAllocateCommandBuffers(device, &allocInfo, computeCommandBuffers.data()), "vkAllocateCommandBuffers");
}

void Renderer::createSyncObjects() {

    imageAvailableSemaphores.resize(MAX_FRAME_IN_FLIGHT);
    renderFinishedSemaphores.resize(swapchainImages.size());
    inFlightFences.resize(MAX_FRAME_IN_FLIGHT);
    computeFinishedSemaphores.resize(MAX_FRAME_IN_FLIGHT);
    computeInFlightFences.resize(MAX_FRAME_IN_FLIGHT);

    VkSemaphoreCreateInfo semaInfo{};
    semaInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaInfo.flags = VK_SEMAPHORE_TYPE_BINARY;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < MAX_FRAME_IN_FLIGHT; i++) {
        vkCreateSemaphore(device, &semaInfo, nullptr, &imageAvailableSemaphores[i]);
        vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]);

        vkCreateSemaphore(device, &semaInfo, nullptr, &computeFinishedSemaphores[i]);
        vkCreateFence(device, &fenceInfo, nullptr, &computeInFlightFences[i]);
    } 
    for (int i = 0; i < swapchainImages.size(); i++) {
        chk(vkCreateSemaphore(device, &semaInfo, nullptr, &renderFinishedSemaphores[i]), "vkCreateSemaphore");
    }
}

void Renderer::recordCommandbuffer(VkCommandBuffer &commandBuffer, uint32_t imageIndex) {

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkClearValue clearColor = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkRenderPassBeginInfo renderBeginInfo{};
    renderBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderBeginInfo.renderPass = renderpass;
    renderBeginInfo.renderArea = {.offset = {0, 0}, .extent = swapchainImageExtent};
    renderBeginInfo.framebuffer = framebuffers[imageIndex];
    renderBeginInfo.clearValueCount = 1;
    renderBeginInfo.pClearValues = &clearColor;
    vkCmdBeginRenderPass(commandBuffer, &renderBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = swapchainImageExtent.width;
    viewport.height = swapchainImageExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = swapchainImageExtent;
    scissor.offset = {0, 0};
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    // SSB를 vertex buffer처럼 bind
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &shaderStorageBuffers[currentFrame], offsets);

    vkCmdDraw(commandBuffer, PARTICLE_COUNT, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffer);
    vkEndCommandBuffer(commandBuffer);
}

void Renderer::recordComputeCommandbuffer(VkCommandBuffer &commandbuffer) {
    
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(commandbuffer, &beginInfo);
    
    vkCmdBindPipeline(commandbuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(commandbuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDesciptorSets[currentFrame], 0, nullptr);

    // 1개의 work group에 256 x 1 x 1의 invocation이 있으니, 그 work group이 PARTICLE_COUNT / 256개 있으면 모든 파티클을 계산 가능 (1개의 work group 내의 invocation은 동시에 계산하지만, work group 간의 순서는 알 수 없음(GPU 내부 스케쥴링))
    // 마지막 3개의 인자는 얼마큼의 work group를 dispatch(호출) 할 지 결정
    // As our particles array is linear, we leave the other two dimensions at one, resulting in a one-dimensional dispatch
    vkCmdDispatch(commandbuffer, PARTICLE_COUNT / 256, 1, 1);

    vkEndCommandBuffer(commandbuffer);
}

uint32_t Renderer::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if (typeFilter & (1 << i) && (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type!");
}

void Renderer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags properties, VkBuffer &buffer, VkDeviceMemory &bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usageFlags;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);

    VkMemoryRequirements memReqs{};
    vkGetBufferMemoryRequirements(device, buffer, &memReqs);

    VkMemoryAllocateInfo mallocInfo{};
    mallocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mallocInfo.allocationSize = size;
    mallocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, properties);
    vkAllocateMemory(device, &mallocInfo, nullptr, &bufferMemory);

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

VkCommandBuffer Renderer::beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmdbuf;
    vkAllocateCommandBuffers(device, &allocInfo, &cmdbuf);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdbuf, &beginInfo);

    return cmdbuf;
}

void Renderer::endSingleTimeCommands(VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);

    vkDeviceWaitIdle(device);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void Renderer::copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    auto cmdbuf = beginSingleTimeCommands();

    VkBufferCopy region{};
    region.size = size;
    region.srcOffset = 0;
    region.dstOffset = 0;
    vkCmdCopyBuffer(cmdbuf, src, dst, 1, &region);

    endSingleTimeCommands(cmdbuf);
}

void Renderer::createVertexBuffer(std::vector<Vertex> &vertices) {
    VkDeviceSize bufferSize = sizeof(Vertex) * vertices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                stagingBuffer, stagingBufferMemory);
    
    void *data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(), bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    createBuffer(bufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                vertexBuffer, vertexBufferMemory);

    copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void Renderer::createIndexBuffer(std::vector<uint16_t> &indices) {
    VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                stagingBuffer, stagingBufferMemory);
    
    void *data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, indices.data(), bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    createBuffer(bufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                indexBuffer, indexBufferMemory);

    copyBuffer(stagingBuffer, indexBuffer, bufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void Renderer::createUniformBuffers() {
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    uniformBuffers.resize(MAX_FRAME_IN_FLIGHT);
    uniformBuffersMemory.resize(MAX_FRAME_IN_FLIGHT);
    uniformBuffersMapped.resize(MAX_FRAME_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAME_IN_FLIGHT; i++) {
        createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    uniformBuffers[i], uniformBuffersMemory[i]);

        vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
    }
}

void Renderer::updateUniformBuffer(uint32_t currentFrame) {
    UniformBufferObject ubo{};
    ubo.dt = lastFrameTime * 2.0f;

    memcpy(uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));
}

void Renderer::createShaderStorageBuffers() {
    shaderStorageBuffers.resize(MAX_FRAME_IN_FLIGHT);
    shaderStorageBuffersMemory.resize(MAX_FRAME_IN_FLIGHT);

    std::default_random_engine rndEngine((unsigned)time(nullptr));
    std::uniform_real_distribution<float> rndDist(0.0f, 1.0f);

    std::vector<Particle> particles(PARTICLE_COUNT);
    for (Particle &particle : particles) {
        float r = 0.25f * sqrt(rndDist(rndEngine));
        float theta = rndDist(rndEngine) * 2 * 3.14159265358979323846; // 0 ~ 2pi
        float x = r * cos(theta) * DEFAULT_HEIGHT / DEFAULT_WIDTH;
        float y = r * sin(theta);
        particle.position = glm::vec2(x, y);
        particle.velocity = glm::normalize(glm::vec2(x, y)) * 0.00025f;
        particle.color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine), 1.0f);
    }

    VkDeviceSize bufferSize = sizeof(Particle) * PARTICLE_COUNT;
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                stagingBuffer, stagingBufferMemory);
    
    void *data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, particles.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    for (int i = 0; i < MAX_FRAME_IN_FLIGHT; i++) {
        createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    shaderStorageBuffers[i], shaderStorageBuffersMemory[i]);

        copyBuffer(stagingBuffer, shaderStorageBuffers[i], bufferSize);
    }

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void Renderer::createDescriptorPool() {
    std::array<VkDescriptorPoolSize, 2> poolSizes;
    // UBO in compute shader
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = MAX_FRAME_IN_FLIGHT;

    // SSBO
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = MAX_FRAME_IN_FLIGHT * 2;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = poolSizes.size();
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = MAX_FRAME_IN_FLIGHT;
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
}

void Renderer::createDescriptorSetLayout() {

    std::array<VkDescriptorSetLayoutBinding, 3> bindings;
    bindings[0].binding = 0;
    bindings[0].descriptorCount = 1;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].pImmutableSamplers = nullptr;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorCount = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].pImmutableSamplers = nullptr;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[2].binding = 2;
    bindings[2].descriptorCount = 1;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].pImmutableSamplers = nullptr;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = bindings.size();
    info.pBindings = bindings.data();

    vkCreateDescriptorSetLayout(device, &info, nullptr, &computeDescriptorSetLayout);
}

void Renderer::createDescriptorSets() {
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAME_IN_FLIGHT, computeDescriptorSetLayout);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = layouts.size();
    allocInfo.pSetLayouts = layouts.data();

    computeDesciptorSets.resize(MAX_FRAME_IN_FLIGHT);
    vkAllocateDescriptorSets(device, &allocInfo, computeDesciptorSets.data());

    for (size_t i = 0; i < MAX_FRAME_IN_FLIGHT; i++) {

        VkDescriptorBufferInfo uboInfo{};
        uboInfo.buffer = uniformBuffers[i];
        uboInfo.offset = 0;
        uboInfo.range = sizeof(UniformBufferObject);

        VkDescriptorBufferInfo sbInfoPrevFrame{};
        sbInfoPrevFrame.buffer = shaderStorageBuffers[(i - 1) % MAX_FRAME_IN_FLIGHT];
        sbInfoPrevFrame.offset = 0;
        sbInfoPrevFrame.range = sizeof(Particle) * PARTICLE_COUNT;

        VkDescriptorBufferInfo sbInfoCurFrame{};
        sbInfoCurFrame.buffer = shaderStorageBuffers[i];
        sbInfoCurFrame.offset = 0;
        sbInfoCurFrame.range = sizeof(Particle) * PARTICLE_COUNT;

        std::array<VkWriteDescriptorSet, 3> descriptorWrites{};
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].dstSet = computeDesciptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].pBufferInfo = &uboInfo;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].dstSet = computeDesciptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[1].pBufferInfo = &sbInfoPrevFrame;

        descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].dstSet = computeDesciptorSets[i];
        descriptorWrites[2].dstBinding = 2;
        descriptorWrites[2].dstArrayElement = 0;
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[2].pBufferInfo = &sbInfoCurFrame;

        vkUpdateDescriptorSets(device, 3, descriptorWrites.data(), 0, nullptr);
    }
}

void Renderer::cleanupSwapchain() {
  for (VkFramebuffer fb : framebuffers) {
    vkDestroyFramebuffer(device, fb, nullptr);
  }

  for (VkImageView view : swapchainImageViews) {
    vkDestroyImageView(device, view, nullptr);
  }

  vkDestroySwapchainKHR(device, swapchain, nullptr);
}

void Renderer::recreateSwapchain(uint32_t imageIndex) {
        int width=0, height=0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            if (glfwWindowShouldClose(window)) return;

            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);

        cleanupSwapchain();

        if (imageIndex != -1) {
            vkDestroySemaphore(this->device, renderFinishedSemaphores[imageIndex], nullptr);
            VkSemaphoreCreateInfo createInfo{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0
            };
            vkCreateSemaphore(this->device, &createInfo, nullptr, &renderFinishedSemaphores[imageIndex]);
        }

        createSwapchain();
        createImageViews();
        createFramebuffers();
    }

// _______________________________________________________________
static void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    auto app = reinterpret_cast<Renderer*> (glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
}

