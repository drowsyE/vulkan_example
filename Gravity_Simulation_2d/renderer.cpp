// #define NDEBUG
#define GLFW_INCLUDE_VULKAN
#include <glfw/glfw3.h>

#include "renderer.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <stdio.h>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#ifdef NDEBUG
bool enableValidationLayers = false;
#else
bool enableValidationLayers = true;
#endif

#define MAX_FRAME_IN_FLIGHT 2

std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};

const std::vector<const char *> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,

#ifdef __APPLE__
    "VK_KHR_portability_subset"
#endif
};

// -------- Vertex -------
VkVertexInputBindingDescription Vertex::getBindingDesc() {
  VkVertexInputBindingDescription bindDesc{};
  bindDesc.binding = 0;
  bindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
  bindDesc.stride = sizeof(Vertex);

  return bindDesc;
}

std::array<VkVertexInputAttributeDescription, 1> Vertex::getAttributeDesc() {
  std::array<VkVertexInputAttributeDescription, 1> attrDescs;
  attrDescs[0] =
      (VkVertexInputAttributeDescription){.location = 0,
                                          .binding = 0,
                                          .format = VK_FORMAT_R32G32_SFLOAT,
                                          .offset = offsetof(Vertex, pos)};

  // color info will given by push constant
  // attrDescs[1] =
  //     (VkVertexInputAttributeDescription){.location = 1,
  //                                         .binding = 0,
  //                                         .format = VK_FORMAT_R32G32B32_SFLOAT,
  //                                         .offset = offsetof(Vertex, color)};

  return attrDescs;
}
// -------- end of Vertex -------

Renderer::Renderer(int width, int height) {

  initWindow(width, height);
  createInstance();
  setupDebugMessenger();
  createSurface();

  selectPhysicalDevice();
  createLogicalDevice();

  createSwapchain();
  createImageViews();

  createRenderpass();
  createGraphicsPipeline();
  createFramebuffer();
  createCommandPool();
  createCommandBuffers();
  createSyncObjects();

  // createVertexBuffer(vertices);
  // createIndexBuffer(indices);
}

Renderer::~Renderer() {

  vkDeviceWaitIdle(device);

  // vkDestroyBuffer(device, indexBuffer, nullptr);
  // vkFreeMemory(device, indexBufferMemory, nullptr);

  // vkDestroyBuffer(device, vertexBuffer, nullptr);
  // vkFreeMemory(device, vertexBufferMemory, nullptr);

  for (int i = 0; i < MAX_FRAME_IN_FLIGHT; i++) {
    vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
    vkDestroyFence(device, inFlightFences[i], nullptr);
  }

  for (int i = 0; i < swapchainImages.size(); i++) {
    vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
  }

  vkDestroyCommandPool(device, commandPool, nullptr);
  for (VkFramebuffer &framebuffer : framebuffers) {
    vkDestroyFramebuffer(device, framebuffer, nullptr);
  }
  vkDestroyPipeline(device, graphicsPipeline, nullptr);
  vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
  vkDestroyRenderPass(device, renderpass, nullptr);
  for (VkImageView &imageView : swapchainImageViews) {
    vkDestroyImageView(device, imageView, nullptr);
  }
  vkDestroySwapchainKHR(device, swapchain, nullptr);
  vkDestroyDevice(device, nullptr);
  vkDestroySurfaceKHR(instance, surface, nullptr);
  if (enableValidationLayers)
    destroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
  vkDestroyInstance(instance, nullptr);
  glfwDestroyWindow(window);
  glfwTerminate();
}

void Renderer::initWindow(int width, int height) {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  window = glfwCreateWindow(width, height, "window", nullptr, nullptr);
  printf("Created window!\n");

  glfwSetWindowUserPointer(window, this);
  glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
}

// callback_signature * void function_name(GLFWwindow* window, int width, int height) *
static void framebufferSizeCallback(GLFWwindow *window, int width, int height) {
  auto app = reinterpret_cast<Renderer*> (glfwGetWindowUserPointer(window));
  app->framebufferResized = true;
}

void Renderer::createInstance() {

  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.apiVersion = VK_API_VERSION_1_0;
  appInfo.pApplicationName = "project";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);

  uint32_t num_exts;
  const char **exts;
  exts = glfwGetRequiredInstanceExtensions(&num_exts);
  std::vector<const char *> extensions(exts, exts + num_exts);

#ifdef __APPLE__
  extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
  extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#endif

  if (enableValidationLayers)
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  VkDebugUtilsMessengerCreateInfoEXT messengerCI{};
  populateDebugMessenger(messengerCI);

  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;
  createInfo.enabledExtensionCount = extensions.size();
  createInfo.ppEnabledExtensionNames = extensions.data();
#ifdef __APPLE__
  createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

  if (enableValidationLayers) {
    createInfo.enabledLayerCount = validationLayers.size();
    createInfo.ppEnabledLayerNames = validationLayers.data();
    createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT *)&messengerCI;
  } else {
    createInfo.enabledLayerCount = 0;
    createInfo.ppEnabledLayerNames = nullptr;
  }

  chk(vkCreateInstance(&createInfo, nullptr, &instance),
      "Failed to create instance!");
  printf("Created vulkan instance!\n");
}

void Renderer::setupDebugMessenger() {
  if (!enableValidationLayers)
    return;

  VkDebugUtilsMessengerCreateInfoEXT createInfo{};
  populateDebugMessenger(createInfo);

  chk(createDebugUtilsMessenger(instance, &createInfo, nullptr,
                                &debugMessenger),
      "Failed to create debug messenger!");
  printf("Created debug messenger!\n");
}

void Renderer::destroyDebugUtilsMessengerEXT(
    VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
    const VkAllocationCallbacks *pAllocator) {
  PFN_vkDestroyDebugUtilsMessengerEXT func =
      (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
          instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}

void Renderer::createSurface() {
  chk(glfwCreateWindowSurface(instance, window, nullptr, &surface),
      "Failed to create window surface!");
  printf("Created window surface!\n");
}

void Renderer::selectPhysicalDevice() {

  uint32_t deviceCnt;
  vkEnumeratePhysicalDevices(instance, &deviceCnt, nullptr);

  if (deviceCnt == 0)
    throw std::runtime_error(
        "There is no available GPUs supporting vulkan! (1)");
  std::vector<VkPhysicalDevice> devices(deviceCnt);
  vkEnumeratePhysicalDevices(instance, &deviceCnt, devices.data());

  for (const VkPhysicalDevice &device : devices) {
    VkPhysicalDeviceProperties deviceProps{};
    vkGetPhysicalDeviceProperties(device, &deviceProps);

    uint32_t numQueueFamilies;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &numQueueFamilies,
                                             nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilyProps(numQueueFamilies);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &numQueueFamilies,
                                             queueFamilyProps.data());

    int i = 0;
    for (const VkQueueFamilyProperties queueFamilyProp : queueFamilyProps) {
      if (queueFamilyProp.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        graphicsFamilyIndex = i;
      }

      VkBool32 presentSupport = VK_FALSE;
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
      if (presentSupport) {
        presentFamilyIndex = i;
      }

      if (graphicsFamilyIndex != -1 && presentFamilyIndex != -1) {
        phys_dev = device;
        printf("\n[Info]\n Device selected.\n Device name : %s\n Api version : "
               "%d\n Driver version : %d\n\n",
               deviceProps.deviceName, deviceProps.apiVersion,
               deviceProps.driverVersion);
        return;
      }
    }

    ++i;
  }

  printf("Graphics family index : %d, Present family index : %d\n",
         graphicsFamilyIndex, presentFamilyIndex);
  throw std::runtime_error("There is no available GPUs supporting vulkan! (1)");
}

void Renderer::createLogicalDevice() {

  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  float priorities[] = {1.0f};

  // graphics queue
  VkDeviceQueueCreateInfo queueCI{};
  queueCI.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCI.queueCount = 1;
  queueCI.queueFamilyIndex = graphicsFamilyIndex;
  queueCI.pQueuePriorities = priorities;
  queueCreateInfos.push_back(queueCI);

  // present queue
  if (presentFamilyIndex != graphicsFamilyIndex) {
    VkDeviceQueueCreateInfo queueCI{};
    queueCI.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCI.queueCount = 1;
    queueCI.queueFamilyIndex = presentFamilyIndex;
    queueCI.pQueuePriorities = priorities;
    queueCreateInfos.push_back(queueCI);
  }

  VkPhysicalDeviceFeatures devFeats{};
  // vkGetPhysicalDeviceFeatures(phys_dev, &devFeats);

  VkDeviceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfo.queueCreateInfoCount = queueCreateInfos.size();
  createInfo.pQueueCreateInfos = queueCreateInfos.data();
  createInfo.pEnabledFeatures = &devFeats;
  createInfo.enabledExtensionCount = deviceExtensions.size();
  createInfo.ppEnabledExtensionNames = deviceExtensions.data();
  if (enableValidationLayers) {
    createInfo.enabledLayerCount = validationLayers.size();
    createInfo.ppEnabledLayerNames = validationLayers.data();
  } else {
    createInfo.enabledLayerCount = 0;
    createInfo.ppEnabledExtensionNames = nullptr;
  }

  chk(vkCreateDevice(phys_dev, &createInfo, nullptr, &device),
      "Failed to create logical device!");
  printf("VkDevice created!\n");

  vkGetDeviceQueue(device, graphicsFamilyIndex, 0, &graphicsQueue);
  vkGetDeviceQueue(device, presentFamilyIndex, 0, &presentQueue);
}

void Renderer::createSwapchain() {
  VkSwapchainCreateInfoKHR createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  createInfo.surface = surface;

  VkSurfaceCapabilitiesKHR surfaceCaps{};
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys_dev, surface, &surfaceCaps);

  // minImageCount
  uint32_t numImgs;
  if (surfaceCaps.minImageCount + 1 > surfaceCaps.maxImageCount) {
    numImgs = surfaceCaps.maxImageCount;
  } else {
    numImgs = surfaceCaps.minImageCount + 1;
  }
  createInfo.minImageCount = numImgs;

  // imageExtent
  if (surfaceCaps.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
    createInfo.imageExtent = surfaceCaps.currentExtent;
    imageExtent = surfaceCaps.currentExtent;
  } else {
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    createInfo.imageExtent.width =
        std::clamp<int>(width, surfaceCaps.minImageExtent.width,
                        surfaceCaps.maxImageExtent.width);
    createInfo.imageExtent.height =
        std::clamp<int>(height, surfaceCaps.minImageExtent.height,
                        surfaceCaps.maxImageExtent.height);
    imageExtent = createInfo.imageExtent;
  }

  // presentMode
  uint32_t presentModeCnt;
  vkGetPhysicalDeviceSurfacePresentModesKHR(phys_dev, surface, &presentModeCnt,
                                            nullptr);
  std::vector<VkPresentModeKHR> presentModes(presentModeCnt);
  vkGetPhysicalDeviceSurfacePresentModesKHR(phys_dev, surface, &presentModeCnt,
                                            presentModes.data());
  for (const VkPresentModeKHR &presentMode : presentModes) {
    if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
      createInfo.presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
    }
  }
  if (createInfo.presentMode != VK_PRESENT_MODE_MAILBOX_KHR)
    createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;

  // imageFormat / imageColorSpace
  uint32_t fmtCnt;
  vkGetPhysicalDeviceSurfaceFormatsKHR(phys_dev, surface, &fmtCnt, nullptr);
  std::vector<VkSurfaceFormatKHR> surfaceFmts(fmtCnt);
  vkGetPhysicalDeviceSurfaceFormatsKHR(phys_dev, surface, &fmtCnt,
                                       surfaceFmts.data());
  for (const VkSurfaceFormatKHR &surfaceFmt : surfaceFmts) {
    if (surfaceFmt.format == VK_FORMAT_R8G8B8A8_SRGB &&
        surfaceFmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      imageFormat = surfaceFmt.format;
      createInfo.imageColorSpace = surfaceFmt.colorSpace;
      break;
    }
  }
  if (createInfo.imageFormat != VK_FORMAT_B8G8R8A8_SRGB) {
    imageFormat = surfaceFmts[0].format;
    createInfo.imageColorSpace = surfaceFmts[0].colorSpace;
  }
  createInfo.imageFormat = imageFormat;

  // imageSharingMode
  createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (graphicsFamilyIndex != presentFamilyIndex) {
    uint32_t familyIndices[] = {graphicsFamilyIndex, presentFamilyIndex};
    createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    createInfo.queueFamilyIndexCount = 2;
    createInfo.pQueueFamilyIndices = familyIndices;
  }

  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  createInfo.preTransform = surfaceCaps.currentTransform;
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  createInfo.clipped = VK_TRUE;
  createInfo.oldSwapchain = VK_NULL_HANDLE;

  chk(vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain),
      "Failed to create swapchain!");
  printf("Created swapchain!\n");

  vkGetSwapchainImagesKHR(device, swapchain, &numImgs, nullptr);
  swapchainImages.resize(numImgs);
  vkGetSwapchainImagesKHR(device, swapchain, &numImgs, swapchainImages.data());
}

void Renderer::createImageViews() {
  swapchainImageViews.resize(swapchainImages.size());

  for (int i = 0; i < swapchainImages.size(); i++) {
    VkImageViewCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = swapchainImages[i];
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    createInfo.format = imageFormat;
    createInfo.components = {.r = VK_COMPONENT_SWIZZLE_R,
                             .g = VK_COMPONENT_SWIZZLE_G,
                             .b = VK_COMPONENT_SWIZZLE_B,
                             .a = VK_COMPONENT_SWIZZLE_A};
    createInfo.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                   .baseMipLevel = 0,
                                   .levelCount = 1,
                                   .baseArrayLayer = 0,
                                   .layerCount = 1};

    chk(vkCreateImageView(device, &createInfo, nullptr,
                          &swapchainImageViews[i]),
        "Failed to create image view!");
  }
  printf("Created Image views!\n");
}

void Renderer::createRenderpass() {

  VkAttachmentDescription attDesc{};
  attDesc.format = imageFormat;
  attDesc.samples = VK_SAMPLE_COUNT_1_BIT;
  attDesc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  attDesc.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  attDesc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attDesc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

  VkAttachmentReference attRef{};
  attRef.attachment = 0,
  attRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &attRef;

  VkSubpassDependency dependency{};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  VkRenderPassCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  createInfo.attachmentCount = 1;
  createInfo.pAttachments = &attDesc;
  createInfo.subpassCount = 1;
  createInfo.pSubpasses = &subpass;
  createInfo.dependencyCount = 1;
  createInfo.pDependencies = &dependency;

  chk(vkCreateRenderPass(device, &createInfo, nullptr, &renderpass),
      "Failed to create renderpass!");
  printf("Created renderpass!\n");
}

VkShaderModule Renderer::createShader(const char *filename) {
  auto shaderCode = readFile(filename);
  VkShaderModule shader;
  VkShaderModuleCreateInfo shaderCI{};
  shaderCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shaderCI.codeSize = shaderCode.size();
  shaderCI.pCode = reinterpret_cast<const uint32_t *>(shaderCode.data());
  vkCreateShaderModule(device, &shaderCI, nullptr, &shader);
  return shader;
}

void Renderer::createGraphicsPipeline() {

  // ----- vertex input -------
  auto bindingDesc = Vertex::getBindingDesc();
  auto attributeDesc = Vertex::getAttributeDesc();

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
  assemInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  assemInfo.primitiveRestartEnable = VK_FALSE;
  // ------- end of input assembly -------

  // ----------- shader -----------
  VkShaderModule vertexShaderModule = createShader("shader/vert.spv");
  VkShaderModule fragShaderModule = createShader("shader/frag.spv");

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

  // tessellation / geometry shader은 생략

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

  // ---------- pipeline layout ----------
  VkPushConstantRange pushRange{};
  pushRange.offset = 0;
  pushRange.size = sizeof(PushConstantData);
  pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  VkPipelineLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layoutInfo.setLayoutCount = 0;
  layoutInfo.pSetLayouts = nullptr;
  layoutInfo.pushConstantRangeCount = 1;
  layoutInfo.pPushConstantRanges = &pushRange;
  vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout);
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
      .layout = pipelineLayout,
      .renderPass = renderpass,
      .subpass = 0,
      .basePipelineHandle = VK_NULL_HANDLE,
      .basePipelineIndex = -1};

  chk(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1,
                                &graphicsPipelineCreateInfo, nullptr,
                                &graphicsPipeline),
      "Failed to create graphics pipeline!");
  printf("Created graphics pipeline!\n");

  vkDestroyShaderModule(device, vertexShaderModule, nullptr);
  vkDestroyShaderModule(device, fragShaderModule, nullptr);
}

void Renderer::createFramebuffer() {
  framebuffers.resize(swapchainImages.size());
  for (int i = 0; i < swapchainImages.size(); i++) {
    VkFramebufferCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    createInfo.width = imageExtent.width;
    createInfo.height = imageExtent.height;
    createInfo.attachmentCount = 1;
    createInfo.pAttachments = &swapchainImageViews[i];
    createInfo.renderPass = renderpass;
    createInfo.layers = 1;
    chk(vkCreateFramebuffer(device, &createInfo, nullptr, &framebuffers[i]),
        "Failed to create framebuffer!");
  }
  printf("Created framebuffer!\n");
}

void Renderer::createCommandPool() {
  VkCommandPoolCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  createInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  createInfo.queueFamilyIndex = graphicsFamilyIndex;

  chk(vkCreateCommandPool(device, &createInfo, nullptr, &commandPool),
      "Failed to create command pool!");
  printf("Created command pool!\n");
}

void Renderer::createCommandBuffers() {
  commandBuffers.resize(MAX_FRAME_IN_FLIGHT);

  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = commandBuffers.size();
  allocInfo.commandPool = commandPool;

  chk(vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()),
      "Failed to create command buffers!");
  printf("Created command buffers!\n");
}

uint32_t Renderer::findMemoryType(uint32_t typeFilter,
                                  VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(phys_dev, &memProps);

  for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
    if (typeFilter & (1 << i) &&
        (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }

  throw std::runtime_error("Failed to find suitable memory type!");
}

void Renderer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usageFlags,
                            VkMemoryPropertyFlags properties, VkBuffer &buffer,
                            VkDeviceMemory &bufferMemory) {

  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.usage = usageFlags;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  bufferInfo.size = size;

  chk(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer),
      "Failed to create buffer!");
  printf("Created Buffer! | usageFlags(hex) : %x\n", usageFlags);

  VkMemoryRequirements memReqs{};
  vkGetBufferMemoryRequirements(device, buffer, &memReqs);

  VkMemoryAllocateInfo mallocInfo{};
  mallocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mallocInfo.allocationSize = size;
  mallocInfo.memoryTypeIndex =
      findMemoryType(memReqs.memoryTypeBits, properties);

  chk(vkAllocateMemory(device, &mallocInfo, nullptr, &bufferMemory),
      "Failed to allocate memory!");
  printf("Successfully allocated memory!\n");

  vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

void Renderer::copyBuffer(VkBuffer &srcBuffer, VkBuffer &dstBuffer,
                          VkDeviceSize size) {

  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = commandPool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer tempCommandbuffer;
  vkAllocateCommandBuffers(device, &allocInfo, &tempCommandbuffer);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(tempCommandbuffer, &beginInfo);

  VkBufferCopy copy{};
  copy.srcOffset = 0;
  copy.dstOffset = 0;
  copy.size = size;
  vkCmdCopyBuffer(tempCommandbuffer, srcBuffer, dstBuffer, 1, &copy);

  vkEndCommandBuffer(tempCommandbuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &tempCommandbuffer;

  vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(graphicsQueue);

  vkFreeCommandBuffers(device, commandPool, 1, &tempCommandbuffer);
}

void Renderer::createVertexBuffer(const std::vector<Vertex> *vertices,
                                  VkBuffer &vertexBuffer,
                                  VkDeviceMemory &vertexBufferMemory) {
  VkDeviceSize bufferSize = sizeof(Vertex) * vertices->size();

  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;
  createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               stagingBuffer, stagingBufferMemory);

  void *data;
  vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
  memcpy(data, vertices->data(), bufferSize);
  vkUnmapMemory(device, stagingBufferMemory);

  createBuffer(
      bufferSize,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

  copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

  vkDestroyBuffer(device, stagingBuffer, nullptr);
  vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void Renderer::createIndexBuffer(const std::vector<uint16_t> *indices,
                                 VkBuffer &indexBuffer,
                                 VkDeviceMemory &indexBufferMemory) {
  VkDeviceSize bufferSize = sizeof(uint16_t) * indices->size();

  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;
  createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               stagingBuffer, stagingBufferMemory);

  void *data;
  vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
  memcpy(data, indices->data(), bufferSize);
  vkUnmapMemory(device, stagingBufferMemory);

  createBuffer(
      bufferSize,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

  copyBuffer(stagingBuffer, indexBuffer, bufferSize);

  vkDestroyBuffer(device, stagingBuffer, nullptr);
  vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void Renderer::createSyncObjects() {

  imageAvailableSemaphores.resize(MAX_FRAME_IN_FLIGHT);
  renderFinishedSemaphores.resize(swapchainImages.size());
  inFlightFences.resize(MAX_FRAME_IN_FLIGHT);

  VkSemaphoreCreateInfo semaphoreCreateInfo{};
  semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (int i = 0; i < MAX_FRAME_IN_FLIGHT; i++) {
    chk(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr,
                          &imageAvailableSemaphores[i]),
        "Failed to create image_available semaphore!");
    chk(vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]),
        "Failed to create fence!");
    ;
  }

  for (int i = 0; i < swapchainImages.size(); i++) {
    chk(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr,
                          &renderFinishedSemaphores[i]),
        "Failed to create render_finished semaphore!");
  }

  printf("Created semaphore / fence!\n");
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
  int width, height;
  glfwGetFramebufferSize(window, &width, &height);
  while (width == 0 || height == 0) {
    if (glfwWindowShouldClose(window))
      return;
    glfwGetFramebufferSize(window, &width, &height);
    glfwPollEvents();
  }

  vkDeviceWaitIdle(device);
  cleanupSwapchain();

  if (imageIndex != -1) {
    vkDestroySemaphore(device, renderFinishedSemaphores[imageIndex], nullptr);
    VkSemaphoreCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    vkCreateSemaphore(device, &createInfo, nullptr, &renderFinishedSemaphores[imageIndex]);
  }

  createSwapchain();
  createImageViews();
  createFramebuffer();
}

void Renderer::recordCommandBuffer(VkCommandBuffer commandbuffer,
                                   uint32_t imageIndex, const Scene &scene) {
  VkCommandBufferBeginInfo cmdBufBeginInfo{};
  cmdBufBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  chk(vkBeginCommandBuffer(commandbuffer, &cmdBufBeginInfo),
      "Failed to begin command buffer!");

  VkClearValue clearColor{};
  clearColor.color = {{.0f, .0f, .0f, 1.0f}};

  VkRenderPassBeginInfo renderpassBeginInfo{};
  renderpassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderpassBeginInfo.renderPass = renderpass;
  renderpassBeginInfo.renderArea = {.offset = {0, 0}, .extent = imageExtent};
  renderpassBeginInfo.framebuffer = framebuffers[imageIndex];
  renderpassBeginInfo.clearValueCount = 1;
  renderpassBeginInfo.pClearValues = &clearColor;

  vkCmdBeginRenderPass(commandbuffer, &renderpassBeginInfo,
                       VK_SUBPASS_CONTENTS_INLINE);
  vkCmdBindPipeline(commandbuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    graphicsPipeline);

  VkViewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = static_cast<float>(imageExtent.width);
  viewport.height = static_cast<float>(imageExtent.height);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(commandbuffer, 0, 1, &viewport);

  VkRect2D scissors{};
  scissors.extent = imageExtent;
  scissors.offset = {0, 0};
  vkCmdSetScissor(commandbuffer, 0, 1, &scissors);

  // compute projction matrix
  glm::mat4 projection =
      glm::ortho(-static_cast<float>(imageExtent.width) * 0.5f,  // left
                 static_cast<float>(imageExtent.width) * 0.5f,   // right
                 -static_cast<float>(imageExtent.height) * 0.5f, // bottom
                 static_cast<float>(imageExtent.height) * 0.5f,  // top
                 -1.0f, 1.0f                                     // near, far
      );

  projection[1][1] *= -1;

  // draw rectangles
  for (const RectObject &obj : scene.rects) {

    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandbuffer, 0, 1, &scene.rectMesh.vertexBuffer,
                           offsets);
    vkCmdBindIndexBuffer(commandbuffer, scene.rectMesh.indexBuffer, 0,
                         VK_INDEX_TYPE_UINT16);

    float magnitude = glm::length(obj.net_force);

    float angle = 0.0f;
    // if (magnitude > 1e-8f) {
    angle = atan2(obj.net_force.y, obj.net_force.x);
    // }

    float sx = std::clamp<float>(log(magnitude * 200.0f), 0.5f, 7.0f);
    // float sy = 1 / sx;
    float sy = 1;

    glm::mat4 model(1.0f);
    model = glm::translate(model, glm::vec3(obj.position, 0.0f));
    model = glm::rotate(model, angle,
                        glm::vec3(0.0f, 0.0f, 1.0f)); // rotate over Z axis
    model = glm::scale(model, glm::vec3(sx, sy, 1.0f));

    glm::mat4 mvp = projection * model;

    glm::vec3 color = {1.0f, 1.0f, 1.0f};
    PushConstantData push{.mvp = mvp, .color = color};

    vkCmdPushConstants(commandbuffer, pipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(PushConstantData),
                       &push);
    vkCmdDrawIndexed(commandbuffer, scene.rectMesh.indexCount, 1, 0, 0, 0);
  }

  // draw circles
  for (const CircleObject &obj : scene.circles) {

    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandbuffer, 0, 1, &scene.circleMesh.vertexBuffer,
                           offsets);
    vkCmdBindIndexBuffer(commandbuffer, scene.circleMesh.indexBuffer, 0,
                         VK_INDEX_TYPE_UINT16);

    glm::mat4 model(1.0f);
    model = glm::translate(model, glm::vec3(obj.position, 0.0f));
    model = glm::scale(model, glm::vec3(obj.radius, obj.radius, 1.0f));

    glm::mat4 mvp = projection * model;

    PushConstantData push{.mvp = mvp, .color = obj.color};
    vkCmdPushConstants(commandbuffer, pipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantData),
                       &push);
    vkCmdDrawIndexed(commandbuffer, scene.circleMesh.indexCount, 1, 0, 0, 0);
  }

  vkCmdEndRenderPass(commandbuffer);
  vkEndCommandBuffer(commandbuffer);
}

void Renderer::drawFrame(const Scene &scene) {
  // printf("draw\n");

  vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE,
                  UINT64_MAX);

  uint32_t imageIndex;
  VkResult res = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
                                       imageAvailableSemaphores[currentFrame],
                                       nullptr, &imageIndex);
  // semaphore is signaled when vkAcquireNextImageKHR returns "VK_SUBOPTIMAL_KHR"
  //  semaphore is not signaled when vkAcquireNextImageKHR returns "VK_ERROR_OUT_OF_DATE_KHR"
  if (res == VK_ERROR_OUT_OF_DATE_KHR) { 
    recreateSwapchain(-1);
    return;
  } else if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR) {
    throw std::runtime_error("Failed to acquire swapchain image!");
  }

  vkResetFences(device, 1, &inFlightFences[currentFrame]);

  vkResetCommandBuffer(commandBuffers[currentFrame], 0);
  recordCommandBuffer(commandBuffers[currentFrame], imageIndex, scene);

  VkPipelineStageFlags waitStages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = &imageAvailableSemaphores[currentFrame];
  submitInfo.pWaitDstStageMask = waitStages;
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = &renderFinishedSemaphores[imageIndex];

  res = vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]);
  if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR || framebufferResized) {
    framebufferResized = false;
    recreateSwapchain(imageIndex);
    return;
  }

  VkPresentInfoKHR presentInfo{};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = &swapchain;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &renderFinishedSemaphores[imageIndex];
  presentInfo.pImageIndices = &imageIndex;

  vkQueuePresentKHR(presentQueue, &presentInfo);

  currentFrame = (currentFrame + 1) % MAX_FRAME_IN_FLIGHT;
}

// --------------- NON-MEMBER FUNCTIONS --------------- //

void chk(VkResult res, const char *msg) {
  if (res != VK_SUCCESS) {
    throw std::runtime_error(msg);
  }
}

const char *
getDebugSeverityStr(VkDebugUtilsMessageSeverityFlagBitsEXT Severity) {
  switch (Severity) {
  case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
    return "Verbose";

  case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
    return "Info";

  case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
    return "Warning";

  case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
    return "Error";

  default:
    printf("Invalid severity code %d\n", Severity);
    throw std::runtime_error("");

    return "No such severity!";
  }
}

const char *getDebugType(VkDebugUtilsMessageTypeFlagsEXT Type) {
  switch (Type) {
  case VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT:
    return "General";

  case VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT:
    return "Validation";

  case VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT:
    return "Performance";

  case VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT:
    return "Device address binding";

  default:
    printf("Invalid type code %d\n", Type);
    throw std::runtime_error("");
  }
  return "No such type!";
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT Severity,
    VkDebugUtilsMessageTypeFlagsEXT Type,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *userData) {
  printf("[Info]\nDebug callback: %s\n", pCallbackData->pMessage);
  printf("Severity : %s\n", getDebugSeverityStr(Severity));
  printf("Type : %s\n", getDebugType(Type));
  printf("Objects : ");

  for (uint32_t i = 0; i < pCallbackData->objectCount; i++) {
    printf("%llx ", pCallbackData->pObjects[i].objectHandle);
  }
  printf("\n\n");

  return VK_FALSE;
}

void populateDebugMessenger(
    VkDebugUtilsMessengerCreateInfoEXT &debugMessengerCreateInfo) {
  debugMessengerCreateInfo.sType =
      VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;

  debugMessengerCreateInfo.messageSeverity =
      // VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
      // VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

  debugMessengerCreateInfo.messageType =
      VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

  debugMessengerCreateInfo.pfnUserCallback = debugCallback;
}

VkResult
createDebugUtilsMessenger(VkInstance &instance,
                          const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
                          const VkAllocationCallbacks *pAllocator,
                          VkDebugUtilsMessengerEXT *pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

std::vector<char> readFile(const char *filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file!");
  }

  size_t fileSize = (size_t)file.tellg();
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);

  file.close();

  return buffer;
}
