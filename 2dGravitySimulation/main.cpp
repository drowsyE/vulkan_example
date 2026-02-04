#define GLFW_INCLUDE_VULKAN
#include <glfw/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

#include "renderer.h"

#define WINDOW_WIDTH 1000
#define WINDOW_HEIGHT 800

#define NUM_RECT_COLS 10
#define NUM_RECT_ROWS 10
#define NUM_CIRCLES 2
#define NUM_CIRCLE_SIDES 32
#define CIRCLE_RADIUS 50.0f
#define GRAVITY 900
#define DT 0.7

#define TARGET_FRAME_TIME 1.0 / 60.0

// -------- Rectangle object. Indicates net force --------
std::vector<Vertex> rect_vertices = {
    {{-7.0f, -5.0f}},
    {{7.0f, -5.0f}}, 
    {{7.0f, 5.0f}},
    {{-7.0f, 5.0f}}
};
std::vector<uint16_t> rect_indices = {0, 1, 2, 2, 3, 0}; // ccw order
// ---------------------------------------------- //

// ---------- Circle Object ---------------------------
std::vector<Vertex> circle_vertices{};
std::vector<uint16_t> circle_indices{};

void createCircleObject(int n_sides) {
  circle_vertices.clear();
  circle_indices.clear();

  float angle;
  for (int i = 0; i < n_sides; i++) {
    angle = glm::two_pi<float>() / n_sides * i;
    circle_vertices.push_back({
        {glm::cos(angle), glm::sin(angle)}
    });
  }

  // add origin
  uint16_t origin_idx = n_sides;
  circle_vertices.push_back({{0.0f, 0.0f}}); // {1.0f, 0.0f, 0.0f}});

  // generate index
  for (int i = 0; i < n_sides; i++) {
    circle_indices.push_back(origin_idx);        
    circle_indices.push_back(i);                 
    circle_indices.push_back((i + 1) % n_sides);
  }
}
// ---------------------------------------------------

enum ShapeType { SHAPE_TYPE_RECTANGLE = 0, SHAPE_TYPE_CIRCLE };

void initMeshBuffers(Renderer &renderer, Mesh &mesh, ShapeType shapeType) {
  switch (shapeType) {

  case SHAPE_TYPE_RECTANGLE:
    renderer.createVertexBuffer(&rect_vertices, mesh.vertexBuffer,
                                mesh.vertexBufferMemory);
    renderer.createIndexBuffer(&rect_indices, mesh.indexBuffer,
                               mesh.indexBufferMemory);
    mesh.indexCount = rect_indices.size();
    break;

  case SHAPE_TYPE_CIRCLE:
    renderer.createVertexBuffer(&circle_vertices, mesh.vertexBuffer,
                                mesh.vertexBufferMemory);
    renderer.createIndexBuffer(&circle_indices, mesh.indexBuffer,
                               mesh.indexBufferMemory);
    mesh.indexCount = circle_indices.size();
    break;

  default:
    throw std::runtime_error("Failed to initialize mesh buffers!");
  }
}

class GravitySystem {

public:
  GravitySystem(float g, float dt) : gravity(g), dt(dt) {}

  void initVectorFieldComponent(std::vector<RectObject> &rects,
                                Renderer &renderer) {
    const float width = static_cast<float>(renderer.imageExtent.width);
    const float height = static_cast<float>(renderer.imageExtent.height);

    const float dx = width / NUM_RECT_COLS;
    const float dy = height / NUM_RECT_ROWS;

    const float startX = -width * 0.5f + dx * 0.5f;
    const float startY = -height * 0.5f + dy * 0.5f;

    for (int row = 0; row < NUM_RECT_ROWS; row++) {
      for (int col = 0; col < NUM_RECT_COLS; col++) {
        int idx = row * NUM_RECT_COLS + col;

        rects[idx].position = glm::vec2(startX + col * dx, startY + row * dy);

        rects[idx].net_force = glm::vec2(0.0f);
      }
    }
  }

  void initCircles(std::vector<CircleObject> &circles, Renderer &renderer) {
    const float width = static_cast<float>(renderer.imageExtent.width);
    const float height = static_cast<float>(renderer.imageExtent.height);

    for (int i = 0; i < circles.size(); i++) {
      circles[i].position = glm::vec2(
          -width * 0.25f + (i * width * 0.5f),
          0.0f);
      circles[i].velocity =
          glm::vec2((i % 2 == 0) ? 5.0f : -5.0f, 
                    (i % 2 == 0) ? 3.0f : -3.0f);
      circles[i].net_force = glm::vec2(0.0f);
      circles[i].mass = 100.0f;
      circles[i].radius = CIRCLE_RADIUS;
      circles[i].color = {1.0f, 1.0f, 1.0f};
    }
  }

  // Initialize circles manually
  void initCircles2(std::vector<CircleObject> &circles, Renderer &renderer) {
    // const float width = static_cast<float>(renderer.imageExtent.width);
    // const float height = static_cast<float>(renderer.imageExtent.height);

    circles[0] = (CircleObject) {
        .position = glm::vec2(-400.0f, -400.0f * sqrt(3) / 3),
        .velocity = glm::vec2(10.0f, 0.0f),
        .net_force = glm::vec2(0.0f),
        .color = glm::vec3(1.0f, 0.0f, 0.0f),
        .mass = 100.0f,
        .radius = 50.0f
    };

    circles[1] = (CircleObject) {
        .position = glm::vec2(400.0f, -400.0f * sqrt(3) / 3),
        .velocity = glm::vec2(-5.0f, 5.0f * sqrt(3)),
        .net_force = glm::vec2(0.0f),
        .color = glm::vec3(0.0f, 1.0f, 0.0f),
        .mass = 100.0f,
        .radius = 50.0f
    };

    circles[2] = (CircleObject) {
        .position = glm::vec2(0.0f, 400.0f * sqrt(3) * 2/3),
        .velocity = glm::vec2(-5.0f, -5.0f * sqrt(3)),
        .net_force = glm::vec2(0.0f),
        .color = glm::vec3(0.0f, 0.0f, 1.0f),
        .mass = 100.0f,
        .radius = 50.0f
    };

  }

  void update(Renderer &renderer, std::vector<RectObject> &rects,
              std::vector<CircleObject> &circles) {
    updateCircle(circles);
    collision(circles);
    updateVectorFieldComponent(rects, circles);
  }

private:
  float gravity;
  float dt;

  void updateCircle(std::vector<CircleObject> &circles) {

    // Compute gravitational force between n-circles
    for (int i = 0; i < circles.size(); i++) {
      circles[i].net_force = glm::vec2(0.0f);

      for (int j = 0; j < circles.size(); j++) {
        if (i == j)
          continue;

        glm::vec2 r = circles[j].position - circles[i].position;
        float len = glm::length(r);
        float dist = len * len + 1e-6f; // softening
        float invDist = glm::inversesqrt(dist);

        glm::vec2 dir = r * invDist;

        float force_mag =
            gravity * circles[i].mass * circles[j].mass * invDist * invDist;

        circles[i].net_force += force_mag * dir;
      }
    }

    // update position
    for (CircleObject &circle : circles) {
      glm::vec2 acc = circle.net_force / circle.mass;
      circle.velocity += acc * dt;
      circle.position += circle.velocity * dt;
    }
  }

  void updateVectorFieldComponent(std::vector<RectObject> &rects,
                                  std::vector<CircleObject> &circles) {
    for (RectObject &rect : rects) {
      rect.net_force = glm::vec2(0.0f);
      for (CircleObject &circle : circles) {
        glm::vec2 r = circle.position - rect.position;

        float len = glm::length(r);
        float dist = len * len + 1e-6f; // softening
        float invDist = glm::inversesqrt(dist);
        glm::vec2 dir = r * invDist;

        float force_mag = gravity * 1 * circle.mass * invDist * invDist;

        rect.net_force += force_mag * dir;
      }
    }
  }

  void collision(std::vector<CircleObject>& circles) {
    for (int i = 0; i < circles.size(); i++) {
        for (int j = i + 1; j < circles.size(); j++) {
            glm::vec2 d = circles[j].position - circles[i].position; // i -> j
            float dist2 = glm::dot(d, d);
            float r = circles[i].radius + circles[j].radius;

            if (dist2 > r * r) continue;

            float dist = sqrt(dist2);
            glm::vec2 n = d / dist;
            glm::vec2 rv = circles[j].velocity - circles[i].velocity;
            float vn = glm::dot(rv, n);

            if (vn > 0.0f) continue;

            float e = 0.99999;

            float j_impulse =
                -(1.0f + e) * vn /
                (1.0f / circles[i].mass + 1.0f / circles[j].mass);

            glm::vec2 impulse = j_impulse * n;

            circles[i].velocity -= impulse / circles[i].mass;
            circles[j].velocity += impulse / circles[j].mass;

        }
    }
  }
};



int main() {

  Renderer renderer(WINDOW_WIDTH, WINDOW_HEIGHT);
  GravitySystem system(GRAVITY, DT);

  std::vector<RectObject> rects(NUM_RECT_COLS * NUM_RECT_ROWS);
  std::vector<CircleObject> circles(NUM_CIRCLES);

  Mesh rectMesh;
  Mesh circleMesh;
  createCircleObject(NUM_CIRCLE_SIDES);

  initMeshBuffers(renderer, rectMesh, SHAPE_TYPE_RECTANGLE);
  initMeshBuffers(renderer, circleMesh, SHAPE_TYPE_CIRCLE);

  system.initVectorFieldComponent(rects, renderer);
  system.initCircles(circles, renderer);
  // system.initCircles2(circles, renderer);

  Scene scene{.circles = circles,
              .rects = rects,
              .circleMesh = circleMesh,
              .rectMesh = rectMesh};

  using clock = std::chrono::high_resolution_clock;
  auto lastTime = clock::now();

  while (!glfwWindowShouldClose(renderer.window)) {

    auto frameStart = clock::now();

    glfwPollEvents();

    system.update(renderer, scene.rects, scene.circles);

    renderer.drawFrame(scene);

    // FPS limit
    auto frameEnd = clock::now();
    std::chrono::duration<double> elapsed = frameEnd - frameStart;

    double sleepTime = TARGET_FRAME_TIME - elapsed.count();
    if (sleepTime > 0.0) {
      std::this_thread::sleep_for(std::chrono::duration<double>(sleepTime));
    }
  }
  vkDeviceWaitIdle(renderer.device);

  vkDestroyBuffer(renderer.device, rectMesh.vertexBuffer, nullptr);
  vkFreeMemory(renderer.device, rectMesh.vertexBufferMemory, nullptr);
  vkDestroyBuffer(renderer.device, rectMesh.indexBuffer, nullptr);
  vkFreeMemory(renderer.device, rectMesh.indexBufferMemory, nullptr);

  vkDestroyBuffer(renderer.device, circleMesh.vertexBuffer, nullptr);
  vkFreeMemory(renderer.device, circleMesh.vertexBufferMemory, nullptr);
  vkDestroyBuffer(renderer.device, circleMesh.indexBuffer, nullptr);
  vkFreeMemory(renderer.device, circleMesh.indexBufferMemory, nullptr);
}