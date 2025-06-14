#pragma once

#include <unordered_map>

#include "common/error.h"
#include "common/glm_common.h"
#include "scene_graph/script.h"
namespace xihe::sg
{
	class CirclePathCamera:public NodeScript
	{
	  public:

	    CirclePathCamera(Node &node);

	    virtual ~CirclePathCamera() = default;

		void update(float delta_time) override;

		bool is_end() override;

		void input_event(const InputEvent &input_event) override;

	    void resize(uint32_t width, uint32_t height) override;

		void set_speed_multiplier(float speed_multiplier);

		void set_center(const glm::vec3 &center);

		void set_radius(float radius);

		void set_rotation_axis(const glm::vec3 &rotation_axis);

	  private:
	    glm::vec3 center_ = glm::vec3(0.0f, 0.0f, 0.0f);
	    float     radius_{0.0f};
	    glm::vec3 rotation_axis_ = glm::vec3(0.0f, 1.0f, 0.0f);
	    glm::vec3 u_ = glm::vec3(1.0f, 0.0f, 0.0f); // 两个正交基向量
	    glm::vec3 v_ = glm::vec3(0.0f, 0.0f, 1.0f);
	    float     angle_{0.0f};
	    float     speed_multiplier_{1.0f};
	    bool      is_end_{false};
	};

	class LinePathCamera : public NodeScript
    {
	  public:
	    LinePathCamera(Node &node);

	    virtual ~LinePathCamera() = default;

	    void update(float delta_time) override;

		bool is_end() override;

	    void input_event(const InputEvent &input_event) override;

	    void resize(uint32_t width, uint32_t height) override;

	    void set_speed_multiplier(float speed_multiplier);

	    void set_start(const glm::vec3 &start);

	    void set_end(const glm::vec3 &end);

	    void set_up_axis(const glm::vec3 &up);

	  private:
	    glm::vec3 start_   = glm::vec3(0.0f, 0.0f, 0.0f);
	    glm::vec3 end_     = glm::vec3(0.0f, 0.0f, 0.0f);
	    glm::vec3 up_axis_ = glm::vec3(0.0f, 0.0f, 0.0f);
	    float     progress_{0.0f};
	    float     speed_multiplier_{1.0f};
	    bool      is_end_{false};
    };
    }
