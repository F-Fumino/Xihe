#include "fixed_path_camera.h"

#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/quaternion.hpp>

#include "scene_graph/components/camera.h"
#include "scene_graph/components/transform.h"
#include "scene_graph/node.h"

namespace xihe::sg
{
CirclePathCamera::CirclePathCamera(Node &node) :
    NodeScript{node, "CirclePathCamera"}
{}

void CirclePathCamera::update(float delta_time)
{
    // 计算圆上摄像机的位置
    angle_ += delta_time * speed_multiplier_;

    // 保证角度在0到2π之间
	is_end_ = false;
    if (angle_ > glm::two_pi<float>()) 
    {
        angle_ -= glm::two_pi<float>();
		is_end_ = true;
    }

    // 计算摄像机在圆上的位置
	float     cos_theta       = cos(angle_);
	float     sin_theta       = sin(angle_);
	glm::vec3 offset          = u_ * (radius_ * cos_theta) + v_ * (radius_ * sin_theta);
	glm::vec3 camera_position = center_ + offset;

    // 设置摄像机的变换
    auto &transform = get_node().get_component<Transform>();
	transform.set_translation(camera_position);

    // 始终朝向圆心
	glm::vec3 forward  = glm::normalize(center_ - camera_position);
	glm::quat rotation = glm::quatLookAt(forward, rotation_axis_);
	transform.set_rotation(rotation);
}

bool CirclePathCamera::is_end()
{
	return is_end_;
}

void CirclePathCamera::input_event(const InputEvent &input_event)
{
}

void CirclePathCamera::resize(uint32_t width, uint32_t height)
{
    auto &camera_node = get_node();

    if (camera_node.has_component<Camera>())
    {
        if (auto camera = dynamic_cast<PerspectiveCamera *>(&camera_node.get_component<Camera>()))
        {
            camera->set_aspect_ratio(static_cast<float>(width) / height);
        }
    }
}

void CirclePathCamera::set_speed_multiplier(float speed_multiplier)
{
    speed_multiplier_ = speed_multiplier;
}

void CirclePathCamera::set_center(const glm::vec3 &center)
{
    center_ = center;
}

void CirclePathCamera::set_radius(float radius)
{
    radius_ = radius;
}

void CirclePathCamera::set_rotation_axis(const glm::vec3 &rotation_axis)
{
	rotation_axis_ = glm::normalize(rotation_axis);

    glm::vec3 a;
	if (glm::abs(rotation_axis_.y) > 0.999f)
	{                                           // 接近Y轴
		a = glm::vec3(1.0f, 0.0f, 0.0f);        // 使用X轴作为参考
	}
	else
	{
		a = glm::vec3(0.0f, 1.0f, 0.0f);        // 使用Y轴作为参考
	}

	// 计算平面内的正交基向量u和v
	u_ = glm::normalize(glm::cross(a, rotation_axis_));
	v_ = glm::normalize(glm::cross(rotation_axis_, u_));
}

LinePathCamera::LinePathCamera(Node &node) :
    NodeScript{node, "LinePathCamera"}
{}

void LinePathCamera::update(float delta_time)
{
	// 更新进度（0到1之间循环）
	is_end_ = false;
	progress_ += delta_time * speed_multiplier_;
	if (progress_ >= 1.0f)
	{
		is_end_ = true;
	}
	progress_ = fmod(progress_, 1.0f);

	// 计算相机位置（线性插值）
	glm::vec3 camera_position = glm::mix(start_, end_, progress_);

	// 计算朝向方向（始终指向终点）
	glm::vec3 to_end  = end_ - camera_position;
	glm::vec3 forward = glm::normalize(to_end);

	// 处理终点重合的特殊情况
	if (glm::length(to_end) < 1e-6f)
	{
		forward = glm::normalize(end_ - start_);
	}

	// 计算旋转四元数
	glm::quat rotation = glm::quatLookAt(forward, up_axis_);

	// 设置变换
	auto &transform = get_node().get_component<Transform>();
	transform.set_translation(camera_position);
	transform.set_rotation(rotation);
}

bool LinePathCamera::is_end()
{
	return is_end_;
}

void LinePathCamera::input_event(const InputEvent &input_event)
{
	// 可添加输入处理逻辑
}

void LinePathCamera::resize(uint32_t width, uint32_t height)
{
	auto &camera_node = get_node();
	if (camera_node.has_component<Camera>())
	{
		if (auto camera = dynamic_cast<PerspectiveCamera *>(&camera_node.get_component<Camera>()))
		{
			camera->set_aspect_ratio(static_cast<float>(width) / height);
		}
	}
}

// 参数设置接口
void LinePathCamera::set_speed_multiplier(float speed_multiplier)
{
	speed_multiplier_ = speed_multiplier;
}

void LinePathCamera::set_start(const glm::vec3 &start)
{
	start_ = start;
}

void LinePathCamera::set_end(const glm::vec3 &end)
{
	end_ = end;
}

void LinePathCamera::set_up_axis(const glm::vec3 &up)
{
	up_axis_ = glm::normalize(up);
}

} // namespace xihe::sg
