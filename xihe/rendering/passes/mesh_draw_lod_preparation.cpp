#include "mesh_draw_lod_preparation.h"

namespace xihe::rendering
{
namespace
{
glm::vec4 normalize_plane(const glm::vec4 &plane)
{
	float length = glm::length(glm::vec3(plane.x, plane.y, plane.z));
	return plane / length;
}
}        // namespace

MeshDrawLoDPreparationPass::MeshDrawLoDPreparationPass(GpuLoDScene &gpu_lod_scene, sg::Camera &camera) :
    gpu_lod_scene_(gpu_lod_scene),
    camera_{camera}
{}

void MeshDrawLoDPreparationPass::execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables)
{
	auto &resource_cache     = command_buffer.get_device().get_resource_cache();
	auto &comp_shader_module = resource_cache.request_shader_module(vk::ShaderStageFlagBits::eCompute, get_compute_shader());

	std::vector<backend::ShaderModule *> shader_modules = {&comp_shader_module};

	auto &pipeline_layout = resource_cache.request_pipeline_layout(shader_modules);
	command_buffer.bind_pipeline_layout(pipeline_layout);

	MeshSceneUniform global_uniform{};
	global_uniform.camera_view_proj = camera_.get_pre_rotation() * vulkan_style_projection(camera_.get_projection()) * camera_.get_view();

	global_uniform.camera_position = glm::vec3((glm::inverse(camera_.get_view())[3]));

	//if (freeze_frustum_)
	//{
	//	global_uniform.view              = frozen_view_;
	//	global_uniform.frustum_planes[0] = frozen_frustum_planes_[0];
	//	global_uniform.frustum_planes[1] = frozen_frustum_planes_[1];
	//	global_uniform.frustum_planes[2] = frozen_frustum_planes_[2];
	//	global_uniform.frustum_planes[3] = frozen_frustum_planes_[3];
	//	global_uniform.frustum_planes[4] = frozen_frustum_planes_[4];
	//	global_uniform.frustum_planes[5] = frozen_frustum_planes_[5];
	//}
	//else
	{
		global_uniform.view              = camera_.get_view();
		glm::mat4 m                      = glm::transpose(camera_.get_pre_rotation() * camera_.get_projection());
		global_uniform.frustum_planes[0] = normalize_plane(m[3] + m[0]);
		global_uniform.frustum_planes[1] = normalize_plane(m[3] - m[0]);
		global_uniform.frustum_planes[2] = normalize_plane(m[3] + m[1]);
		global_uniform.frustum_planes[3] = normalize_plane(m[3] - m[1]);
		global_uniform.frustum_planes[4] = normalize_plane(m[3] + m[2]);
		global_uniform.frustum_planes[5] = normalize_plane(m[3] - m[2]);
	}

	auto allocation = active_frame.allocate_buffer(vk::BufferUsageFlagBits::eUniformBuffer, sizeof(MeshSceneUniform), thread_index_);

	allocation.update(global_uniform);

	command_buffer.bind_buffer(allocation.get_buffer(), allocation.get_offset(), allocation.get_size(), 0, 1, 0);

	command_buffer.bind_buffer(gpu_lod_scene_.get_mesh_draws_buffer(), 0, gpu_lod_scene_.get_mesh_draws_buffer().get_size(), 0, 2, 0);
	command_buffer.bind_buffer(gpu_lod_scene_.get_instance_buffer(), 0, gpu_lod_scene_.get_instance_buffer().get_size(), 0, 3, 0);

	// command_buffer.bind_buffer(input_bindables[0].buffer(), 0, input_bindables[0].buffer().get_size(), 0, 3, 0);
	command_buffer.bind_buffer(gpu_lod_scene_.get_draw_command_buffer(), 0, gpu_lod_scene_.get_draw_command_buffer().get_size(), 0, 4, 0);
	command_buffer.bind_buffer(gpu_lod_scene_.get_draw_counts_buffer(), 0, gpu_lod_scene_.get_draw_counts_buffer().get_size(), 0, 5, 0);
	command_buffer.bind_buffer(gpu_lod_scene_.get_mesh_bounds_buffer(), 0, gpu_lod_scene_.get_mesh_bounds_buffer().get_size(), 0, 6, 0);


	command_buffer.dispatch((gpu_lod_scene_.get_instance_count() + 255)/256, 1, 1);
}

}        // namespace xihe::rendering
