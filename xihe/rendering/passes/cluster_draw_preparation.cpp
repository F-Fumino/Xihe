#include "cluster_draw_preparation.h"

#include "common/timer.h"

namespace xihe::rendering
{

ClusterDrawPreparationPass::ClusterDrawPreparationPass(GpuLoDScene &gpu_lod_scene, sg::Camera &camera) :
    gpu_lod_scene_(gpu_lod_scene),
    camera_{camera}
{}

void ClusterDrawPreparationPass::execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables)
{
	Timer cluster_preparation_timer;
	cluster_preparation_timer.start();

	auto &resource_cache     = command_buffer.get_device().get_resource_cache();
	auto &comp_shader_module = resource_cache.request_shader_module(vk::ShaderStageFlagBits::eCompute, get_compute_shader());

	std::vector<backend::ShaderModule *> shader_modules = {&comp_shader_module};

	auto &pipeline_layout = resource_cache.request_pipeline_layout(shader_modules);
	command_buffer.bind_pipeline_layout(pipeline_layout);

	gpu_lod_scene_.get_draw_counts_buffer().update(std::vector<uint32_t>{0});
	gpu_lod_scene_.get_counts_buffer().update(std::vector<uint32_t>(2, 0));

	command_buffer.bind_buffer(gpu_lod_scene_.get_cluster_visibility_buffer(), 0, gpu_lod_scene_.get_cluster_visibility_buffer().get_size(), 0, 1, 0);
	command_buffer.bind_buffer(gpu_lod_scene_.get_scene_data_buffer_address(), 0, gpu_lod_scene_.get_scene_data_buffer_address().get_size(), 0, 2, 0);
	command_buffer.bind_buffer(gpu_lod_scene_.get_cluster_group_buffer(), 0, gpu_lod_scene_.get_cluster_group_buffer().get_size(), 0, 3, 0);
	command_buffer.bind_buffer(gpu_lod_scene_.get_cluster_buffer_address(), 0, gpu_lod_scene_.get_cluster_buffer_address().get_size(), 0, 4, 0);

	command_buffer.bind_buffer(gpu_lod_scene_.get_draw_counts_buffer(), 0, gpu_lod_scene_.get_draw_counts_buffer().get_size(), 0, 5, 0);
	command_buffer.bind_buffer(gpu_lod_scene_.get_counts_buffer(), 0, gpu_lod_scene_.get_counts_buffer().get_size(), 0, 6, 0);
	command_buffer.bind_buffer(gpu_lod_scene_.get_indirect_command_buffer(), 0, gpu_lod_scene_.get_indirect_command_buffer().get_size(), 0, 7, 0);
	command_buffer.bind_buffer(gpu_lod_scene_.get_global_index_buffer_address(), 0, gpu_lod_scene_.get_global_index_buffer_address().get_size(), 0, 8, 0);

	command_buffer.push_constants(gpu_lod_scene_.get_cluster_count());

	command_buffer.dispatch((gpu_lod_scene_.get_cluster_count() + 31) / 32, 1, 1);

	auto cluster_preparation_time = cluster_preparation_timer.stop();
	LOGI("Cluster draw preparation time: {} ms", cluster_preparation_time * 1000.0f);
}

}        // namespace xihe::rendering
