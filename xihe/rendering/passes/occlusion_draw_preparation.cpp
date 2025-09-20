#include "occlusion_draw_preparation.h"

namespace xihe::rendering
{
OcclusionPreparationPass::OcclusionPreparationPass(GpuLoDScene &gpu_lod_scene) :
    gpu_lod_scene_(gpu_lod_scene)
{}

void OcclusionPreparationPass::execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables)
{
	auto &resource_cache     = command_buffer.get_device().get_resource_cache();
	auto &comp_shader_module = resource_cache.request_shader_module(vk::ShaderStageFlagBits::eCompute, get_compute_shader());

	std::vector<backend::ShaderModule *> shader_modules = {&comp_shader_module};

	auto &pipeline_layout = resource_cache.request_pipeline_layout(shader_modules);
	command_buffer.bind_pipeline_layout(pipeline_layout);

	gpu_lod_scene_.get_occlusion_counts_buffer().update(std::vector<uint32_t>{0});
	gpu_lod_scene_.get_recheck_counts_buffer().update(std::vector<uint32_t>{0});

	command_buffer.bind_buffer(gpu_lod_scene_.get_occlusion_counts_buffer(), 0, gpu_lod_scene_.get_occlusion_counts_buffer().get_size(), 0, 0, 0);
	command_buffer.bind_buffer(gpu_lod_scene_.get_occlusion_command_buffer(), 0, gpu_lod_scene_.get_occlusion_command_buffer().get_size(), 0, 1, 0);
	command_buffer.bind_buffer(gpu_lod_scene_.get_recheck_list_buffer(), 0, gpu_lod_scene_.get_recheck_list_buffer().get_size(), 0, 2, 0);
	command_buffer.bind_buffer(gpu_lod_scene_.get_recheck_counts_buffer(), 0, gpu_lod_scene_.get_recheck_counts_buffer().get_size(), 0, 3, 0);
	command_buffer.bind_buffer(gpu_lod_scene_.get_recheck_cluster_buffer(), 0, gpu_lod_scene_.get_recheck_cluster_buffer().get_size(), 0, 4, 0);

	command_buffer.dispatch((gpu_lod_scene_.get_cluster_count() + 255) / 256, 1, 1);
}

}        // namespace xihe::rendering
