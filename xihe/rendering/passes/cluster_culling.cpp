#include "cluster_culling.h"

namespace xihe::rendering
{

namespace
{
glm::vec4 normalize_plane(const glm::vec4 &plane)
{
	float length = glm::length(glm::vec3(plane.x, plane.y, plane.z));
	return plane / length;
}

vk::SamplerCreateInfo get_nearest_sampler()
{
	auto sampler_info         = vk::SamplerCreateInfo{};
	sampler_info.addressModeU = vk::SamplerAddressMode::eClampToEdge;
	sampler_info.addressModeV = vk::SamplerAddressMode::eClampToEdge;
	sampler_info.addressModeW = vk::SamplerAddressMode::eClampToEdge;
	sampler_info.minFilter    = vk::Filter::eNearest;
	sampler_info.magFilter    = vk::Filter::eNearest;
	sampler_info.mipmapMode   = vk::SamplerMipmapMode::eNearest;
	sampler_info.maxLod       = VK_LOD_CLAMP_NONE;

	return sampler_info;
}
}        // namespace

ClusterCullingPass::ClusterCullingPass(GpuLoDScene &gpu_scene, sg::Camera &camera) :

    gpu_scene_{gpu_scene},
    camera_{camera}
{}

void ClusterCullingPass::execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables)
{
	auto &resource_cache     = command_buffer.get_device().get_resource_cache();
	auto &comp_shader_module = resource_cache.request_shader_module(vk::ShaderStageFlagBits::eCompute, get_compute_shader(), shader_variant_);

	std::vector<backend::ShaderModule *> shader_modules = {&comp_shader_module};

	auto &pipeline_layout = resource_cache.request_pipeline_layout(shader_modules);
	command_buffer.bind_pipeline_layout(pipeline_layout);

	MeshSceneUniform global_uniform{};
	global_uniform.camera_view_proj = camera_.get_pre_rotation() * vulkan_style_projection(camera_.get_projection()) * camera_.get_view();

	global_uniform.camera_position = glm::vec3((glm::inverse(camera_.get_view())[3]));

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

	command_buffer.bind_buffer(allocation.get_buffer(), allocation.get_offset(), allocation.get_size(), 0, 2, 0);

	command_buffer.bind_buffer(gpu_scene_.get_mesh_draws_buffer(), 0, gpu_scene_.get_mesh_draws_buffer().get_size(), 0, 3, 0);
	command_buffer.bind_buffer(gpu_scene_.get_instance_buffer(), 0, gpu_scene_.get_instance_buffer().get_size(), 0, 4, 0);
	command_buffer.bind_buffer(gpu_scene_.get_instance_visibility_buffer(), 0, gpu_scene_.get_instance_visibility_buffer().get_size(), 0, 5, 0);

	command_buffer.bind_buffer(gpu_scene_.get_scene_data_buffer_address(), 0, gpu_scene_.get_scene_data_buffer_address().get_size(), 0, 6, 0);
	command_buffer.bind_buffer(gpu_scene_.get_page_state_buffer(), 0, gpu_scene_.get_page_state_buffer().get_size(), 0, 7, 0);

	command_buffer.bind_buffer(gpu_scene_.get_cluster_group_buffer(), 0, gpu_scene_.get_cluster_group_buffer().get_size(), 0, 8, 0);
	command_buffer.bind_buffer(gpu_scene_.get_cluster_buffer(), 0, gpu_scene_.get_cluster_buffer().get_size(), 0, 9, 0);

	command_buffer.bind_buffer(gpu_scene_.get_valid_data_size_buffer(), 0, gpu_scene_.get_valid_data_size_buffer().get_size(), 0, 10, 0);

	gpu_scene_.get_counts_buffer().update(std::vector<uint32_t>(3, 0));

	command_buffer.bind_buffer(gpu_scene_.get_counts_buffer(), 0, gpu_scene_.get_counts_buffer().get_size(), 0, 11, 0);
	command_buffer.bind_buffer(gpu_scene_.get_indirect_command_buffer(), 0, gpu_scene_.get_indirect_command_buffer().get_size(), 0, 12, 0);

	command_buffer.bind_buffer(gpu_scene_.get_global_index_buffer(), 0, gpu_scene_.get_global_index_buffer().get_size(), 0, 13, 0);

	gpu_scene_.get_recheck_list_buffer().update(std::vector<int32_t>(gpu_scene_.get_cluster_count(), -1));
	command_buffer.bind_buffer(gpu_scene_.get_recheck_list_buffer(), 0, gpu_scene_.get_recheck_list_buffer().get_size(), 0, 14, 0);

	auto &hzb_view = input_bindables[0].image_view();

	command_buffer.bind_image(hzb_view, resource_cache.request_sampler(get_nearest_sampler()), 0, 15, 0);

	OcclusionUniform uniform;
	uniform.width          = hzb_view.get_image().get_extent().width;
	uniform.height         = hzb_view.get_image().get_extent().height;
	uniform.mip_count      = hzb_view.get_image().get_mip_levels();
	uniform.is_first_frame = is_first_frame_;
	is_first_frame_        = false;
	uniform.depth_bias     = 1e-4f;

	auto allocation_occlusion = active_frame.allocate_buffer(
	    vk::BufferUsageFlagBits::eUniformBuffer,
	    sizeof(OcclusionUniform),
	    thread_index_);
	allocation_occlusion.update(uniform);
	command_buffer.bind_buffer(allocation_occlusion.get_buffer(), allocation_occlusion.get_offset(), allocation_occlusion.get_size(), 0, 16, 0);

	command_buffer.push_constants(gpu_scene_.get_lod_threshold());

	command_buffer.dispatch((gpu_scene_.get_cluster_count() + 31) / 32, 1, 1);
}

void ClusterCullingPass::use_lod(bool use)
{
	if (use == use_lod_)
	{
		return;
	}
	use_lod_ = use;
	if (use)
	{
		shader_variant_.add_define("USE_LOD");
	}
	else
	{
		shader_variant_.remove_define("USE_LOD");
	}
}

void ClusterCullingPass::use_occlusion(bool use)
{
	if (use == use_occlusion_)
	{
		return;
	}
	use_occlusion_ = use;
	if (use)
	{
		shader_variant_.add_define("USE_OCCLUSION");
	}
	else
	{
		shader_variant_.remove_define("USE_OCCLUSION");
	}
}

}        // namespace xihe::rendering
