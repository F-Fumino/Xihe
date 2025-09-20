#include "geometry_mesh_pass.h"

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

GeometryMeshPass::GeometryMeshPass(GpuLoDScene &gpu_scene, sg::Camera &camera) :

    gpu_scene_{gpu_scene},
    camera_{camera}
{}

void GeometryMeshPass::execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables)
{
	RasterizationState rasterization_state;
	rasterization_state.polygon_mode = polygon_mode_;
	command_buffer.set_rasterization_state(rasterization_state);

	auto &resource_cache = command_buffer.get_device().get_resource_cache();

	auto &vert_shader_module = resource_cache.request_shader_module(vk::ShaderStageFlagBits::eVertex, get_vertex_shader(), shader_variant_);
	auto &frag_shader_module = resource_cache.request_shader_module(vk::ShaderStageFlagBits::eFragment, get_fragment_shader(), shader_variant_);

	std::vector<backend::ShaderModule *> shader_modules{&vert_shader_module, &frag_shader_module};

	DepthStencilState depth_stencil_state{};
	depth_stencil_state.depth_test_enable  = true;
	depth_stencil_state.depth_write_enable = true;

	command_buffer.set_depth_stencil_state(depth_stencil_state);

	auto &pipeline_layout = resource_cache.request_pipeline_layout(shader_modules, &resource_cache.request_bindless_descriptor_set());

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

	command_buffer.bind_buffer(allocation.get_buffer(), allocation.get_offset(), allocation.get_size(), 0, 1, 0);

	command_buffer.bind_buffer(gpu_scene_.get_indirect_command_buffer(), 0, gpu_scene_.get_indirect_command_buffer().get_size(), 0, 2, 0);
	command_buffer.bind_buffer(gpu_scene_.get_instance_buffer(), 0, gpu_scene_.get_instance_buffer().get_size(), 0, 3, 0);

	command_buffer.bind_buffer(gpu_scene_.get_scene_data_buffer_address(), 0, gpu_scene_.get_scene_data_buffer_address().get_size(), 0, 4, 0);
	command_buffer.bind_buffer(gpu_scene_.get_cluster_buffer(), 0, gpu_scene_.get_cluster_buffer().get_size(), 0, 5, 0);
	command_buffer.bind_buffer(gpu_scene_.get_cluster_group_buffer(), 0, gpu_scene_.get_cluster_group_buffer().get_size(), 0, 6, 0);

	command_buffer.bind_buffer(gpu_scene_.get_mesh_draws_buffer(), 0, gpu_scene_.get_mesh_draws_buffer().get_size(), 0, 7, 0);

	command_buffer.bind_index_buffer(gpu_scene_.get_global_index_buffer(), 0, vk::IndexType::eUint32);

	uint32_t *counts     = reinterpret_cast<uint32_t *>(gpu_scene_.get_counts_buffer().map());
	uint32_t  draw_count = counts[0];

	command_buffer.draw_indexed_indirect(gpu_scene_.get_indirect_command_buffer(), 0, gpu_scene_.get_cluster_count(), sizeof(IndirectDrawCommand));

	// command_buffer.draw_indexed_indirect(gpu_scene_.get_indirect_command_buffer(), 0, draw_count, sizeof(IndirectDrawCommand));
}

void GeometryMeshPass::show_meshlet_view(bool show)
{
	if (show == show_debug_view_)
	{
		return;
	}
	show_debug_view_ = show;

	if (show)
	{
		shader_variant_.add_define("SHOW_MESHLET_VIEW");
	}
	else
	{
		shader_variant_.remove_define("SHOW_MESHLET_VIEW");
	}
}

void GeometryMeshPass::show_lod_view(bool show)
{
	if (show == show_lod_view_)
	{
		return;
	}
	show_lod_view_ = show;
	if (show)
	{
		shader_variant_.add_define("SHOW_LOD_VIEW");
	}
	else
	{
		shader_variant_.remove_define("SHOW_LOD_VIEW");
	}
}

void GeometryMeshPass::show_line(bool show)
{
	polygon_mode_ = show ? vk::PolygonMode::eLine : vk::PolygonMode::eFill;
}

void GeometryMeshPass::show_texture()
{
	shader_variant_.add_define("HAS_BASE_COLOR_TEXTURE");
}
}        // namespace xihe::rendering
