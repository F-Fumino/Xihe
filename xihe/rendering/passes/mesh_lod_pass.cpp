#include "mesh_lod_pass.h"

namespace xihe::rendering
{

namespace
{
glm::vec4 normalize_plane(const glm::vec4 &plane)
{
	float length = glm::length(glm::vec3(plane.x, plane.y, plane.z));
	return plane / length;
}

vk::SamplerCreateInfo get_linear_sampler()
{
	auto sampler_info         = vk::SamplerCreateInfo{};
	sampler_info.addressModeU = vk::SamplerAddressMode::eClampToEdge;
	sampler_info.addressModeV = vk::SamplerAddressMode::eClampToEdge;
	sampler_info.addressModeW = vk::SamplerAddressMode::eClampToEdge;
	sampler_info.minFilter    = vk::Filter::eLinear;
	sampler_info.magFilter    = vk::Filter::eLinear;
	sampler_info.maxLod       = VK_LOD_CLAMP_NONE;

	return sampler_info;
}
}        // namespace

MeshLoDPass::MeshLoDPass(GpuLoDScene &gpu_scene, sg::Camera &camera) :

    gpu_scene_{gpu_scene},
    camera_{camera}
{}

void MeshLoDPass::execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables)
{
	command_buffer.set_has_mesh_shader(true);

	RasterizationState rasterization_state;
	rasterization_state.polygon_mode = polygon_mode_;
	command_buffer.set_rasterization_state(rasterization_state);

	auto &resource_cache = command_buffer.get_device().get_resource_cache();

	auto &task_shader_module = resource_cache.request_shader_module(vk::ShaderStageFlagBits::eTaskEXT, get_task_shader(), shader_variant_);
	auto &mesh_shader_module = resource_cache.request_shader_module(vk::ShaderStageFlagBits::eMeshEXT, get_mesh_shader(), shader_variant_);
	auto &frag_shader_module = resource_cache.request_shader_module(vk::ShaderStageFlagBits::eFragment, get_fragment_shader(), shader_variant_);

	std::vector<backend::ShaderModule *> shader_modules{&task_shader_module, &mesh_shader_module, &frag_shader_module};

	auto &pipeline_layout = resource_cache.request_pipeline_layout(shader_modules, &resource_cache.request_bindless_descriptor_set());
	command_buffer.bind_pipeline_layout(pipeline_layout);

	DepthStencilState depth_stencil_state{};
	depth_stencil_state.depth_test_enable  = true;
	depth_stencil_state.depth_write_enable = true;

	command_buffer.set_depth_stencil_state(depth_stencil_state);

	MeshSceneUniform global_uniform{};
	global_uniform.camera_view_proj = camera_.get_pre_rotation() * vulkan_style_projection(camera_.get_projection()) * camera_.get_view();

	global_uniform.camera_position = glm::vec3((glm::inverse(camera_.get_view())[3]));

	if (freeze_frustum_)
	{
		global_uniform.view              = frozen_view_;
		global_uniform.frustum_planes[0] = frozen_frustum_planes_[0];
		global_uniform.frustum_planes[1] = frozen_frustum_planes_[1];
		global_uniform.frustum_planes[2] = frozen_frustum_planes_[2];
		global_uniform.frustum_planes[3] = frozen_frustum_planes_[3];
		global_uniform.frustum_planes[4] = frozen_frustum_planes_[4];
		global_uniform.frustum_planes[5] = frozen_frustum_planes_[5];
	}
	else
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
	command_buffer.bind_buffer(gpu_scene_.get_draw_command_buffer(), 0, gpu_scene_.get_draw_command_buffer().get_size(), 0, 5, 0);

	command_buffer.bind_buffer(gpu_scene_.get_scene_data_buffer_address(), 0, gpu_scene_.get_scene_data_buffer_address().get_size(), 0, 7, 0);
	/*command_buffer.bind_buffer(gpu_scene_.get_global_vertex_buffer(), 0, gpu_scene_.get_global_vertex_buffer().get_size(), 0, 8, 0);*/
	command_buffer.bind_buffer(gpu_scene_.get_cluster_group_buffer(), 0, gpu_scene_.get_cluster_group_buffer().get_size(), 0, 8, 0);

	command_buffer.bind_buffer(gpu_scene_.get_cluster_buffer(), 0, gpu_scene_.get_cluster_buffer().get_size(), 0, 10, 0);
	command_buffer.bind_buffer(gpu_scene_.get_page_state_buffer(), 0, gpu_scene_.get_page_state_buffer().get_size(), 0, 11, 0);

	command_buffer.bind_buffer(gpu_scene_.get_valid_data_size_buffer(), 0, gpu_scene_.get_valid_data_size_buffer().get_size(), 0, 12, 0);

	auto &hzb_view = input_bindables[2].image_view();

	//if (is_first_frame_)
	//{
	//	is_first_frame_     = false;
	//	auto &hzb_image = hzb_view.get_image();
	//	uint32_t mip_levels = hzb_image.get_mip_levels();

	//	auto                    &device        = gpu_scene_.get_device();
	//	backend::CommandBuffer &command_buffer = device.request_command_buffer();
	//	command_buffer.begin(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

	//	/*common::ImageMemoryBarrier barrier0;
	//	barrier0.old_layout = vk::ImageLayout::eUndefined;
	//	barrier0.new_layout = vk::ImageLayout::eTransferDstOptimal;

	//	common::ImageMemoryBarrier barrier1;
	//	barrier0.old_layout = vk::ImageLayout::eTransferDstOptimal;
	//	barrier0.new_layout = vk::ImageLayout::eShaderReadOnlyOptimal;

	//	command_buffer.image_memory_barrier(hzb_view, barrier0);*/
	//	command_buffer.clear_image(hzb_image, vk::ClearColorValue{0.0f, 0.0f, 0.0f, 0.0f}, {{vk::ImageAspectFlagBits::eColor, 0, mip_levels, 0, 1}});
	//	/*command_buffer.image_memory_barrier(hzb_view, barrier1);*/

	//	command_buffer.end();

	//	const auto &queue = device.get_queue_by_flags(vk::QueueFlagBits::eGraphics, 0);
	//	queue.submit(command_buffer, device.request_fence());

	//	device.get_fence_pool().wait();
	//	device.get_fence_pool().reset();
	//	device.get_command_pool().reset_pool();
	//	device.wait_idle();
	//}

	command_buffer.bind_image(hzb_view, resource_cache.request_sampler(get_linear_sampler()), 0, 13, 0);

	OcclusionUniform uniform;
	uniform.width = hzb_view.get_image().get_extent().width;
	uniform.height = hzb_view.get_image().get_extent().height;
	uniform.mip_count = hzb_view.get_image().get_mip_levels();
	uniform.is_first_frame = is_first_frame_;
	is_first_frame_        = false;
	uniform.depth_bias = 1e-4f;

	auto allocation_occlusion = active_frame.allocate_buffer(
	    vk::BufferUsageFlagBits::eUniformBuffer,
	    sizeof(OcclusionUniform),
	    thread_index_);
	allocation_occlusion.update(uniform);
	command_buffer.bind_buffer(allocation_occlusion.get_buffer(), allocation_occlusion.get_offset(), allocation_occlusion.get_size(), 0, 14, 0);

	command_buffer.push_constants(gpu_scene_.get_lod_threshold());

	command_buffer.draw_mesh_tasks_indirect_count(gpu_scene_.get_draw_command_buffer(), 0, gpu_scene_.get_draw_counts_buffer(), 0, gpu_scene_.get_instance_count(), sizeof(MeshDrawCommand));

	command_buffer.set_has_mesh_shader(false);
}

void MeshLoDPass::show_meshlet_view(bool show)
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

void MeshLoDPass::use_lod(bool use)
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

void MeshLoDPass::show_lod_view(bool show)
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

void MeshLoDPass::show_line(bool show)
{
	polygon_mode_ = show ? vk::PolygonMode::eLine : vk::PolygonMode::eFill;
}

void MeshLoDPass::show_texture()
{
	shader_variant_.add_define("HAS_BASE_COLOR_TEXTURE");
}

void MeshLoDPass::use_occlusion(bool use)
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

void MeshLoDPass::freeze_frustum(bool freeze, sg::Camera *camera)
{
	assert(camera);
	if (freeze == freeze_frustum_)
	{
		return;
	}
	freeze_frustum_ = freeze;
	if (freeze)
	{
		frozen_view_              = camera->get_view();
		glm::mat4 m               = glm::transpose(camera->get_pre_rotation() * camera->get_projection());
		frozen_frustum_planes_[0] = normalize_plane(m[3] + m[0]);
		frozen_frustum_planes_[1] = normalize_plane(m[3] - m[0]);
		frozen_frustum_planes_[2] = normalize_plane(m[3] + m[1]);
		frozen_frustum_planes_[3] = normalize_plane(m[3] - m[1]);
		frozen_frustum_planes_[4] = normalize_plane(m[3] + m[2]);
		frozen_frustum_planes_[5] = normalize_plane(m[3] - m[2]);
	}
}
}        // namespace xihe::rendering
