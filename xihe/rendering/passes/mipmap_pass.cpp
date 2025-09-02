#include "mipmap_pass.h"

namespace xihe::rendering
{

void MipmapPass::execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables)
{
	auto &resource_cache     = command_buffer.get_device().get_resource_cache();
	auto &comp_shader_module = resource_cache.request_shader_module(vk::ShaderStageFlagBits::eCompute, get_compute_shader());

	std::vector<backend::ShaderModule *> shader_modules = {&comp_shader_module};

	auto &pipeline_layout = resource_cache.request_pipeline_layout(shader_modules);
	command_buffer.bind_pipeline_layout(pipeline_layout);

	auto    &hzb_full_view = input_bindables[0].image_view();
	auto    &hzb_image     = hzb_full_view.get_image();
	uint32_t mip_levels    = hzb_image.get_mip_levels();
	auto     extent        = hzb_image.get_extent();
	uint32_t full_w        = extent.width;
	uint32_t full_h        = extent.height;

	if (width_ != full_w || height_ != full_h)
	{
		mip_views_.clear();
		for (uint32_t i = 0; i < mip_levels; i++)
		{
			mip_views_.emplace_back(
			    hzb_image,
			    vk::ImageViewType::e2D,
			    hzb_image.get_format(),
			    i, 0, 1, 1);
		}
		width_  = full_w;
		height_ = full_h;
	}

	uint32_t prev_w = full_w;
	uint32_t prev_h = full_h;
	uint32_t cur_w;
	uint32_t cur_h;

	for (uint32_t mip = 1; mip < mip_levels; mip++)
	{
		cur_w = std::max(1u, full_w >> mip);
		cur_h = std::max(1u, full_h >> mip);

		HZBUniform uniform;
		uniform.src_width = prev_w;
		uniform.src_height = prev_h;
		uniform.dst_width  = cur_w;
		uniform.dst_height = cur_h;

		auto        allocation = active_frame.allocate_buffer(
            vk::BufferUsageFlagBits::eUniformBuffer,
		    sizeof(HZBUniform),
            thread_index_);
		allocation.update(uniform);

		command_buffer.bind_image(mip_views_[mip - 1], 0, 0, 0);
		command_buffer.bind_image(mip_views_[mip], 0, 1, 0);
		command_buffer.bind_buffer(allocation.get_buffer(), allocation.get_offset(), allocation.get_size(), 0, 2, 0);

		command_buffer.dispatch(
		    (cur_w + 7) / 8,
		    (cur_h + 7) / 8,
		    1);

		common::ImageMemoryBarrier barrier{};
		barrier.old_layout      = vk::ImageLayout::eGeneral;
		barrier.new_layout      = vk::ImageLayout::eGeneral;
		barrier.src_access_mask = vk::AccessFlagBits2::eShaderWrite;
		barrier.dst_access_mask = vk::AccessFlagBits2::eShaderRead;
		barrier.src_stage_mask  = vk::PipelineStageFlagBits2::eComputeShader;
		barrier.dst_stage_mask  = vk::PipelineStageFlagBits2::eComputeShader;

		command_buffer.image_memory_barrier(mip_views_[mip], barrier);

		prev_w = cur_w;
		prev_h = cur_h;
	}
}
}        // namespace xihe::rendering
