#include "copy_pass.h"

namespace xihe::rendering
{
namespace
{
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


void CopyPass::execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables)
{
	auto &resource_cache     = command_buffer.get_device().get_resource_cache();
	auto &comp_shader_module = resource_cache.request_shader_module(vk::ShaderStageFlagBits::eCompute, get_compute_shader());

	std::vector<backend::ShaderModule *> shader_modules = {&comp_shader_module};

	auto &pipeline_layout = resource_cache.request_pipeline_layout(shader_modules);
	command_buffer.bind_pipeline_layout(pipeline_layout);

	auto    &depth_view    = input_bindables[0].image_view();
	auto    &hzb_full_view = input_bindables[1].image_view();
	auto    &hzb_image     = hzb_full_view.get_image();
	auto     extent        = hzb_image.get_extent();
	uint32_t full_w        = extent.width;
	uint32_t full_h        = extent.height;

	if (full_w != width_ || full_h != height_)
	{
		mip_views_.clear();
		mip_views_ .emplace_back(
		    hzb_image,
		    vk::ImageViewType::e2D,
		    hzb_image.get_format(),
		    0, 0, 1, 1);
		width_ = full_w;
		height_ = full_h;
	}

	{
		ScreenUniform uniform;
		uniform.width = width_;
		uniform.height = height_;

		auto allocation = active_frame.allocate_buffer(
		    vk::BufferUsageFlagBits::eUniformBuffer,
		    sizeof(ScreenUniform),
		    thread_index_);
		allocation.update(uniform);

		command_buffer.bind_image(depth_view, resource_cache.request_sampler(get_linear_sampler()), 0, 0, 0);
		command_buffer.bind_image(mip_views_[0], 0, 1, 0);
		command_buffer.bind_buffer(allocation.get_buffer(), allocation.get_offset(), allocation.get_size(), 0, 2, 0);

		command_buffer.dispatch((full_w + 7) / 8, (full_h + 7) / 8, 1);

		common::ImageMemoryBarrier barrier{};
		barrier.old_layout      = vk::ImageLayout::eGeneral;
		barrier.new_layout      = vk::ImageLayout::eGeneral;
		barrier.src_access_mask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite;
		barrier.dst_access_mask = vk::AccessFlagBits2::eShaderRead;
		barrier.src_stage_mask  = vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer;
		barrier.dst_stage_mask  = vk::PipelineStageFlagBits2::eComputeShader;

		command_buffer.image_memory_barrier(mip_views_[0], barrier);
	}
}
}        // namespace xihe::rendering
