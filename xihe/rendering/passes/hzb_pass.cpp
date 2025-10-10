#include "hzb_pass.h"

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


void HZBPass::execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables)
{
	auto &resource_cache     = command_buffer.get_device().get_resource_cache();
	auto &comp_shader_module = resource_cache.request_shader_module(vk::ShaderStageFlagBits::eCompute, get_compute_shader());

	std::vector<backend::ShaderModule *> shader_modules = {&comp_shader_module};

	auto &pipeline_layout = resource_cache.request_pipeline_layout(shader_modules);
	command_buffer.bind_pipeline_layout(pipeline_layout);

	auto    &depth_view    = input_bindables[0].image_view();
	auto    &hzb_full_view = input_bindables[1].image_view();
	auto    &hzb_image     = hzb_full_view.get_image();
	uint32_t mip_levels    = hzb_image.get_mip_levels();
	auto     extent        = hzb_image.get_extent();
	uint32_t full_w        = extent.width;
	uint32_t full_h        = extent.height;

	if (mip_levels != mip_views_.size())
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
	}

	/*{
		common::ImageMemoryBarrier barrier{};
		barrier.old_layout      = vk::ImageLayout::eUndefined;
		barrier.new_layout      = vk::ImageLayout::eGeneral;
		barrier.src_access_mask = {};
		barrier.dst_access_mask = vk::AccessFlagBits2::eShaderWrite;
		barrier.src_stage_mask  = vk::PipelineStageFlagBits2::eTopOfPipe;
		barrier.dst_stage_mask  = vk::PipelineStageFlagBits2::eComputeShader;

		command_buffer.image_memory_barrier(mip_views_[0], barrier);
	}*/

	{
		HZBUniforms uniforms;
		uniforms.mode = 0;
		uniforms.src_width = full_w;
		uniforms.src_height = full_h;
		uniforms.dst_width  = full_w;
		uniforms.dst_height = full_h;

		auto allocation = active_frame.allocate_buffer(
		    vk::BufferUsageFlagBits::eUniformBuffer,
		    sizeof(HZBUniforms),
		    thread_index_);
		allocation.update(uniforms);

		command_buffer.bind_image(depth_view, resource_cache.request_sampler(get_linear_sampler()), 0, 0, 0);
		command_buffer.bind_image(mip_views_[0], 0, 1, 0);
		command_buffer.bind_buffer(allocation.get_buffer(), allocation.get_offset(), allocation.get_size(), 0, 2, 0);

		command_buffer.dispatch((full_w + 7) / 8, (full_h + 7) / 8, 1);
	}

	uint32_t prev_w = full_w;
	uint32_t prev_h = full_h;
	uint32_t cur_w;
	uint32_t cur_h;

	for (uint32_t mip = 1; mip < mip_levels; mip++)
	{
		{
			common::ImageMemoryBarrier barrier{};
			barrier.old_layout      = vk::ImageLayout::eGeneral;
			barrier.new_layout      = vk::ImageLayout::eShaderReadOnlyOptimal;
			barrier.src_access_mask = vk::AccessFlagBits2::eShaderWrite;
			barrier.dst_access_mask = vk::AccessFlagBits2::eShaderRead;
			barrier.src_stage_mask  = vk::PipelineStageFlagBits2::eComputeShader;
			barrier.dst_stage_mask  = vk::PipelineStageFlagBits2::eComputeShader;
			command_buffer.image_memory_barrier(mip_views_[mip - 1], barrier);
		}

		/*{
			common::ImageMemoryBarrier barrier{};
			barrier.old_layout      = vk::ImageLayout::eUndefined;
			barrier.new_layout      = vk::ImageLayout::eGeneral;
			barrier.src_access_mask = {};
			barrier.dst_access_mask = vk::AccessFlagBits2::eShaderWrite;
			barrier.src_stage_mask  = vk::PipelineStageFlagBits2::eTopOfPipe;
			barrier.dst_stage_mask  = vk::PipelineStageFlagBits2::eComputeShader;
			command_buffer.image_memory_barrier(mip_views_[mip], barrier);
		}*/

		cur_w = std::max(1u, full_w >> mip);
		cur_h = std::max(1u, full_h >> mip);

		HZBUniforms uniforms;
		uniforms.mode = 1;
		uniforms.src_width = prev_w;
		uniforms.src_height = prev_h;
		uniforms.dst_width  = cur_w;
		uniforms.dst_height = cur_h;

		auto        allocation = active_frame.allocate_buffer(
            vk::BufferUsageFlagBits::eUniformBuffer,
            sizeof(HZBUniforms),
            thread_index_);
		allocation.update(uniforms);

		command_buffer.bind_image(mip_views_[mip - 1], resource_cache.request_sampler(get_linear_sampler()), 0, 0, 0);
		command_buffer.bind_image(mip_views_[mip], 0, 1, 0);
		command_buffer.bind_buffer(allocation.get_buffer(), allocation.get_offset(), allocation.get_size(), 0, 2, 0);

		command_buffer.dispatch(
		    (cur_w + 7) / 8,
		    (cur_h + 7) / 8,
		    1);

		prev_w = cur_w;
		prev_h = cur_h;
	}

	common::ImageMemoryBarrier barrier{};
	barrier.old_layout      = vk::ImageLayout::eGeneral;
	barrier.new_layout      = vk::ImageLayout::eShaderReadOnlyOptimal;
	barrier.src_access_mask = vk::AccessFlagBits2::eShaderWrite;
	barrier.dst_access_mask = vk::AccessFlagBits2::eShaderRead;
	barrier.src_stage_mask  = vk::PipelineStageFlagBits2::eComputeShader;
	barrier.dst_stage_mask  = vk::PipelineStageFlagBits2::eComputeShader;
	command_buffer.image_memory_barrier(mip_views_[mip_levels - 1], barrier);
}
}        // namespace xihe::rendering
