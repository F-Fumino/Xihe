#include "render_graph.h"

#include "rendering/render_frame.h"

#include <ranges>

namespace xihe::rendering
{
void set_viewport_and_scissor(backend::CommandBuffer const &command_buffer, vk::Extent2D const &extent)
{
	command_buffer.get_handle().setViewport(0, {{0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f}});
	command_buffer.get_handle().setScissor(0, vk::Rect2D({}, extent));
}

RenderGraph::RenderGraph(RenderContext &render_context, stats::Stats *stats) :
    render_context_{render_context}, stats_{stats}
{}

void RenderGraph::execute(bool present)
{
	render_context_.begin_frame();
	
	bool is_first = true;

	size_t batch_count = pass_batches_.size();
	for (size_t i = 0; i < batch_count; ++i)
	{
		//bool is_first = (i == 0);
		bool is_last  = (i == batch_count - 1);
		if (pass_batches_[i].type == PassType::kRaster)
		{
			bool is_before_stream = !is_last && pass_batches_[i + 1].type == PassType::kStreaming;
			execute_raster_batch(pass_batches_[i], is_first, is_last, present, is_before_stream);
			is_first = false;
		}
		else if (pass_batches_[i].type == PassType::kCompute)
		{
			execute_compute_batch(pass_batches_[i], is_first, is_last);
		}
		else if (pass_batches_[i].type == PassType::kStreaming)
		{
			execute_streaming_batch(pass_batches_[i]);
		}
	}
}

ShaderBindable RenderGraph::get_resource_bindable(ResourceHandle handle) const
{
	auto it = resources_.find(handle);
	if (it == resources_.end())
	{
		throw std::runtime_error("Resource not found");
	}
	return it->second.get_bindable();
}

void RenderGraph::add_pass_node(PassNode &&pass_node)
{
	pass_nodes_.push_back(std::move(pass_node));
}

void RenderGraph::execute_raster_batch(PassBatch &pass_batch, bool is_first, bool is_last, bool present, bool is_before_stream)
{
	auto &command_buffer = render_context_.request_graphics_command_buffer(
	    backend::CommandBuffer::ResetMode::kResetPool,
	    vk::CommandBufferLevel::ePrimary, 0);

	command_buffer.begin(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

	//backend::Device &device = render_context_.get_device();
	//bool flag = false;

	if (stats_)
	{
		stats_->begin_sampling(command_buffer);
	}

	for (const auto pass_node : pass_batch.pass_nodes)
	{
		RenderTarget *render_target = pass_node->get_render_target();

		if (!render_target)
		{
			render_target = &render_context_.get_active_frame().get_render_target();
		}

		set_viewport_and_scissor(command_buffer, render_target->get_extent());

		pass_node->execute(command_buffer, *render_target, render_context_.get_active_frame());

		//if (pass_node->get_name() == "Geometry")
		//{
		//	flag = true;
		//}
	}

	if (stats_)
	{
		stats_->end_sampling(command_buffer);
	}

	//if (flag)
	//{
	//	device.wait_idle();
	//}

	command_buffer.end();

	//if (flag)
	//{
	//	device.wait_idle();
	//}

	const auto     last_wait_batch      = pass_batch.wait_batch_index;
	const uint64_t wait_semaphore_value = last_wait_batch >= 0 ? pass_batches_[last_wait_batch].signal_semaphore_value : 0;

	render_context_.graphics_submit(
	    {&command_buffer},        // list of command buffers
	    pass_batch.signal_semaphore_value,
	    wait_semaphore_value,
	    is_first,
	    is_last,
		present,
		is_before_stream);
	
	//if (flag)
	//{
	//	device.wait_idle();
	//}
}

void RenderGraph::execute_compute_batch(PassBatch &pass_batch, bool is_first, bool is_last)
{
	auto &command_buffer = render_context_.request_compute_command_buffer(
	    backend::CommandBuffer::ResetMode::kResetPool,
	    vk::CommandBufferLevel::ePrimary, 0);
	command_buffer.begin(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
	if (stats_)
	{
		stats_->begin_sampling(command_buffer);
	}
	for (const auto pass_node : pass_batch.pass_nodes)
	{
		pass_node->execute(command_buffer, render_context_.get_active_frame().get_render_target(), render_context_.get_active_frame());
	}
	if (stats_)
	{
		stats_->end_sampling(command_buffer);
	}

	command_buffer.end();
	const auto     last_wait_batch      = pass_batch.wait_batch_index;
	const uint64_t wait_semaphore_value = last_wait_batch >= 0 ? pass_batches_[last_wait_batch].signal_semaphore_value : 0;

	render_context_.compute_submit(
	    {&command_buffer},        // list of command buffers
	    pass_batch.signal_semaphore_value,
	    wait_semaphore_value);
}

void RenderGraph::execute_streaming_batch(PassBatch &pass_batch)
{
	auto &command_buffer = render_context_.request_sparse_command_buffer(
	    backend::CommandBuffer::ResetMode::kResetPool,
	    vk::CommandBufferLevel::ePrimary, 0);

	command_buffer.begin(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
	//auto            &command_buffer = device.request_command_buffer();
	//command_buffer.begin(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

	//if (stats_)
	//{
	//	stats_->begin_sampling(command_buffer);
	//}

	for (const auto pass_node : pass_batch.pass_nodes)
	{
		RenderTarget *render_target = pass_node->get_render_target();

		//if (!render_target)
		//{
		//	render_target = &render_context_.get_active_frame().get_render_target();
		//}

		//set_viewport_and_scissor(command_buffer, render_target->get_extent());

		pass_node->execute(command_buffer, *render_target, render_context_.get_active_frame());
	}

	//if (stats_)
	//{
	//	stats_->end_sampling(command_buffer);
	//}

	command_buffer.end();

	const auto     last_wait_batch      = pass_batch.wait_batch_index;
	const uint64_t wait_semaphore_value = last_wait_batch >= 0 ? pass_batches_[last_wait_batch].signal_semaphore_value : 0;

	backend::Device &device = render_context_.get_device();
	device.wait_idle();

	render_context_.sparse_submit(
	    {&command_buffer},
	    pass_batch.signal_semaphore_value,
		0);

	device.wait_idle();

	//const auto &queue = device.get_queue_by_flags(vk::QueueFlagBits::eGraphics, 0);
	//queue.submit(command_buffer, device.request_fence());

	//device.get_fence_pool().wait();
	//device.get_fence_pool().reset();
	//device.get_command_pool().reset_pool();
}
}        // namespace xihe::rendering
