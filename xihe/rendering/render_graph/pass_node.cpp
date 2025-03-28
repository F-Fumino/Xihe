#include "pass_node.h"

#include "render_graph.h"

namespace xihe::rendering
{
ExtentDescriptor ExtentDescriptor::SwapchainRelative(float width_scale, float height_scale, uint32_t depth)
{
	ExtentDescriptor desc(Type::kSwapchainRelative, {});
	desc.scale_x_ = width_scale;
	desc.scale_y_ = height_scale;
	desc.depth_   = depth;
	return desc;
}

vk::Extent3D ExtentDescriptor::calculate(const vk::Extent2D &swapchain_extent) const
{
	switch (type_)
	{
		case Type::kFixed:
			return extent_;
		case Type::kSwapchainRelative:
			return vk::Extent3D{
			    static_cast<uint32_t>(swapchain_extent.width * scale_x_),
			    static_cast<uint32_t>(swapchain_extent.height * scale_y_),
			    depth_};
		default:
			return extent_;
	}
}

ExtentDescriptor::ExtentDescriptor(Type t, const vk::Extent3D &e) :
    type_(t), extent_(e)
{}

PassNode::PassNode(RenderGraph &render_graph, std::string name, PassInfo &&pass_info, std::unique_ptr<RenderPass> &&render_pass) :
    render_graph_{render_graph}, name_{std::move(name)}, type_{render_pass->get_type()}, pass_info_{std::move(pass_info)}, render_pass_{std::move(render_pass)}
{
}

void PassNode::execute(backend::CommandBuffer &command_buffer, RenderTarget &render_target, RenderFrame &render_frame)
{
	backend::ScopedDebugLabel subpass_debug_label{command_buffer, name_.c_str()};

	std::vector<ShaderBindable> shader_bindable(bindables_.size());

	for (const auto &[index, input_info] : bindables_)
	{
		shader_bindable[index] = render_graph_.get_resource_bindable(input_info.handle);

		if (!input_info.barrier.has_value())
		{
			continue;
		}

		if (shader_bindable[index].is_buffer())
		{
			auto &buffer = shader_bindable[index].buffer();
			// todo offset and size
			command_buffer.buffer_memory_barrier(buffer, 0, buffer.get_size(), std::get<common::BufferMemoryBarrier>(input_info.barrier.value()));
		}
		else
		{
			command_buffer.image_memory_barrier(shader_bindable[index].image_view(), std::get<common::ImageMemoryBarrier>(input_info.barrier.value()));
		}
	}

	auto &output_views = render_target.get_views();
	for (const auto &[index, barrier] : attachment_barriers_)
	{
		if (std::holds_alternative<common::ImageMemoryBarrier>(barrier))
		{
			command_buffer.image_memory_barrier(output_views[index], std::get<common::ImageMemoryBarrier>(barrier));
		}
		else
		{

		}
	}

	if (type_ == PassType::kRaster)
	{
		command_buffer.begin_rendering(render_target);
	}

	render_pass_->execute(command_buffer, render_frame, shader_bindable);

	if (gui_)
	{
		gui_->draw(command_buffer);
	}

	if (type_ == PassType::kRaster)
	{
		command_buffer.end_rendering();
		if (!render_target_)
		{
			auto                      &views = render_target.get_views();
			common::ImageMemoryBarrier memory_barrier{};
			memory_barrier.old_layout      = vk::ImageLayout::eColorAttachmentOptimal;
			memory_barrier.new_layout      = vk::ImageLayout::ePresentSrcKHR;
			memory_barrier.src_access_mask = vk::AccessFlagBits2::eColorAttachmentWrite;
			memory_barrier.src_stage_mask  = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
			memory_barrier.dst_stage_mask  = vk::PipelineStageFlagBits2::eBottomOfPipe;

			command_buffer.image_memory_barrier(views[0], memory_barrier);
		}
	}

	if (image_read_back_)
	{
		{
			common::ImageMemoryBarrier memory_barrier{};
			memory_barrier.new_layout      = vk::ImageLayout::eTransferDstOptimal;
			memory_barrier.dst_access_mask = vk::AccessFlagBits2::eTransferWrite;
			memory_barrier.src_stage_mask  = vk::PipelineStageFlagBits2::eAllCommands;
			memory_barrier.dst_stage_mask  = vk::PipelineStageFlagBits2::eAllCommands;

			command_buffer.image_memory_barrier(*image_read_back_->image_view, memory_barrier);
		}

		assert(image_read_back_->attachment_index < render_target.get_views().size());
		const auto &src_image_view = render_target.get_views()[image_read_back_->attachment_index];

		const vk::ImageLayout  layout      = std::get<common::ImageMemoryBarrier>(attachment_barriers_[image_read_back_->attachment_index]).new_layout;
		const vk::AccessFlags2 aspect_mask = std::get<common::ImageMemoryBarrier>(attachment_barriers_[image_read_back_->attachment_index]).dst_access_mask;

		{
			common::ImageMemoryBarrier memory_barrier;
			memory_barrier.old_layout      = layout;
			memory_barrier.src_access_mask = aspect_mask;
			memory_barrier.new_layout      = vk::ImageLayout::eTransferSrcOptimal;
			memory_barrier.dst_access_mask = vk::AccessFlagBits2::eTransferRead;
			memory_barrier.src_stage_mask  = vk::PipelineStageFlagBits2::eAllCommands;
			// todo
			memory_barrier.dst_stage_mask  = vk::PipelineStageFlagBits2::eAllCommands;

			command_buffer.image_memory_barrier(src_image_view, memory_barrier);	
		}

		command_buffer.copy_image(src_image_view.get_image(), image_read_back_->image_view->get_image(), {image_read_back_->copy_region});

		{
			common::ImageMemoryBarrier memory_barrier;
			memory_barrier.old_layout      = vk::ImageLayout::eTransferSrcOptimal;
			memory_barrier.src_access_mask = vk::AccessFlagBits2::eTransferRead;
			memory_barrier.new_layout      = layout;
			memory_barrier.dst_access_mask = aspect_mask;
			memory_barrier.src_stage_mask  = vk::PipelineStageFlagBits2::eAllCommands;
			memory_barrier.dst_stage_mask  = vk::PipelineStageFlagBits2::eAllCommands;

			command_buffer.image_memory_barrier(src_image_view, memory_barrier);
		}

		{
			common::ImageMemoryBarrier memory_barrier;
			memory_barrier.old_layout      = vk::ImageLayout::eTransferDstOptimal;
			memory_barrier.src_access_mask = vk::AccessFlagBits2::eTransferWrite;
			memory_barrier.new_layout      = vk::ImageLayout::eShaderReadOnlyOptimal;
			memory_barrier.dst_access_mask = vk::AccessFlagBits2::eHostWrite | vk::AccessFlagBits2::eTransferWrite;
			memory_barrier.src_stage_mask  = vk::PipelineStageFlagBits2::eAllCommands;
			memory_barrier.dst_stage_mask  = vk::PipelineStageFlagBits2::eAllCommands;

			command_buffer.image_memory_barrier(*image_read_back_->image_view, memory_barrier);
		}
	}

	for (auto &[handle, barrier] : release_barriers_)
	{
		if (std::holds_alternative<common::ImageMemoryBarrier>(barrier))
		{
			command_buffer.image_memory_barrier(render_graph_.get_resource_bindable(handle).image_view(), std::get<common::ImageMemoryBarrier>(barrier));
		}
		else
		{
			auto &buffer = render_graph_.get_resource_bindable(handle).buffer();
			// todo offset and size
			command_buffer.buffer_memory_barrier(buffer, 0, buffer.get_size(), std::get<common::BufferMemoryBarrier>(barrier));
		}
	}
}

PassInfo &PassNode::get_pass_info()
{
	return pass_info_;
}

PassType PassNode::get_type() const
{
	return type_;
}

std::string PassNode::get_name() const
{
	return name_;
}

void PassNode::set_render_target(std::unique_ptr<RenderTarget> &&render_target)
{
	render_target_ = std::move(render_target);
}

RenderTarget *PassNode::get_render_target()
{
	return render_target_.get();
}

void PassNode::set_gui(Gui *gui)
{
	gui_ = gui;
}

void PassNode::set_image_copy_info(std::unique_ptr<ImageCopyInfo> &&image_read_back)
{
	image_read_back_ = std::move(image_read_back);
}

void PassNode::set_batch_index(uint64_t batch_index)
{
	batch_index_ = batch_index;
}

int64_t PassNode::get_batch_index() const
{
	if (batch_index_ == -1)
	{
		throw std::runtime_error("Batch index not set");
	}
	return batch_index_;
}

void PassNode::add_bindable(uint32_t index, const ResourceHandle &handle, Barrier &&barrier)
{
	bindables_[index] = {handle, barrier};
}

void PassNode::add_bindable(uint32_t index, const ResourceHandle &handle)
{
	bindables_[index] = {handle, std::nullopt};
}

void PassNode::add_attachment_memory_barrier(uint32_t index, Barrier &&barrier)
{
	attachment_barriers_[index] = barrier;
}

void PassNode::add_release_barrier(const ResourceHandle &handle, Barrier &&barrier)
{
	release_barriers_[handle] = barrier;
}
}        // namespace xihe::rendering
