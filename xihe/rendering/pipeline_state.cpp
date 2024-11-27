#include "pipeline_state.h"

bool operator==(const VkVertexInputAttributeDescription &lhs, const VkVertexInputAttributeDescription &rhs)
{
	return std::tie(lhs.binding, lhs.format, lhs.location, lhs.offset) == std::tie(rhs.binding, rhs.format, rhs.location, rhs.offset);
}

bool operator==(const VkVertexInputBindingDescription &lhs, const VkVertexInputBindingDescription &rhs)
{
	return std::tie(lhs.binding, lhs.inputRate, lhs.stride) == std::tie(rhs.binding, rhs.inputRate, rhs.stride);
}

bool operator==(const xihe::ColorBlendAttachmentState &lhs, const xihe::ColorBlendAttachmentState &rhs)
{
	return std::tie(lhs.alpha_blend_op, lhs.blend_enable, lhs.color_blend_op, lhs.color_write_mask, lhs.dst_alpha_blend_factor, lhs.dst_color_blend_factor, lhs.src_alpha_blend_factor, lhs.src_color_blend_factor) ==
	       std::tie(rhs.alpha_blend_op, rhs.blend_enable, rhs.color_blend_op, rhs.color_write_mask, rhs.dst_alpha_blend_factor, rhs.dst_color_blend_factor, rhs.src_alpha_blend_factor, rhs.src_color_blend_factor);
}

bool operator!=(const xihe::AttachmentsState &lhs, const xihe::AttachmentsState &rhs)
{
	return lhs.color_attachment_formats != rhs.color_attachment_formats || lhs.depth_attachment_format != rhs.depth_attachment_format || lhs.stencil_attachment_format != rhs.stencil_attachment_format;
}

bool operator!=(const xihe::VertexInputState &lhs, const xihe::VertexInputState &rhs)
{
	return lhs.attributes != rhs.attributes || lhs.bindings != rhs.bindings;
}

bool operator!=(const xihe::InputAssemblyState &lhs, const xihe::InputAssemblyState &rhs)
{
	return std::tie(lhs.primitive_restart_enable, lhs.topology) != std::tie(rhs.primitive_restart_enable, rhs.topology);
}

bool operator!=(const xihe::RasterizationState &lhs, const xihe::RasterizationState &rhs)
{
	return std::tie(lhs.cull_mode, lhs.depth_bias_enable, lhs.depth_clamp_enable, lhs.front_face, lhs.front_face, lhs.polygon_mode, lhs.rasterizer_discard_enable) !=
	       std::tie(rhs.cull_mode, rhs.depth_bias_enable, rhs.depth_clamp_enable, rhs.front_face, rhs.front_face, rhs.polygon_mode, rhs.rasterizer_discard_enable);
}

bool operator!=(const xihe::ViewportState &lhs, const xihe::ViewportState &rhs)
{
	return lhs.viewport_count != rhs.viewport_count || lhs.scissor_count != rhs.scissor_count;
}

bool operator!=(const xihe::MultisampleState &lhs, const xihe::MultisampleState &rhs)
{
	return std::tie(lhs.alpha_to_coverage_enable, lhs.alpha_to_one_enable, lhs.min_sample_shading, lhs.rasterization_samples, lhs.sample_mask, lhs.sample_shading_enable) !=
	       std::tie(rhs.alpha_to_coverage_enable, rhs.alpha_to_one_enable, rhs.min_sample_shading, rhs.rasterization_samples, rhs.sample_mask, rhs.sample_shading_enable);
}

bool operator!=(const xihe::DepthStencilState &lhs, const xihe::DepthStencilState &rhs)
{
	return std::tie(lhs.depth_bounds_test_enable, lhs.depth_compare_op, lhs.depth_test_enable, lhs.depth_write_enable, lhs.stencil_test_enable) !=
	           std::tie(rhs.depth_bounds_test_enable, rhs.depth_compare_op, rhs.depth_test_enable, rhs.depth_write_enable, rhs.stencil_test_enable) ||
	       lhs.back != rhs.back || lhs.front != rhs.front;
}

bool operator!=(const xihe::ColorBlendState &lhs, const xihe::ColorBlendState &rhs)
{
	return std::tie(lhs.logic_op, lhs.logic_op_enable) != std::tie(rhs.logic_op, rhs.logic_op_enable) ||
	       lhs.attachments.size() != rhs.attachments.size() ||
	       !std::equal(lhs.attachments.begin(), lhs.attachments.end(), rhs.attachments.begin(),
	                   [](const xihe::ColorBlendAttachmentState &lhs, const xihe::ColorBlendAttachmentState &rhs) {
		                   return lhs == rhs;
	                   });
}

namespace xihe
{
void SpecializationConstantState::reset()
{
	if (dirty_)
	{
		specialization_constant_state_.clear();
	}

	dirty_ = false;
}

bool SpecializationConstantState::is_dirty() const
{
	return dirty_;
}

void SpecializationConstantState::clear_dirty()
{
	dirty_ = false;
}

void SpecializationConstantState::set_constant(uint32_t constant_id, const std::vector<uint8_t> &value)
{
	const auto data = specialization_constant_state_.find(constant_id);

	if (data != specialization_constant_state_.end() && data->second == value)
	{
		return;
	}

	dirty_ = true;

	specialization_constant_state_[constant_id] = value;
}

void SpecializationConstantState::set_specialization_constant_state(const std::map<uint32_t, std::vector<uint8_t>> &state)
{
	specialization_constant_state_ = state;
}

const std::map<uint32_t, std::vector<uint8_t>> &SpecializationConstantState::get_specialization_constant_state() const
{
	return specialization_constant_state_;
}

void PipelineState::reset()
{
	clear_dirty();

	pipeline_layout_ = nullptr;
	// render_pass_     = nullptr;

	specialization_constant_state_.reset();

	vertex_input_state_   = {};
	input_assembly_state_ = {};
	rasterization_state_  = {};
	multisample_state_    = {};
	depth_stencil_state_  = {};
	color_blend_state_    = {};
	subpass_index_        = {0U};
}

void PipelineState::set_pipeline_layout(backend::PipelineLayout &pipeline_layout)
{
	if (pipeline_layout_)
	{
		if (pipeline_layout_->get_handle() != pipeline_layout.get_handle())
		{
			pipeline_layout_ = &pipeline_layout;

			dirty_ = true;
		}
	}
	else
	{
		pipeline_layout_ = &pipeline_layout;

		dirty_ = true;
	}
}

void PipelineState::set_specialization_constant(uint32_t constant_id, const std::vector<uint8_t> &data)
{
	specialization_constant_state_.set_constant(constant_id, data);

	if (specialization_constant_state_.is_dirty())
	{
		dirty_ = true;
	}
}

void PipelineState::set_attachments_state(const AttachmentsState &attachments_state)
{
	if (attachments_state_ != attachments_state)
	{
		attachments_state_ = attachments_state;

		dirty_ = true;
	}
}

void PipelineState::set_vertex_input_state(const VertexInputState &vertex_input_state)
{
	if (vertex_input_state_ != vertex_input_state)
	{
		vertex_input_state_ = vertex_input_state;

		dirty_ = true;
	}
}

void PipelineState::set_input_assembly_state(const InputAssemblyState &input_assembly_state)
{
	if (input_assembly_state_ != input_assembly_state)
	{
		input_assembly_state_ = input_assembly_state;

		dirty_ = true;
	}
}

void PipelineState::set_rasterization_state(const RasterizationState &rasterization_state)
{
	if (rasterization_state_ != rasterization_state)
	{
		rasterization_state_ = rasterization_state;

		dirty_ = true;
	}
}

void PipelineState::set_viewport_state(const ViewportState &viewport_state)
{
	if (viewport_state_ != viewport_state)
	{
		viewport_state_ = viewport_state;

		dirty_ = true;
	}
}

void PipelineState::set_multisample_state(const MultisampleState &multisample_state)
{
	if (multisample_state_ != multisample_state)
	{
		multisample_state_ = multisample_state;

		dirty_ = true;
	}
}

void PipelineState::set_depth_stencil_state(const DepthStencilState &depth_stencil_state)
{
	if (depth_stencil_state_ != depth_stencil_state)
	{
		depth_stencil_state_ = depth_stencil_state;

		dirty_ = true;
	}
}

void PipelineState::set_color_blend_state(const ColorBlendState &color_blend_state)
{
	if (color_blend_state_ != color_blend_state)
	{
		color_blend_state_ = color_blend_state;

		dirty_ = true;
	}
}

void PipelineState::set_subpass_index(uint32_t subpass_index)
{
	if (subpass_index_ != subpass_index)
	{
		subpass_index_ = subpass_index;

		dirty_ = true;
	}
}

void PipelineState::set_has_mesh_shader(bool has_mesh_shader)
{
	has_mesh_shader_ = has_mesh_shader;
}

const backend::PipelineLayout &PipelineState::get_pipeline_layout() const
{
	assert(pipeline_layout_ && "Graphics state Pipeline layout is not set");
	return *pipeline_layout_;
}

const SpecializationConstantState &PipelineState::get_specialization_constant_state() const
{
	return specialization_constant_state_;
}

const AttachmentsState & PipelineState::get_attachments_state() const
{
	return attachments_state_;
}

const VertexInputState &PipelineState::get_vertex_input_state() const
{
	return vertex_input_state_;
}

const InputAssemblyState &PipelineState::get_input_assembly_state() const
{
	return input_assembly_state_;
}

const RasterizationState &PipelineState::get_rasterization_state() const
{
	return rasterization_state_;
}

const ViewportState &PipelineState::get_viewport_state() const
{
	return viewport_state_;
}

const MultisampleState &PipelineState::get_multisample_state() const
{
	return multisample_state_;
}

const DepthStencilState &PipelineState::get_depth_stencil_state() const
{
	return depth_stencil_state_;
}

const ColorBlendState &PipelineState::get_color_blend_state() const
{
	return color_blend_state_;
}

bool PipelineState::has_mesh_shader() const
{
	return has_mesh_shader_;
}

//uint32_t PipelineState::get_subpass_index() const
//{
//	return subpass_index_;
//}

bool PipelineState::is_dirty() const
{
	return dirty_ || specialization_constant_state_.is_dirty();
}

void PipelineState::clear_dirty()
{
	dirty_ = false;
	specialization_constant_state_.clear_dirty();
}
}        // namespace xihe
