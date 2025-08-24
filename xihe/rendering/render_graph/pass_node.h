#pragma once

#include "backend/buffer.h"
#include "common/vk_common.h"
#include "gui.h"
#include "render_resource.h"
#include "rendering/passes/render_pass.h"

#include <optional>
#include <variant>

namespace xihe::rendering
{
class RenderGraph;

using Barrier = std::variant<common::ImageMemoryBarrier, common::BufferMemoryBarrier>;

struct ImageProperties
{
	uint32_t array_layers   = 1;
	uint32_t current_layer  = 0;
	uint32_t n_use_layer    = 0;        // 0 means use all layers.
	bool     has_mip_levels = false;
	uint32_t mip_levels     = 1;
	bool     has_initial_value = false;
	vk::ClearColorValue initial_value{};
};

class ExtentDescriptor
{
  public:
	enum class Type
	{
		kFixed,
		kSwapchainRelative,
		kCustom
	};

	ExtentDescriptor() = default;

	static ExtentDescriptor Fixed(const vk::Extent3D &extent)
	{
		return {Type::kFixed, extent};
	}

	static ExtentDescriptor SwapchainRelative(float    width_scale  = 1.0f,
	                                          float    height_scale = 1.0f,
	                                          uint32_t depth        = 1);

	vk::Extent3D calculate(const vk::Extent2D &swapchain_extent) const;

  private:
	Type         type_{Type::kSwapchainRelative};
	vk::Extent3D extent_;
	float        scale_x_ = 1.0f;
	float        scale_y_ = 1.0f;
	uint32_t     depth_   = 1;

	ExtentDescriptor(Type t, const vk::Extent3D &e);
};

struct PassBindable
{
	BindableType     type;
	std::string      name;
	vk::Format       format;
	ExtentDescriptor extent_desc{};

	// only for buffer
	uint32_t buffer_size;

	// for image
	ImageProperties image_properties;
};
struct PassAttachment
{
	AttachmentType   type;
	std::string      name;
	vk::Format       format;
	ExtentDescriptor extent_desc{};

	ImageProperties image_properties;

	bool is_external{false};
};
struct PassInfo
{
	std::vector<PassBindable>   bindables;
	std::vector<PassAttachment> attachments;
};

class PassNode
{
  public:
	struct BindableInfo
	{
		ResourceHandle         handle;
		std::optional<Barrier> barrier;
	};

	struct ImageCopyInfo
	{
		uint32_t        attachment_index{};
		std::unique_ptr<backend::ImageView> image_view;
		vk::ImageCopy   copy_region;
	};

	PassNode(RenderGraph &render_graph, std::string name, PassInfo &&pass_info, std::unique_ptr<RenderPass> &&render_pass);

	void execute(backend::CommandBuffer &command_buffer, RenderTarget &render_target, RenderFrame &render_frame);

	PassInfo &get_pass_info();

	PassType get_type() const;

	std::string get_name() const;

	void set_render_target(std::unique_ptr<RenderTarget> &&render_target);

	/**
	 * \brief
	 * \return If nullptr is returned, it indicates that this pass uses the render target of the render frame
	 */
	RenderTarget *get_render_target();

	void set_gui(Gui *gui);

	void set_image_copy_info(std::unique_ptr<ImageCopyInfo> &&image_read_back);

	void set_batch_index(uint64_t batch_index);

	int64_t get_batch_index() const;

	void add_bindable(uint32_t index, const ResourceHandle &handle, Barrier &&barrier);
	void add_bindable(uint32_t index, const ResourceHandle &handle);

	void add_attachment_memory_barrier(uint32_t index, Barrier &&barrier);

	void add_release_barrier(const ResourceHandle &handle, Barrier &&barrier);

  private:
	RenderGraph &render_graph_;

	std::string name_;

	PassType type_;

	PassInfo pass_info_;

	int64_t batch_index_{-1};

	std::unique_ptr<RenderPass> render_pass_;

	std::unique_ptr<RenderTarget> render_target_;

	// Barriers applied before execution to ensure the input resources are in the correct state for reading.
	std::unordered_map<uint32_t, BindableInfo> bindables_;

	// Barriers applied before execution to ensure the output resources are in the correct state for writing.
	std::unordered_map<uint32_t, Barrier> attachment_barriers_;

	// Barriers applied after execution to release resource ownership for cross-queue synchronization.
	std::unordered_map<ResourceHandle, Barrier> release_barriers_;

	Gui *gui_{nullptr};

	std::unique_ptr<ImageCopyInfo> image_read_back_;
};
}        // namespace xihe::rendering
