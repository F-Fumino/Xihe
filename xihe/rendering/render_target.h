#pragma once
#include <functional>
#include <memory>

#include "backend/image.h"
#include "backend/image_view.h"

namespace xihe
{
namespace backend
{
class Device;
}

namespace rendering
{

struct Attachment
{
	Attachment() = default;

	Attachment(vk::Format format, vk::SampleCountFlagBits samples, vk::ImageUsageFlags usage) :
	    format{format},
	    samples{samples},
	    usage{usage}
	{}

	vk::Format              format         = vk::Format::eUndefined;
	vk::SampleCountFlagBits samples        = vk::SampleCountFlagBits::e1;
	vk::ImageUsageFlags     usage          = vk::ImageUsageFlagBits::eSampled;
	vk::ImageLayout         initial_layout = vk::ImageLayout::eUndefined;
};

class RenderTarget
{
  public:
	using CreateFunc = std::function<std::unique_ptr<RenderTarget>(backend::Image &&)>;
	static const CreateFunc kDefaultCreateFunc;

	RenderTarget(std::vector<backend::Image> &&images, uint32_t base_layer=0, uint32_t layer_count=0);

	RenderTarget(std::vector<backend::ImageView> &&image_views);

	const vk::Extent2D                    &get_extent() const;
	// todo need const?
	std::vector<backend::ImageView> &get_views();

	void     set_first_bindless_descriptor_set_index(uint32_t index);
	uint32_t get_first_bindless_descriptor_set_index() const;

  private:
	backend::Device                &device_;
	vk::Extent2D                    extent_;
	std::vector<backend::Image>     images_; // Can be empty. Ownership is managed externally (e.g., by Render Graph)

	std::vector<backend::ImageView> image_views_;


	uint32_t first_bindless_descriptor_set_index_ = 0;
};
}        // namespace rendering
}        // namespace xihe
