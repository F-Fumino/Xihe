#pragma once

#include "render_pass.h"

namespace xihe::rendering
{
struct HZBUniform
{
	uint32_t src_width;
	uint32_t src_height;
	uint32_t dst_width;
	uint32_t dst_height;
};

class MipmapPass : public RenderPass
{
  public:
	MipmapPass() = default;

	void execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables) override;

  private:

	  uint32_t width_  = 0;
	  uint32_t height_ = 0;

	  std::vector<backend::ImageView> mip_views_;
};
}        // namespace xihe::rendering
