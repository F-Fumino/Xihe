#pragma once

#include "render_pass.h"

namespace xihe::rendering
{
struct ScreenUniform
{
	uint32_t width;
	uint32_t height;
};

class CopyPass : public RenderPass
{
  public:
	CopyPass() = default;

	void execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables) override;

  private:

	  std::vector<backend::ImageView> mip_views_;

	  uint32_t width_  = 0;
	  uint32_t height_ = 0;
};
}        // namespace xihe::rendering
