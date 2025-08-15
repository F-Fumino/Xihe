#pragma once

#include "render_pass.h"

namespace xihe::rendering
{
struct HZBUniforms
{
	uint32_t mode;
	uint32_t src_width;
	uint32_t src_height;
	uint32_t dst_width;
	uint32_t dst_height;
};

class HZBPass : public RenderPass
{
  public:
	HZBPass() = default;

	void execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables) override;

  private:

	  std::vector<backend::ImageView> mip_views_;
};
}        // namespace xihe::rendering
