#pragma once

#include "render_pass.h"
#include "gpu_lod_scene.h"

namespace xihe::rendering
{
class StreamingPass : public RenderPass
{
public:
	StreamingPass(GpuLoDScene &gpu_lod_scene);

	void execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables) override;

private:
	GpuLoDScene &gpu_lod_scene_;
};
}
