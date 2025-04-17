#pragma once

#include "render_pass.h"
#include "gpu_scene.h"
#include "scene_graph/components/camera.h"

namespace xihe::rendering
{
class MeshDrawPreparationPass : public RenderPass
{
public:
	MeshDrawPreparationPass(GpuScene &gpu_scene, sg::Camera &camera);

	void execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables) override;

private:
	GpuScene &gpu_scene_;
    sg::Camera &camera_;
};
}
