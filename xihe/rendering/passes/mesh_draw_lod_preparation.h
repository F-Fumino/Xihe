#pragma once

#include "render_pass.h"
#include "gpu_lod_scene.h"
#include "scene_graph/components/camera.h"

namespace xihe::rendering
{
class MeshDrawLoDPreparationPass : public RenderPass
{
public:
	MeshDrawLoDPreparationPass(GpuLoDScene &gpu_lod_scene, sg::Camera &camera);

	void execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables) override;

private:
	GpuLoDScene &gpu_lod_scene_;
    sg::Camera  &camera_;
};
}
