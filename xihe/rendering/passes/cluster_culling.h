#pragma once

#include "gpu_lod_scene.h"
#include "render_pass.h"
#include "scene_graph/components/camera.h"

namespace xihe::rendering
{

class ClusterCullingPass : public RenderPass
{
  public:
	ClusterCullingPass(GpuLoDScene &gpu_scene, sg::Camera &camera);

	void execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables) override;

	static void use_lod(bool use);
	static void use_occlusion(bool use);

  private:
	GpuLoDScene &gpu_scene_;
	sg::Camera  &camera_;

	bool is_first_frame_{true};

	inline static bool use_lod_{false};
	inline static bool use_occlusion_{false};

	inline static backend::ShaderVariant shader_variant_;
};
}        // namespace xihe::rendering
