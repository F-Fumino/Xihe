#pragma once

#include "gpu_scene.h"
#include "render_pass.h"
#include "scene_graph/components/camera.h"

namespace xihe::rendering
{
class MeshPass : public RenderPass
{
public:
	MeshPass(GpuScene &gpu_scene, sg::Camera &camera);

	void execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables) override;

	static void show_meshlet_view(bool show);
	static void show_texture();

	static void freeze_frustum(bool freeze, sg::Camera *camera);

private:
	GpuScene &gpu_scene_;
	sg::Camera &camera_;

	inline static backend::ShaderVariant shader_variant_;

	inline static bool show_debug_view_{false};

	inline static bool      freeze_frustum_{false};
	inline static glm::mat4 frozen_view_;
	inline static glm::vec4 frozen_frustum_planes_[6];
};
}
