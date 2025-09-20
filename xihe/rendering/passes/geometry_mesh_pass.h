#pragma once

#include "gpu_lod_scene.h"
#include "render_pass.h"
#include "scene_graph/components/camera.h"

namespace xihe::rendering
{

class GeometryMeshPass : public RenderPass
{
public:
	GeometryMeshPass(GpuLoDScene &gpu_scene, sg::Camera &camera);

	void execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables) override;

	static void show_meshlet_view(bool show);
	static void show_lod_view(bool show);
	static void show_line(bool show);
	static void show_texture();

private:
	GpuLoDScene &gpu_scene_;
	sg::Camera &camera_;

	inline static backend::ShaderVariant shader_variant_;

	inline static bool show_debug_view_{false};
	inline static bool show_lod_view_{false};
	inline static vk::PolygonMode polygon_mode_ = vk::PolygonMode::eFill;
};
}
