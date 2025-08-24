#pragma once

#include "gpu_lod_scene.h"
#include "render_pass.h"
#include "mesh_pass.h"
#include "scene_graph/components/camera.h"

namespace xihe::rendering
{
struct OcclusionUniform
{
	uint32_t width;
	uint32_t height;
	uint32_t mip_count;
	uint32_t is_first_frame;
	float    depth_bias;
};

class MeshLoDPass : public RenderPass
{
public:
	MeshLoDPass(GpuLoDScene &gpu_scene, sg::Camera &camera);

	void execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables) override;

	static void show_meshlet_view(bool show);
	static void use_lod(bool use); 
	static void show_lod_view(bool show);
	static void show_line(bool show);
	static void show_texture();
	static void use_occlusion(bool use);

	static void freeze_frustum(bool freeze, sg::Camera *camera);

private:
	GpuLoDScene &gpu_scene_;
	sg::Camera &camera_;

	bool is_first_frame_{true};

	inline static backend::ShaderVariant shader_variant_;

	inline static bool show_debug_view_{false};
	inline static bool use_lod_{false};
	inline static bool show_lod_view_{false};
	inline static bool use_occlusion_{false};
	inline static vk::PolygonMode polygon_mode_ = vk::PolygonMode::eFill;

	inline static bool      freeze_frustum_{false};
	inline static glm::mat4 frozen_view_;
	inline static glm::vec4 frozen_frustum_planes_[6];
};
}
