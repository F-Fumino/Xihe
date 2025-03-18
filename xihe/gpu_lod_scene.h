#pragma once

#include "gpu_scene.h"
#include "backend/buffer.h"
#include "backend/device.h"
#include "scene_graph/geometry_data.h"
#include "scene_graph/scene.h"

namespace xihe
{

struct MeshLoDDraw
{
	// x = diffuse index, y = roughness index, z = normal index, w = occlusion index.
	glm::uvec4 texture_indices;
	glm::vec4  base_color_factor;
	glm::vec4  metallic_roughness_occlusion_factor;

	uint32_t   meshlet_offset;
	uint32_t   meshlet_count;
	// Global offset into the vertex buffer for all meshlets in this mesh.
	// Individual meshlet vertex offsets are stored in their respective Meshlet structs.
	uint32_t mesh_vertex_offset;
	// Global offset into the triangle buffer for all meshlets in this mesh.
	// Individual meshlet triangle offsets are stored in their respective Meshlet structs.
	uint32_t mesh_triangle_offset;
};

struct MeshLoDData
{
	MeshLoDData(const MeshPrimitiveData &primitive_data);

	std::vector<PackedVertex> vertices;
	std::vector<uint32_t>     triangles;
	std::vector<Meshlet>      meshlets;
	glm::vec4                 bounds;
	uint32_t                  meshlet_count{0};

  private:
	void prepare_meshlets(const MeshPrimitiveData &primitive_data);
};

class GpuLoDScene
{
  public:
	GpuLoDScene(backend::Device &device);

	void initialize(sg::Scene &scene);

	backend::Buffer &get_instance_buffer() const;
	backend::Buffer &get_mesh_draws_buffer() const;
	backend::Buffer &get_mesh_bounds_buffer() const;
	backend::Buffer &get_draw_command_buffer() const;
	backend::Buffer &get_draw_counts_buffer() const;
	backend::Buffer &get_page_request_buffer() const;

	backend::Buffer &get_global_vertex_buffer() const;
	backend::Buffer &get_global_triangle_buffer() const;
	backend::Buffer &get_global_meshlet_buffer() const;

	uint32_t get_instance_count() const;

	backend::Device &get_device() const;

	void streaming(backend::CommandBuffer &command_buffer);
	void swap_in(backend::CommandBuffer &command_buffer, uint32_t page_index);
	void swap_out(backend::CommandBuffer &command_buffer, uint32_t page_index);

  private:
	backend::Device &device_;

	uint32_t instance_count_{};

	uint32_t vertex_page_count_{};

	std::unique_ptr<backend::Buffer> global_vertex_buffer_;
	std::unique_ptr<backend::Buffer> global_triangle_buffer_;
	std::unique_ptr<backend::Buffer> global_meshlet_buffer_;

	std::unique_ptr<backend::Buffer> instance_buffer_;

	std::unique_ptr<backend::Buffer> mesh_draws_buffer_;
	std::unique_ptr<backend::Buffer> mesh_bounds_buffer_;

	std::unique_ptr<backend::Buffer> draw_command_buffer_;
	std::unique_ptr<backend::Buffer> draw_counts_buffer_;

	std::unique_ptr<backend::Buffer> page_request_buffer_;

	std::vector<PackedVertex>                     global_vertex_sets_;
	std::vector<std::vector<PackedVertex>> global_vertices_;
	std::vector<std::unique_ptr<backend::Buffer>> staging_vertex_buffers_;
	std::vector<bool>                             page_swapped_in_;
};
}        // namespace xihe
