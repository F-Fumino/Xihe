#pragma once

#include "gpu_scene.h"
#include "page_table.h"
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

	template <class Archive>
	void serialize(Archive &archive)
	{
		archive(texture_indices, base_color_factor, metallic_roughness_occlusion_factor, meshlet_offset, meshlet_count, mesh_vertex_offset, mesh_triangle_offset);
	}
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

	backend::Buffer &get_vertex_page_state_buffer() const;
	backend::Buffer &get_triangle_page_state_buffer() const;

	backend::Buffer &get_vertex_buffer_address() const;
	backend::Buffer &get_triangle_buffer_address() const;
	backend::Buffer &get_global_meshlet_buffer() const;

	uint32_t get_instance_count() const;

	backend::Device &get_device() const;

	void streaming(backend::CommandBuffer &command_buffer);

  private:
	backend::Device &device_;

	uint32_t instance_count_{};

	std::unique_ptr<backend::Buffer>         vertex_buffer_address_; // address buffer for all vertex buffers
	std::unique_ptr<PageTable<PackedVertex>> vertex_page_table_; // vertex page table

	std::unique_ptr<backend::Buffer> triangle_buffer_address_;
	std::unique_ptr<PageTable<uint32_t>> triangle_page_table_;

	std::unique_ptr<backend::Buffer> global_meshlet_buffer_;

	std::unique_ptr<backend::Buffer> instance_buffer_;

	std::unique_ptr<backend::Buffer> mesh_draws_buffer_;
	std::unique_ptr<backend::Buffer> mesh_bounds_buffer_;

	std::unique_ptr<backend::Buffer> draw_command_buffer_;
	std::unique_ptr<backend::Buffer> draw_counts_buffer_;

	std::unique_ptr<backend::Buffer> vertex_page_state_buffer_;
	std::unique_ptr<backend::Buffer> triangle_page_state_buffer_;
};
}        // namespace xihe
