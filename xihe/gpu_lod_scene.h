#pragma once

#include <list>

#include "gpu_scene.h"
#include "backend/buffer.h"
#include "backend/device.h"
#include "scene_graph/geometry_data.h"
#include "scene_graph/scene.h"

namespace xihe
{
template <typename DataType>
class PageTable : public backend::allocated::SparseResources
{
  public:
	PageTable(backend::Device &device, uint32_t table_page_num, vk::DeviceSize page_size);

	void init(uint32_t buffer_count, uint32_t buffer_page_count);
	void allocate_pages();

	void execute(backend::CommandBuffer &command_buffer, uint16_t *page_request);
	
	int32_t swap_in(uint16_t *page_request);

	backend::Device &device_;

	const backend::Queue *sparse_queue_{nullptr};

	uint32_t buffer_page_count_{};        // number of pages in the all buffers
	uint32_t buffer_count_{};             // number of buffers

	std::vector<std::unique_ptr<backend::Buffer>> buffers_;     // all buffers
	std::vector<std::vector<DataType>>            data_;        // every buffer page's data
	std::vector<std::unique_ptr<backend::Buffer>> staging_buffers_;        // every buffer page's staging buffer, for data transfer
	std::vector<uint32_t>                         page_swapped_in_;        // ervery buffer page's table index, -1 for not in the table
	
	// random
	std::list<uint32_t> free_list_;

	// LRU
	std::list<uint32_t> lru_list_;
	std::unordered_map<int, std::list<uint32_t>::iterator> lru_page_table_;
	
	std::vector<uint32_t> table_to_buffer_;        // every table page's vertex page index
	std::vector<uint32_t> buffer_to_table_;        // every vertex page's buffer page index
};

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
