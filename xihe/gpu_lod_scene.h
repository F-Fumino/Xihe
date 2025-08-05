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

	uint32_t   cluster_offset;
	uint32_t   cluster_count;
	uint32_t   padding1;
	uint32_t   padding2;

	template <class Archive>
	void serialize(Archive &archive)
	{
		archive(texture_indices, base_color_factor, metallic_roughness_occlusion_factor, cluster_offset, cluster_count, padding1, padding2);
	}
};

struct MeshLoDData
{
	MeshLoDData(const MeshPrimitiveData &primitive_data);

	std::vector<uint32_t>     scene_data;
	std::vector<ClusterGroup> cluster_groups;
	std::vector<Cluster>      clusters;

	//std::vector<PackedVertex> vertices;
	//std::vector<uint32_t>     triangles;
	//std::vector<Meshlet>      meshlets;
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

	float get_lod_threshold() const;

	backend::Buffer &get_scene_data_buffer_address() const;
	backend::Buffer &get_cluster_group_buffer() const;
	backend::Buffer &get_cluster_buffer() const;

	backend::Buffer &get_instance_buffer() const;
	backend::Buffer &get_mesh_draws_buffer() const;
	backend::Buffer &get_mesh_bounds_buffer() const;
	backend::Buffer &get_draw_command_buffer() const;
	backend::Buffer &get_draw_counts_buffer() const;

	backend::Buffer &get_page_state_buffer() const;

	backend::Buffer &get_valid_data_size_buffer() const;

	uint32_t get_instance_count() const;

	double get_page_table_time() const;
	double get_bind_time() const;
	double get_page_table_hit_probability() const;
	double get_memory_utilization() const;

	backend::Device &get_device() const;

	void streaming(backend::CommandBuffer &command_buffer);

  private:
	backend::Device &device_;

	uint32_t instance_count_{};

	double sum_utilization_{0};

	float lod_threshold_ = 0.05f;

	std::unique_ptr<backend::Buffer>     scene_data_buffer_address_; // address buffer for all vertex buffers
	std::unique_ptr<PageTable<uint32_t>> scene_data_page_table_;            // scene page table

	std::unique_ptr<backend::Buffer> cluster_group_buffer_;
	std::unique_ptr<backend::Buffer> cluster_buffer_;

	std::unique_ptr<backend::Buffer> instance_buffer_;

	std::unique_ptr<backend::Buffer> mesh_draws_buffer_;
	std::unique_ptr<backend::Buffer> mesh_bounds_buffer_;

	std::unique_ptr<backend::Buffer> draw_command_buffer_;
	std::unique_ptr<backend::Buffer> draw_counts_buffer_;

	std::unique_ptr<backend::Buffer> page_state_buffer_;

	std::unique_ptr<backend::Buffer> valid_data_size_buffer_;
};
}        // namespace xihe
