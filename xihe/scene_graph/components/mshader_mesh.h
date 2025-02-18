#pragma once

#include "backend/buffer.h"
#include "backend/shader_module.h"
#include "scene_graph/component.h"
#include "scene_graph/geometry_data.h"

namespace xihe::sg
{
class Material;

struct PackedVertex
{
	glm::vec4 pos;
	glm::vec4 normal;
};

struct MeshDrawCounts
{
	uint32_t meshlet_count;
};

class MshaderMesh : public Component
{
  public:
	MshaderMesh(MeshPrimitiveData &primitive_data, backend::Device &device);

	virtual ~MshaderMesh() = default;

	virtual std::type_index get_type() override;

	backend::Buffer &get_vertex_buffer() const;
	backend::Buffer &get_meshlet_buffer() const;
	backend::Buffer &get_meshlet_vertices_buffer() const;
	backend::Buffer &get_packed_meshlet_indices_buffer() const;
	backend::Buffer &get_mesh_draw_counts_buffer() const;

	void set_material(const Material &material);

	const Material *get_material() const;

	uint32_t get_meshlet_count() const;

	const backend::ShaderVariant &get_shader_variant() const;
	backend::ShaderVariant       &get_mut_shader_variant();

  private:
	void compute_shader_variant();

	void prepare_meshlets(std::vector<Meshlet> &meshlets, const MeshPrimitiveData &primitive_data, backend::Device &device);

	void use_lod_meshlets(std::vector<Meshlet> &meshlets, MeshPrimitiveData &primitive_data, backend::Device &device);

	void use_last_lod_meshlets(std::vector<Meshlet> &meshlets, MeshPrimitiveData &primitive_data, backend::Device &device);

	uint32_t meshlet_count_{0};

	std::unique_ptr<backend::Buffer> vertex_data_buffer_;
	std::unique_ptr<backend::Buffer> meshlet_buffer_;

	std::unique_ptr<backend::Buffer> meshlet_vertices_buffer_;
	std::unique_ptr<backend::Buffer> packed_meshlet_indices_buffer_;

	std::unique_ptr<backend::Buffer> mesh_draw_counts_buffer_;

	const Material *material_{nullptr};

	backend::ShaderVariant shader_variant_;
};
}        // namespace xihe::sg
