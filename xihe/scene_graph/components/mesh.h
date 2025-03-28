#pragma once
#include "scene_graph/component.h"

#include "scene_graph/components/aabb.h"

namespace xihe::sg
{
struct SubMeshData
{
	Material         *material;
	MeshPrimitiveData primitive_data;
};
class Mesh : public Component
{
  public:
	Mesh(const std::string &name);

	virtual ~Mesh() = default;

	void update_bounds(const std::vector<glm::vec3> &vertex_data, const std::vector<uint16_t> &index_data = {});

	virtual std::type_index get_type() override;

	const AABB &get_bounds() const;

	void add_submesh(SubMesh &submesh);

	void add_submesh_data(Material &material, MeshPrimitiveData &&primitive_data);

	const std::vector<SubMesh *> &get_submeshes() const;

	const std::vector<SubMeshData> &get_submeshes_data() const;

	void add_node(Node &node);

	const std::vector<Node *> &get_nodes() const;

  private:
	AABB bounds_;

	std::vector<SubMesh *> submeshes_;

	std::vector<SubMeshData> submeshes_data_;        // used for gpu scene

	std::vector<Node *> nodes_;
};
}        // namespace xihe::sg