#pragma once

#include "backend/buffer.h"
#include "backend/shader_module.h"
#include "scene_graph/component.h"
#include "scene_graph/geometry_data.h"
#include <meshoptimizer.h>
#include "metis.h"

namespace xihe::sg
{
/**
 * Connections betweens meshlets
 */

struct VertexWrapper
{
	const float *vertices = nullptr;
	size_t       index    = 0;

	VertexWrapper() :
	    vertices(nullptr), index(0)
	{}

	VertexWrapper(const float *vertices, const std::size_t index) :
	    vertices(vertices), index(index)
	{}

	glm::vec3 getPosition() const
	{
		return glm::vec3(vertices[3 * index], vertices[3 * index + 1], vertices[3 * index + 2]);
	}
};

void generate_cluster_hierarchy(const MeshPrimitiveData &primitive, std::vector<uint32_t> &scene_data, std::vector<ClusterGroup> &meshletgroups, std::vector<Cluster> &clusters);
}        // namespace xihe::sg