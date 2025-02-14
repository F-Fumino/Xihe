#pragma once

#include "backend/buffer.h"
#include "backend/shader_module.h"
#include "scene_graph/component.h"
#include "scene_graph/geometry_data.h"
#include "mshader_mesh.h"
#include <meshoptimizer.h>
#include "metis.h"

namespace xihe::sg
{
struct MeshletGroup
{
	std::vector<std::size_t> meshlets;
};
/**
 * Connections betweens meshlets
 */
struct MeshletEdge
{
	explicit MeshletEdge(std::size_t a, std::size_t b) :
	    first(std::min(a, b)), second(std::max(a, b))
	{}

	bool operator==(const MeshletEdge &other) const = default;

	const std::size_t first;
	const std::size_t second;
};

struct MeshletEdgeHasher
{
	std::size_t operator()(const MeshletEdge &edge) const
	{
		std::size_t h = edge.first;
		h ^= (100007 * edge.second + 233333) + 0x9e3779b9 + (h << 6) + (h >> 2);

		//h = 10007 * h + edge.second * 23333;
		return h;
	}
};

struct VertexWrapper
{
	const float *vertices = nullptr;
	std::size_t           index    = 0;

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

void generateClusterHierarchy(MeshPrimitiveData &primitive);
}        // namespace xihe::sg