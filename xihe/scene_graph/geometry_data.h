#pragma once
#include <vector>
#include <string>
#include <vulkan/vulkan.hpp>

#include "common/serialize.h"

namespace xihe
{
struct MeshletGroup
{
	std::vector<size_t> meshlets;
};

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

		// h = 10007 * h + edge.second * 23333;
		return h;
	}
};

struct PackedVertex
{
	glm::vec4 pos;
	glm::vec4 normal;

	template<class Archive>
	void serialize(Archive &archive)
	{
		archive(pos, normal);
	}
};

struct VertexAttribute
{
	vk::Format format = vk::Format::eUndefined;
	std::uint32_t stride = 0;
	std::uint32_t offset = 0;
};

struct VertexAttributeData
{
	// std::string          name;
	vk::Format           format;
	uint32_t             stride;
	std::vector<uint8_t> data;
};

// 这里的单位统一成 uint32_t 可能比较好
struct ClusterGroup
{
	uint32_t page_index;
	uint32_t page_offset;
	uint32_t size;                    // for allocate page and debug
	uint32_t offset;                  // for allocate page
	
	uint32_t vertices_offset;
	uint32_t vertex_indices_offset;
	uint32_t triangles_offset;
	uint32_t meshlets_offset;

	// for lod

	uint32_t  lod;         // for debug
	float     parent_error = std::numeric_limits<float>::infinity();
	float     pdd1;
	float     pdd2;

	glm::vec4 parent_bounding_sphere;

	template <class Archive>
	void serialize(Archive &archive)
	{
		archive(page_index, page_offset, size, offset, vertices_offset, vertex_indices_offset, triangles_offset, meshlets_offset, lod, parent_error, pdd1, pdd2, parent_bounding_sphere);
	}
};

struct Cluster
{
	uint32_t cluster_group_index;
	uint32_t cluster_index;
	uint32_t mesh_draw_index;

	// for lod

	float cluster_error = 0.0f;
	glm::vec4 lod_bounding_sphere;

	// for culling

	glm::vec4 bounding_sphere;
	glm::vec3 cone_axis;
	float     cone_cutoff;

	glm::vec3 bbmin;
	float     pdd1;
	glm::vec3 bbmax;
	uint32_t  occlusion = 0;

	template <class Archive>
	void serialize(Archive &archive)
	{
		archive(cluster_group_index, cluster_index, mesh_draw_index, cluster_error, lod_bounding_sphere, bounding_sphere, cone_axis, cone_cutoff, bbmin, pdd1, bbmax, occlusion);
	}
};

struct Meshlet
{
	uint32_t vertex_offset;
	uint32_t triangle_offset;

	uint32_t vertex_count;
	uint32_t triangle_count;

	glm::vec3 center;
	float     radius;
	glm::vec3 cone_axis;
	float     cone_cutoff;

	uint32_t   mesh_draw_index;
	glm::uvec3 padding;

	uint32_t vertex_page_index1;
	uint32_t vertex_page_index2;
	uint32_t triangle_page_index1;
	uint32_t triangle_page_index2;

	// lod

	glm::vec3 cone_apex;
	uint32_t  lod;

	float parent_error  = std::numeric_limits<float>::infinity();
	float cluster_error = 0.0f;
	uint32_t cluster_group_index;
	float pdd2;

	glm::vec4 bounding_sphere;
	glm::vec4 parent_bounding_sphere;

	template <class Archive>
	void serialize(Archive &archive)
	{
		archive(vertex_offset, triangle_offset, vertex_count, triangle_count, center, radius, cone_axis, cone_cutoff, mesh_draw_index, padding, vertex_page_index1, vertex_page_index2, triangle_page_index1, triangle_page_index2, cone_apex, lod, parent_error, cluster_error, cluster_group_index, pdd2, parent_bounding_sphere);
	}
};

struct MeshPrimitiveData
{
	std::string                      name;
	uint32_t                         vertex_count;
	std::unordered_map<std::string, VertexAttributeData> attributes;
	std::vector<uint8_t>             indices;
	vk::IndexType                    index_type;
	uint32_t                         index_count;
};
}