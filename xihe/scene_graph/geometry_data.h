#pragma once
#include <vector>
#include <string>
#include <vulkan/vulkan.hpp>

#include "common/serialize.h"

namespace xihe
{
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

	uint32_t lod;   // for debug
	float cluster_error = 0.0f;
	float parent_error  = std::numeric_limits<float>::infinity();
	uint32_t padding2;

	glm::vec4 bounding_sphere;
	glm::vec4 parent_bounding_sphere;

	template <class Archive>
	void serialize(Archive &archive)
	{
		archive(page_index, page_offset, size, offset, vertices_offset, vertex_indices_offset, triangles_offset, meshlets_offset, lod, cluster_error, parent_error, padding2, bounding_sphere, parent_bounding_sphere);
	}
};

struct Cluster
{
	uint32_t cluster_group_index;
	uint32_t cluster_index;
	uint32_t mesh_draw_index;
	uint32_t padding;

	// for culling

	glm::vec4 bounding_sphere;
	glm::vec3 cone_axis;
	float     cone_cutoff;

	template <class Archive>
	void serialize(Archive &archive)
	{
		archive(cluster_group_index, cluster_index, mesh_draw_index, padding, bounding_sphere, cone_axis, cone_cutoff);
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