#pragma once
#include <vector>
#include <string>
#include <vulkan/vulkan.hpp>

namespace xihe
{
struct PackedVertex
{
	glm::vec4 pos;
	glm::vec4 normal;
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

struct Meshlet
{
	uint32_t vertex_offset;
	uint32_t triangle_offset;

	/* number of vertices and triangles used in the meshlet; data is stored in consecutive range defined by offset and count */
	uint32_t vertex_count;
	uint32_t triangle_count;

	glm::vec3 center;
	float     radius;
	glm::vec3 cone_axis;
	float     cone_cutoff;

	uint32_t   mesh_draw_index;
	glm::uvec3 padding;

	// lod

	glm::vec3 cone_apex;
	uint32_t  lod;

	float parentError  = std::numeric_limits<float>::infinity();
	float clusterError = 0.0f;
	float pdd1;
	float pdd2;

	glm::vec4 parentBoundingSphere;
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