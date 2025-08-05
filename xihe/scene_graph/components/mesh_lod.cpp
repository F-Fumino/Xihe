#include "mesh_lod.h"
#include "kdtree.h"
#include "vcg_mesh.h"
#include "metis.h"
#include <common/timer.h>
#include <glm/gtx/norm.hpp>
#include <tbb/parallel_for.h>
#include <scene_graph/geometry_data.h>
#include <meshoptimizer.h>

//#define USE_MESHOPT
#define USE_VCG

#define USE_WELDING 1

#define THRESHOLD 0.5f
#define SIMPLIFY_SCALE 30.0f

namespace xihe::sg
{
glm::vec4 convert_to_vec4(const std::vector<uint8_t> &data, uint32_t offset, float padding = 1.0f)
{
	if (data.size() < offset + 3 * sizeof(float))
		throw std::runtime_error("Data size is too small for conversion to vec4.");

	float x, y, z;
	std::memcpy(&x, &data[offset], sizeof(float));
	std::memcpy(&y, &data[offset + sizeof(float)], sizeof(float));
	std::memcpy(&z, &data[offset + 2 * sizeof(float)], sizeof(float));

	return {x, y, z, padding};
}

static std::vector<bool> find_boundary_vertices(const std::vector<uint32_t> &meshlet_vertices, const std::vector<uint32_t> &triangles, std::span<Meshlet> meshlets, uint32_t vertex_count)
{
	std::vector<bool> boundary_vertices;
	boundary_vertices.resize(vertex_count);

	for (size_t i = 0; i < vertex_count; i++)
	{
		boundary_vertices[i] = false;
	}

	std::unordered_map<MeshletEdge, std::unordered_set<size_t>, MeshletEdgeHasher> edges_to_meshlets;

	for (size_t meshlet_index = 0; meshlet_index < meshlets.size(); meshlet_index++)
	{
		const auto &meshlet          = meshlets[meshlet_index];
		auto        get_vertex_index = [&](size_t index) {
            uint32_t packed_vertex_index = triangles[index / 3 + meshlet.triangle_offset];
            uint8_t  vertex_index        = (packed_vertex_index >> ((index % 3) * 8)) & 0xFF;
            return meshlet_vertices[meshlet.vertex_offset + vertex_index];
		};

		const size_t triangle_count = meshlet.triangle_count;
		for (size_t triangle_index = 0; triangle_index < triangle_count; triangle_index++)
		{
			for (size_t i = 0; i < 3; i++)
			{
				MeshletEdge edge{get_vertex_index(i + triangle_index * 3), get_vertex_index(((i + 1) % 3) + triangle_index * 3)};
				if (edge.first != edge.second)
				{
					edges_to_meshlets[edge].insert(meshlet_index);
				}
			}
		}
	}

	for (const auto &[edge, meshlet_indices] : edges_to_meshlets)
	{
		if (meshlet_indices.size() > 1)
		{
			boundary_vertices[edge.first]  = true;
			boundary_vertices[edge.second] = true;
		}
	}

	return boundary_vertices;
}

static std::vector<std::int64_t> merge_by_distance(const std::vector<glm::vec3> &vertex_positions, const std::vector<glm::vec3> &vertex_normals, std::vector<glm::vec2> &vertex_uvs, const std::vector<bool> &boundary, std::span<const VertexWrapper> group_vertices_pre_weld, float max_distance, float max_normal_distance, float max_uv_distance, const KDTree<VertexWrapper> &kdtree)
{
	std::vector<std::int64_t> vertex_remap;
	const std::size_t         vertex_count = vertex_positions.size();
	vertex_remap.resize(vertex_count);

#ifndef USE_WELDING
	for (int i = 0; i < vertex_count; i++)
	{
		vertex_remap[i] = i;
	}
#else
	for (auto &v : vertex_remap)
	{
		v = -1;
	}

	std::vector<std::vector<std::size_t>> neighbors_for_all_vertices;
	neighbors_for_all_vertices.resize(group_vertices_pre_weld.size());

	tbb::parallel_for(std::size_t(0), group_vertices_pre_weld.size(), [&](std::size_t v) {
		const VertexWrapper &current_vertex_wrapped = group_vertices_pre_weld[v];
		if (boundary[current_vertex_wrapped.index])
		{
			return;
		}
		kdtree.getNeighbors(neighbors_for_all_vertices[v], current_vertex_wrapped, max_distance);
	});

	for (std::int64_t v = 0; v < group_vertices_pre_weld.size(); v++)
	{
		std::int64_t replacement            = -1;
		const auto  &current_vertex_wrapped = group_vertices_pre_weld[v];
		 if (!boundary[current_vertex_wrapped.index])
		{
			auto            &neighbors          = neighbors_for_all_vertices[v];
			const glm::vec3 &current_vertex_pos = current_vertex_wrapped.getPosition();

			/*const float *vertex_normal = vertex_normals + current_vertex_wrapped.index * 3;*/
			const glm::vec3 current_vertex_normal = glm::normalize(vertex_normals[current_vertex_wrapped.index]);

			glm::vec2        current_vertex_uv  = glm::vec2(0.0, 0.0);

			//if (vertex_uvs)
			//{
				/*const float *vertex_uv = vertex_uvs + current_vertex_wrapped.index * 2;*/
			current_vertex_uv = vertex_uvs[current_vertex_wrapped.index];
			//}

			float max_distance_sq        = max_distance * max_distance;
			float max_normal_distance_sq = max_normal_distance;
			float max_uv_distance_sq     = max_uv_distance * max_uv_distance;

			for (const std::size_t &neighbor : neighbors)
			{
				if (vertex_remap[group_vertices_pre_weld[neighbor].index] == -1)
				{
					continue;
				}
				/*auto             other_vertex_wrapped = VertexWrapper(vertex_positions, vertex_remap[group_vertices_pre_weld[neighbor].index]);
				const glm::vec3 &other_vertex_pos     = other_vertex_wrapped.getPosition();*/
				const glm::vec3 &other_vertex_pos = vertex_positions[vertex_remap[group_vertices_pre_weld[neighbor].index]];

				/*const float *vertex_normal       = vertex_normals + other_vertex_wrapped.index * 3;
				const glm::vec3 other_vertex_normal = glm::normalize(glm::vec3(vertex_normal[0], vertex_normal[1], vertex_normal[2]));*/
				const glm::vec3 &other_vertex_normal = vertex_normals[vertex_remap[group_vertices_pre_weld[neighbor].index]];

				const float      vertex_distance_sq   = glm::distance2(current_vertex_pos, other_vertex_pos);
				// [0, 1], 0 means 2 normal equal
				float vertex_normal_distance_sq = (1 - glm::dot(current_vertex_normal, other_vertex_normal)) / 2;

				vertex_normal_distance_sq = vertex_normal_distance_sq <= 0 ? 0 : vertex_normal_distance_sq;
				vertex_normal_distance_sq = vertex_normal_distance_sq >= 1 ? 1 : vertex_normal_distance_sq;

				if (vertex_distance_sq <= max_distance_sq && vertex_normal_distance_sq <= max_normal_distance_sq)
				/*if (vertex_distance_sq <= max_distance_sq)*/
				{
					//if (!vertex_uvs)
					//{
						/*replacement     = vertex_remap[group_vertices_pre_weld[neighbor].index];
						max_distance_sq = vertex_distance_sq;
						max_normal_distance_sq = vertex_normal_distance_sq;
					}*/
					//else
					//{
						/*const float     *vertex_uv             = vertex_uvs + group_vertices_pre_weld[neighbor].index * 2;*/
					const glm::vec2 &other_vertex_uv       = vertex_uvs[vertex_remap[group_vertices_pre_weld[neighbor].index]];
						const float      vertex_uv_distance_sq = glm::distance2(current_vertex_uv, other_vertex_uv);
						if (vertex_uv_distance_sq <= max_uv_distance_sq)
						{
							replacement        = vertex_remap[group_vertices_pre_weld[neighbor].index];
							max_distance_sq    = vertex_distance_sq;
							max_uv_distance_sq = vertex_uv_distance_sq;
							max_normal_distance_sq = vertex_normal_distance_sq;
						}
					//}
				}
			}
		}

		if (replacement == -1)
		{
			vertex_remap[current_vertex_wrapped.index] = current_vertex_wrapped.index;
		}
		else
		{
			vertex_remap[current_vertex_wrapped.index] = replacement;
		}
	}
#endif

	return vertex_remap;
}

static std::vector<MeshletGroup> group_meshlets_remap(const std::vector<uint32_t> &meshlet_vertices, const std::vector<uint32_t> &triangles, std::span<Meshlet> meshlets, std::span<const std::int64_t> vertex_remap)
{
	auto group_with_all_meshlets = [&]() {
		MeshletGroup group;
		for (int i = 0; i < meshlets.size(); ++i)
		{
			group.meshlets.push_back(i);
		}
		return std::vector{group};
	};

	if (meshlets.size() < 16)
	{
		return group_with_all_meshlets();
	}

	std::unordered_map<MeshletEdge, std::vector<std::size_t>, MeshletEdgeHasher> edges_to_meshlets;
	std::unordered_map<std::size_t, std::vector<MeshletEdge>>                    meshlets_to_edges;

	for (std::size_t meshlet_index = 0; meshlet_index < meshlets.size(); meshlet_index++)
	{
		const auto &meshlet          = meshlets[meshlet_index];
		auto        get_vertex_index = [&](std::size_t index) {
            uint32_t packed_vertex_index = triangles[index / 3 + meshlet.triangle_offset];
            uint8_t  vertex_index        = (packed_vertex_index >> ((index % 3) * 8)) & 0xFF;
            return static_cast<std::size_t>(
                vertex_remap[meshlet_vertices[meshlet.vertex_offset + vertex_index]]);
		};

		const std::size_t triangle_count = meshlet.triangle_count;
		for (std::size_t triangle_index = 0; triangle_index < triangle_count; triangle_index++)
		{
			for (std::size_t i = 0; i < 3; i++)
			{
				MeshletEdge edge{get_vertex_index(i + triangle_index * 3), get_vertex_index(((i + 1) % 3) + triangle_index * 3)};
				if (edge.first != edge.second)
				{
					edges_to_meshlets[edge].push_back(meshlet_index);
					meshlets_to_edges[meshlet_index].emplace_back(edge);
				}
			}
		}
	}

	std::erase_if(edges_to_meshlets, [&](const auto &pair) {
		return pair.second.size() <= 1;
	});

	if (edges_to_meshlets.empty())
	{
		return group_with_all_meshlets();
	}

	std::vector<MeshletGroup> groups;

	idx_t vertex_count = meshlets.size();
	idx_t ncon         = 1;
	idx_t nparts       = meshlets.size() / 8;
	assert(nparts > 1);
	idx_t options[METIS_NOPTIONS];
	METIS_SetDefaultOptions(options);
	options[METIS_OPTION_OBJTYPE]   = METIS_OBJTYPE_CUT;
	options[METIS_OPTION_CCORDER]   = 1;
	options[METIS_OPTION_NUMBERING] = 0;

	std::vector<idx_t> partition(vertex_count);

	std::vector<idx_t> x_adjacency;
	x_adjacency.reserve(vertex_count + 1);

	std::vector<idx_t> edge_adjacency;
	std::vector<idx_t> edge_weights;

	for (std::size_t meshlet_index = 0; meshlet_index < meshlets.size(); meshlet_index++)
	{
		std::size_t start_index_in_edge_adjacency = edge_adjacency.size();
		for (const auto &edge : meshlets_to_edges[meshlet_index])
		{
			auto connections_iter = edges_to_meshlets.find(edge);
			if (connections_iter == edges_to_meshlets.end())
			{
				continue;
			}
			const auto &connections = connections_iter->second;
			for (const auto &connected_meshlet : connections)
			{
				if (connected_meshlet != meshlet_index)
				{
					auto existing_edge_iter = std::find(edge_adjacency.begin() + start_index_in_edge_adjacency, edge_adjacency.end(), connected_meshlet);
					if (existing_edge_iter == edge_adjacency.end())
					{
						edge_adjacency.emplace_back(connected_meshlet);
						edge_weights.emplace_back(1);
					}
					else
					{
						std::ptrdiff_t d = std::distance(edge_adjacency.begin(), existing_edge_iter);
						edge_weights[d]++;
					}
				}
			}
		}
		x_adjacency.push_back(start_index_in_edge_adjacency);
	}
	x_adjacency.push_back(edge_adjacency.size());

	idx_t edge_cut;
	int   result = METIS_PartGraphKway(&vertex_count,
	                                   &ncon,
	                                   x_adjacency.data(),
	                                   edge_adjacency.data(),
	                                   NULL,
	                                   NULL,
	                                   edge_weights.data(),
	                                   &nparts,
	                                   NULL,
	                                   NULL,
	                                   options,
	                                   &edge_cut,
	                                   partition.data());

	assert(result == METIS_OK);

	groups.resize(nparts);
	for (std::size_t i = 0; i < meshlets.size(); i++)
	{
		idx_t partition_number = partition[i];
		groups[partition_number].meshlets.push_back(i);
	}

	return groups;
}

PackedVertex get_packed_vertex(const MeshPrimitiveData &primitive_data, uint32_t index)
{
	const float *vertex_positions = reinterpret_cast<const float *>(primitive_data.attributes.at("position").data.data());
	const float *vertex_normals   = reinterpret_cast<const float *>(primitive_data.attributes.at("normal").data.data());
	const float *vertex_texcoords = nullptr;

	if (primitive_data.attributes.find("texcoord_0") != primitive_data.attributes.end())
	{
		vertex_texcoords = reinterpret_cast<const float *>(primitive_data.attributes.at("texcoord_0").data.data());
	}

	glm::vec4 pos = glm::vec4(vertex_positions[index * 3 + 0], vertex_positions[index * 3 + 1], vertex_positions[index * 3 + 2], 0.0f);

	/*glm::vec4 normal  = glm::vec4(vertex_normals[index * 3 + 0], vertex_normals[index * 3 + 1], vertex_normals[index * 3 + 2], 0.0);
	normal            = glm::normalize(normal);*/

	glm::vec4 normal = glm::vec4(0.0, 0.0, 0.0, 0.0);

	if (vertex_texcoords)
	{
		pos.w    = vertex_texcoords[index * 2 + 0];
		normal.w = vertex_texcoords[index * 2 + 1];
	}

	return {pos, normal};
}

PackedVertex get_packed_vertex(const std::vector<glm::vec3>& vertex_positions, const std::vector<glm::vec3>& vertex_normals, const std::vector<glm::vec2>& vertex_uvs, uint32_t index)
{
	glm::vec4 pos    = glm::vec4(vertex_positions[index].xyz, vertex_uvs[index].x);
	glm::vec4 normal = glm::vec4(vertex_normals[index].xyz, vertex_uvs[index].y);
	/*glm::vec4 normal = glm::vec4(0.0f, 0.0f, 0.0f, vertex_uvs[index].y);*/
	return {pos, normal};
}

inline glm::vec3 get_vertex_position(const MeshPrimitiveData &primitive_data, uint32_t index)
{
	const float *vertex_positions = reinterpret_cast<const float *>(primitive_data.attributes.at("position").data.data());
	return glm::vec3(vertex_positions[index * 3 + 0], vertex_positions[index * 3 + 1], vertex_positions[index * 3 + 2]);
}

//static void append_meshlets(std::vector<glm::vec3>& vertex_positions, std::vector<glm::vec3>& vertex_normals, std::vector<glm::vec2> vertex_uvs, std::vector<uint32_t>& meshlet_vertices, std::vector<uint32_t>& meshlet_triangles, std::vector<Meshlet>& meshlets, MyMesh &mesh)
//{
//	constexpr std::size_t max_vertices  = 64;
//	constexpr std::size_t max_triangles = 124;
//	const float           cone_weight   = 0.0f;
//
//	const std::size_t max_meshlets = meshopt_buildMeshletsBound(index_buffer.size(), max_vertices, max_triangles);
//
//	std::vector<meshopt_Meshlet> local_meshlets(max_meshlets);
//	std::vector<unsigned int>    meshlet_vertex_indices(max_meshlets * max_vertices);
//	std::vector<unsigned char>   meshlet_triangle_indices(max_meshlets * max_triangles * 3);
//
//	size_t meshlet_count = meshopt_buildMeshlets(
//	    local_meshlets.data(),
//	    meshlet_vertex_indices.data(),
//	    meshlet_triangle_indices.data(),
//	    index_buffer.data(),
//	    index_buffer.size(),
//	    vertex_positions,
//	    vertex_positions_count,
//	    sizeof(float) * 3,
//	    max_vertices,
//	    max_triangles,
//	    cone_weight);
//
//	local_meshlets.resize(meshlet_count);
//
//	const meshopt_Meshlet &last           = local_meshlets[meshlet_count - 1];
//	const std::size_t      vertex_count   = last.vertex_offset + last.vertex_count;
//	std::size_t            triangle_count = last.triangle_offset / 3 + last.triangle_count;
//
//	const std::size_t global_vertex_offset   = meshlet_vertices.size();
//	const std::size_t global_triangle_offset = meshlet_triangles.size();
//	const std::size_t global_meshlet_offset  = meshlets.size();
//
//	meshlet_vertices.resize(global_vertex_offset + vertex_count);
//	meshlet_triangles.resize(global_triangle_offset + triangle_count);
//	meshlets.resize(global_meshlet_offset + meshlet_count);
//
//
//	size_t vertex_count = mesh.vert.size();
//}

static void append_meshlets(std::vector<uint32_t> &meshlet_vertices, std::vector<uint32_t> &meshlet_triangles, std::vector<Meshlet> &meshlets, const float *vertex_positions, uint32_t vertex_positions_count, std::span<std::uint32_t> index_buffer, float cluster_error, glm::vec4 bounding_sphere, std::span<size_t> vertex_remap = std::span<size_t>())
{
	constexpr std::size_t max_vertices  = 64;
	constexpr std::size_t max_triangles = 124;
	const float           cone_weight   = 0.0f;

	const std::size_t max_meshlets = meshopt_buildMeshletsBound(index_buffer.size(), max_vertices, max_triangles);

	std::vector<meshopt_Meshlet> local_meshlets(max_meshlets);
	std::vector<unsigned int>    meshlet_vertex_indices(max_meshlets * max_vertices);
	std::vector<unsigned char>   meshlet_triangle_indices(max_meshlets * max_triangles * 3);

	size_t meshlet_count = meshopt_buildMeshlets(
	    local_meshlets.data(),
	    meshlet_vertex_indices.data(),
	    meshlet_triangle_indices.data(),
	    index_buffer.data(),
	    index_buffer.size(),
	    vertex_positions,
	    vertex_positions_count,
	    sizeof(float) * 3,
	    max_vertices,
	    max_triangles,
	    cone_weight);

	local_meshlets.resize(meshlet_count);

	const meshopt_Meshlet &last           = local_meshlets[meshlet_count - 1];
	const std::size_t      vertex_count   = last.vertex_offset + last.vertex_count;
	std::size_t            triangle_count = last.triangle_offset / 3 + last.triangle_count;
	
	const std::size_t      global_vertex_offset   = meshlet_vertices.size();
	const std::size_t      global_triangle_offset = meshlet_triangles.size();
	const std::size_t      global_meshlet_offset  = meshlets.size();

	meshlet_vertices.resize(global_vertex_offset + vertex_count);
	meshlet_triangles.resize(global_triangle_offset + triangle_count);
	meshlets.resize(global_meshlet_offset + meshlet_count);

	if (vertex_remap.empty())
	{
		tbb::parallel_for(std::size_t(0), vertex_count, [&](std::size_t index) {
			meshlet_vertices[global_vertex_offset + index] = meshlet_vertex_indices[index];
		});
	}
	else
	{
		tbb::parallel_for(std::size_t(0), vertex_count, [&](std::size_t index) {
			meshlet_vertices[global_vertex_offset + index] = vertex_remap[meshlet_vertex_indices[index]];
		});
	}

	tbb::parallel_for(std::size_t(0), triangle_count, [&](std::size_t index) {
		uint8_t idx0 = meshlet_triangle_indices[index * 3 + 0];
		uint8_t idx1 = meshlet_triangle_indices[index * 3 + 1];
		uint8_t idx2 = meshlet_triangle_indices[index * 3 + 2];

		uint32_t packed_triangle = idx0 | (idx1 << 8) | (idx2 << 16);

		meshlet_triangles[global_triangle_offset + index] = packed_triangle;
	});

	tbb::parallel_for(std::size_t(0), meshlet_count, [&](std::size_t index) {
		auto &local_meshlet = local_meshlets[index];
		auto &meshlet       = meshlets[global_meshlet_offset + index];

		meshlet.vertex_offset = global_vertex_offset + local_meshlet.vertex_offset;
		meshlet.vertex_count  = local_meshlet.vertex_count;

		meshlet.triangle_offset = global_triangle_offset + local_meshlet.triangle_offset / 3;
		meshlet.triangle_count  = local_meshlet.triangle_count;

		meshopt_Bounds meshlet_bounds = meshopt_computeMeshletBounds(
		    meshlet_vertex_indices.data() + local_meshlet.vertex_offset,
		    meshlet_triangle_indices.data() + local_meshlet.triangle_offset,
		    local_meshlet.triangle_count, vertex_positions, vertex_positions_count, sizeof(float) * 3);

		meshlet.cone_axis   = glm::vec3(meshlet_bounds.cone_axis[0], meshlet_bounds.cone_axis[1], meshlet_bounds.cone_axis[2]);
		meshlet.cone_cutoff = meshlet_bounds.cone_cutoff;
		meshlet.cluster_error = cluster_error;
		meshlet.bounding_sphere = bounding_sphere;
	});
}

// 如果是lod 0就不要有顶点映射
static void append_meshlet_groups(const std::vector<glm::vec3>& vertex_positions, const std::vector<glm::vec3>& vertex_normals, const std::vector<glm::vec2>& vertex_uvs, std::vector<uint32_t> &scene_data, std::vector<ClusterGroup> &cluster_groups, std::vector<Cluster> &clusters, const std::vector<uint32_t> &meshlet_vertices, const std::vector<uint32_t> &meshlet_triangles, std::span<Meshlet> &previous_level_meshlets, const MeshletGroup &group, const std::vector<std::int64_t> &merge_vertex_remap, uint32_t lod)
{
	// cluster group的结构：
	// |顶点数据（无重复顶点）| meshlet_indices（每个meshlet有哪些顶点）| 索引数据 | 每个meshlet的信息 

	Timer append_timer;
	append_timer.start();

	Timer meshlet_timer;
	meshlet_timer.start();

	std::vector<PackedVertex> group_vertices;
	std::vector<uint32_t>     group_vertex_indices;
	std::vector<uint32_t>     group_triangles;
	std::vector<Meshlet>      group_meshlets;

	group_vertices.reserve(group.meshlets.size() * 64);
	group_vertex_indices.reserve(group.meshlets.size() * 64);
	group_triangles.reserve(group.meshlets.size() * 124);
	group_meshlets.reserve(group.meshlets.size());

	//auto vertex_positions = reinterpret_cast<const float *>(primitive_data.attributes.at("position").data.data());

	std::unordered_map<std::size_t, std::size_t> mesh_to_group_vertex_remap;

	uint32_t cluster_index = 0;

	double parallel_time = 0;
	double cluster_time  = 0;

	for (const auto& meshlet_index : group.meshlets)
	{
		Timer parallel_timer;
		parallel_timer.start();

		uint32_t previous_vertex_indices_size = group_vertex_indices.size();
		uint32_t previous_triangles_size      = group_triangles.size();

		auto &meshlet = previous_level_meshlets[meshlet_index];
		meshlet.cluster_group_index = cluster_groups.size();

		glm::vec3 cluster_min{+INFINITY, +INFINITY, +INFINITY};
		glm::vec3 cluster_max{-INFINITY, -INFINITY, -INFINITY};

		std::vector<int32_t> used(64 * group.meshlets.size(), -1);

		for (std::size_t j = 0; j < meshlet.triangle_count * 3; j += 3)
		{
			auto get_vertex_index = [&](std::size_t index) {
				uint32_t packed_vertex_index = meshlet_triangles[index / 3 + meshlet.triangle_offset];
				uint8_t  vertex_index        = (packed_vertex_index >> ((index % 3) * 8)) & 0xFF;
				return meshlet_vertices[meshlet.vertex_offset + vertex_index];
			};

			std::int64_t triangle[3];

			triangle[0] = merge_vertex_remap[get_vertex_index(j + 0)];
			triangle[1] = merge_vertex_remap[get_vertex_index(j + 1)];
			triangle[2] = merge_vertex_remap[get_vertex_index(j + 2)];

			// remove triangles which have collapsed on themselves due to vertex merge
			if (triangle[0] == triangle[1] || triangle[0] == triangle[2] || triangle[1] == triangle[2])
			{
				continue;
			}

			glm::vec3 v0 = vertex_positions[triangle[0]];
			glm::vec3 v1 = vertex_positions[triangle[1]];
			glm::vec3 v2 = vertex_positions[triangle[2]];

			glm::vec3 normal = glm::cross(v1 - v0, v2 - v0);
			float     angle  = acos(dot(normalize(v1 - v0), normalize(v2 - v0)));

			uint32_t local_triangle[3];
			for (std::size_t vertex = 0; vertex < 3; vertex++)
			{
				const std::size_t vertex_index = triangle[vertex];

				glm::vec3 pos = vertex_positions[vertex_index];
				cluster_min   = glm::min(cluster_min, pos);
				cluster_max   = glm::max(cluster_max, pos);

				// 总体的index到group内index的映射
				auto [iter0, was_new0] = mesh_to_group_vertex_remap.try_emplace(vertex_index);
				if (was_new0)
				{
					iter0->second = group_vertices.size();
					group_vertices.push_back(get_packed_vertex(vertex_positions, vertex_normals, vertex_uvs, vertex_index));
				}

				uint32_t group_vertex_index = iter0->second;
				/*group_vertices[group_vertex_index].normal += glm::vec4(normal, 0.0f) * angle;*/

				if (used[group_vertex_index] == -1)
				{
					used[group_vertex_index] = group_vertex_indices.size() - previous_vertex_indices_size;
					group_vertex_indices.push_back(group_vertex_index);
				}

				local_triangle[vertex] = used[group_vertex_index];
			}

			group_triangles.push_back(
			    (local_triangle[0] << 0) |
			    (local_triangle[1] << 8) |
			    (local_triangle[2] << 16)
			);
		}

		parallel_time += parallel_timer.stop();

		Timer cluster_timer;
		cluster_timer.start();

		Cluster cluster;
		cluster.cluster_error       = meshlet.cluster_error;
		cluster.lod_bounding_sphere = meshlet.bounding_sphere;
		cluster.bounding_sphere     = glm::vec4((cluster_min + cluster_max) / 2.0f, glm::distance(cluster_min, cluster_max) / 2.0f);
		cluster.cone_axis           = meshlet.cone_axis;
		cluster.cone_cutoff         = meshlet.cone_cutoff;
		cluster.cluster_group_index = cluster_groups.size();
		cluster.cluster_index       = cluster_index++;
		clusters.push_back(cluster);

		Meshlet temp_meshlet;
		temp_meshlet.vertex_offset = previous_vertex_indices_size;
		temp_meshlet.vertex_count  = group_vertex_indices.size() - previous_vertex_indices_size;
		temp_meshlet.triangle_offset = previous_triangles_size;
		temp_meshlet.triangle_count  = group_triangles.size() - previous_triangles_size;
		group_meshlets.push_back(temp_meshlet);

		cluster_time += cluster_timer.stop();
	}

	auto meshlet_time = meshlet_timer.stop();
	/*LOGI("meshlet time: {}, cluster time: {}, parallel time: {}", meshlet_time, cluster_time, parallel_time);*/

	Timer data_timer;
	data_timer.start();

	ClusterGroup cluster_group;
	cluster_group.offset          = scene_data.size();

	cluster_group.vertices_offset = 0;
	for (auto &v : group_vertices)
	{
		glm::vec3 normal    = v.normal.xyz;
		v.normal.xyz        = glm::normalize(normal);
		const uint32_t *raw = reinterpret_cast<const uint32_t *>(&v);
		scene_data.insert(scene_data.end(), raw, raw + sizeof(PackedVertex) / sizeof(uint32_t));
	}

	cluster_group.vertex_indices_offset = group_vertices.size() * sizeof(PackedVertex) / sizeof(uint32_t);
	scene_data.insert(scene_data.end(), group_vertex_indices.begin(), group_vertex_indices.end());

	cluster_group.triangles_offset = cluster_group.vertex_indices_offset + group_vertex_indices.size();
	scene_data.insert(scene_data.end(), group_triangles.begin(), group_triangles.end());

	cluster_group.meshlets_offset = cluster_group.triangles_offset + group_triangles.size();
	for (const auto &meshlet : group_meshlets)
	{
		scene_data.push_back(meshlet.vertex_offset);
		scene_data.push_back(meshlet.vertex_count);
		scene_data.push_back(meshlet.triangle_offset);
		scene_data.push_back(meshlet.triangle_count);
	}

	cluster_group.size = cluster_group.meshlets_offset + group_meshlets.size() * 4;
	assert(cluster_group.size == scene_data.size() - cluster_group.offset);

	cluster_groups.push_back(cluster_group);

	auto data_time = data_timer.stop();
	//LOGI("data time: {}", data_time);

	auto append_time = append_timer.stop();
	/*LOGI("append time: {}, data time: {}, meshlet time: {}, total: {}", append_time, data_time, meshlet_time, data_time + meshlet_time);*/
}

bool simplify_group(std::vector<glm::vec3> &vertex_positions, std::vector<glm::vec3> &vertex_normals, std::vector<glm::vec2> &vertex_uvs, std::vector<ClusterGroup> &cluster_groups, std::vector<uint32_t> &meshlet_vertices, std::vector<uint32_t> &meshlet_triangles, std::vector<Meshlet> &meshlets, std::span<Meshlet> &previous_level_meshlets, const MeshletGroup &group, const std::vector<std::int64_t> &merge_vertex_remap, float target_error)
{
	Timer meshlet_timer;
	meshlet_timer.start();

	std::vector<uint32_t>                        group_vertex_indices;
	std::vector<glm::vec3>                       group_vertex_buffer;
	std::vector<PackedVertex>                    group_vertex_buffer_wrapped;
	std::vector<std::size_t>                     group_to_mesh_vertex_remap;
	std::unordered_map<std::size_t, std::size_t> mesh_to_group_vertex_remap;

	// add cluster vertices to this group
	for (const auto &meshlet_index : group.meshlets)
	{
		const auto &meshlet = previous_level_meshlets[meshlet_index];
		std::size_t start   = group_vertex_indices.size();
		group_vertex_indices.reserve(start + meshlet.triangle_count * 3);

		for (std::size_t j = 0; j < meshlet.triangle_count * 3; j += 3)
		{        // triangle per triangle
			auto get_vertex_index = [&](std::size_t index) {
				uint32_t packed_vertex_index = meshlet_triangles[index / 3 + meshlet.triangle_offset];
				uint8_t  vertex_index        = (packed_vertex_index >> ((index % 3) * 8)) & 0xFF;
				return meshlet_vertices[meshlet.vertex_offset + vertex_index];
			};

			std::int64_t triangle[3] = {
			    merge_vertex_remap[get_vertex_index(j + 0)],
			    merge_vertex_remap[get_vertex_index(j + 1)],
			    merge_vertex_remap[get_vertex_index(j + 2)],
			};

			// remove triangles which have collapsed on themselves due to vertex merge
			if (triangle[0] == triangle[1] && triangle[0] == triangle[2])
			{
				continue;
			}

			for (std::size_t vertex = 0; vertex < 3; vertex++)
			{
				const std::size_t vertex_index = triangle[vertex];

				// 总体的index到group内index的映射
				auto [iter, was_new] = mesh_to_group_vertex_remap.try_emplace(vertex_index);
				if (was_new)
				{
					iter->second = group_vertex_buffer.size();
					group_vertex_buffer.push_back(VertexWrapper(&vertex_positions[0].x, vertex_index).getPosition());
					group_vertex_buffer_wrapped.push_back(get_packed_vertex(vertex_positions, vertex_normals, vertex_uvs, vertex_index));
				}
				group_vertex_indices.push_back(iter->second);
			}
		}
	}

	// group vertex buffer和group index buffer已经准备好了

	if (group_vertex_indices.empty())
		return false;

	// create reverse mapping from group to mesh-wide vertex indices
	group_to_mesh_vertex_remap.resize(group_vertex_buffer.size());

	for (const auto &[mesh_index, group_index] : mesh_to_group_vertex_remap)
	{
		assert(group_index < group_to_mesh_vertex_remap.size());
		group_to_mesh_vertex_remap[group_index] = mesh_index;
	}

#ifdef USE_VCG
	MyMesh mesh;
	vcg::tri::Allocator<MyMesh>::AddVertices(mesh, group_vertex_buffer.size());
	for (size_t i = 0; i < group_vertex_buffer.size(); ++i)
	{
		mesh.vert[i].P() = MyMesh::CoordType(group_vertex_buffer[i].x, group_vertex_buffer[i].y, group_vertex_buffer[i].z);
		mesh.vert[i].T().U() = group_vertex_buffer_wrapped[i].pos.w;
		mesh.vert[i].T().V() = group_vertex_buffer_wrapped[i].normal.w;
	}

	vcg::tri::Allocator<MyMesh>::AddFaces(mesh, group_vertex_indices.size() / 3);
	for (size_t i = 0; i < group_vertex_indices.size(); i += 3)
	{
		mesh.face[i / 3].V(0) = &mesh.vert[group_vertex_indices[i]];
		mesh.face[i / 3].V(1) = &mesh.vert[group_vertex_indices[i + 1]];
		mesh.face[i / 3].V(2) = &mesh.vert[group_vertex_indices[i + 2]];
	}

	std::unordered_map<MeshletEdge, uint32_t, MeshletEdgeHasher> edges_to_num;

	for (size_t i = 0; i < group_vertex_indices.size(); i += 3)
	{
		MeshletEdge edge1{group_vertex_indices[i], group_vertex_indices[i + 1]};
		MeshletEdge edge2{group_vertex_indices[i + 1], group_vertex_indices[i + 2]};
		MeshletEdge edge3{group_vertex_indices[i + 2], group_vertex_indices[i]};

		edges_to_num[edge1]++;
		edges_to_num[edge2]++;
		edges_to_num[edge3]++;
	}

	std::vector<bool> boundary_vertices(group_vertex_buffer.size());
	for (const auto& [edge, count] : edges_to_num)
	{
		if (count == 1)
		{
			boundary_vertices[edge.first] = true;
			boundary_vertices[edge.second] = true;
		}
	}
	for (size_t i = 0; i < boundary_vertices.size(); i++)
	{
		if (boundary_vertices[i] == false)
		{
			/*LOGI("not boundary vertex: {}", i);*/
		}
	}

	/*for (const auto &[edge, count] : edges_to_num)
	{
		if (count != 1)
			LOGI("not Boundary edge: ({}, {})", edge.first, edge.second);
	}*/

	// ====== 更新邻接关系和法线 ======
	/*vcg::tri::UpdateTopology<MyMesh>::FaceFace(mesh);*/
	/*vcg::tri::UpdateNormal<MyMesh>::PerVertexNormalized(mesh);*/
	vcg::tri::Clean<MyMesh>::RemoveDegenerateFace(mesh);
	vcg::tri::UpdateNormal<MyMesh>::PerVertexNormalized(mesh);
	vcg::tri::UpdateTopology<MyMesh>::VertexFace(mesh);
	vcg::tri::UpdateBounding<MyMesh>::Box(mesh);
	// ====== 初始化简化参数 ======
	vcg::tri::TriEdgeCollapseQuadricParameter qparams;
	qparams.QualityThr       = 0.3;
	qparams.OptimalPlacement = true;
	qparams.PreserveTopology = false;
	qparams.BoundaryQuadricWeight = 0.5f;
	qparams.PreserveBoundary = false;
	qparams.QualityCheck     = true;
	qparams.NormalCheck      = true;

	// ====== 初始化优化器 ======
	vcg::LocalOptimization<MyMesh> decimator(mesh, &qparams);
	decimator.Init<MyTriEdgeCollapse>();

	const float threshold         = THRESHOLD;
	uint32_t    target_face_count = group_vertex_indices.size() / 3 * threshold;
	decimator.SetTargetSimplices(target_face_count);
	decimator.SetTargetMetric(target_error);
	decimator.SetTimeBudget(1.0f);
	decimator.SetTargetOperations(100000);

	/*LOGI("Heap size after Init: {}", decimator.h.size());*/

	// ====== 执行简化过程 ======
	while (decimator.DoOptimization() &&
	       mesh.fn > target_face_count &&
	       decimator.currMetric < target_error)
	{

	}

	vcg::tri::Allocator<MyMesh>::CompactFaceVector(mesh);
	vcg::tri::Allocator<MyMesh>::CompactVertexVector(mesh);
	vcg::tri::UpdateNormal<MyMesh>::PerVertexNormalized(mesh);

	double total_error = decimator.currMetric;

	LOGI("before simplify: {}, after simplify: {}, total error: {}", group_vertex_indices.size() / 3, mesh.fn, total_error);

	/*append_meshlets(vertex_positions, vertex_normals, vertex_uvs, meshlet_vertices, meshlet_triangles, meshlets, mesh);*/

	uint32_t vertex_offset = vertex_positions.size();
	uint32_t vertex_count  = mesh.vert.size();

	if (mesh.fn < group_vertex_indices.size() / 3 && total_error < target_error)
	{
		glm::vec3 mesh_min{+INFINITY, +INFINITY, +INFINITY};
		glm::vec3 mesh_max{-INFINITY, -INFINITY, -INFINITY};

		for (auto &v : mesh.vert)
		{
			if (!v.IsD())
			{
				vertex_positions.push_back(glm::vec3(v.P().X(), v.P().Y(), v.P().Z()));
				vertex_normals.push_back(glm::vec3(v.N().X(), v.N().Y(), v.N().Z()));
				/*if (vertex_uvs)*/
				{
					vertex_uvs.push_back(glm::vec2(v.T().U(), v.T().V()));
				}
				mesh_min = glm::min(mesh_min, glm::vec3(v.P().X(), v.P().Y(), v.P().Z()));
				mesh_max = glm::max(mesh_max, glm::vec3(v.P().X(), v.P().Y(), v.P().Z()));
			}
		}

		group_to_mesh_vertex_remap.resize(vertex_count);
		for (size_t i = 0; i < vertex_count; ++i)
		{
			group_to_mesh_vertex_remap[i] = vertex_offset + i;
		}

		std::vector<uint32_t> simplified_index_buffer;

		for (auto &f : mesh.face)
		{
			if (!f.IsD())
			{
				for (int i = 0; i < 3; ++i)
					simplified_index_buffer.push_back(vcg::tri::Index(mesh, f.V(i)));
			}
		}

		float local_scale      = meshopt_simplifyScale(&group_vertex_buffer[0].x, group_vertex_buffer.size(), sizeof(glm::vec3));
		float mesh_space_error = total_error * local_scale;
		float max_child_error  = 0.0f;

		for (const auto &meshlet_index : group.meshlets)
		{
			const auto &previous_meshlet = previous_level_meshlets[meshlet_index];
			max_child_error              = std::max(max_child_error, previous_meshlet.cluster_error);
		}

		mesh_space_error += max_child_error;

		glm::vec4 simplified_cluster_bounds = glm::vec4((mesh_min + mesh_max) / 2.0f, glm::distance(mesh_min, mesh_max) / 2.0f);

		assert(!group.meshlets.empty());
		uint32_t cluster_group_index                               = previous_level_meshlets[group.meshlets[0]].cluster_group_index;
		cluster_groups[cluster_group_index].parent_error           = mesh_space_error;
		cluster_groups[cluster_group_index].parent_bounding_sphere = simplified_cluster_bounds;

		append_meshlets(meshlet_vertices, meshlet_triangles, meshlets, &vertex_positions[vertex_offset].x, vertex_count, simplified_index_buffer, mesh_space_error, simplified_cluster_bounds, group_to_mesh_vertex_remap);

		return true;
	}
	return false;
#endif        // USE_VCG

#ifdef USE_MESHOPT

	// simplify this group
	const float  threshold          = THRESHOLD;
	std::size_t  target_index_count = group_vertex_indices.size() * threshold;
	unsigned int options            = meshopt_SimplifyLockBorder;        // we want all group borders to be locked (because they are shared between groups)

	std::vector<uint32_t> simplified_index_buffer;
	simplified_index_buffer.resize(group_vertex_indices.size());
	float simplification_error = 0.f;

	std::size_t simplified_index_count = meshopt_simplify(
	    simplified_index_buffer.data(),        // output
	    group_vertex_indices.data(),
	    group_vertex_indices.size(),        // index buffer
	    &group_vertex_buffer[0].x,          // pointer to position data
	    group_vertex_buffer.size(),
	    sizeof(glm::vec3),
	    target_index_count,
	    target_error,
	    options,
	    &simplification_error);
	simplified_index_buffer.resize(simplified_index_count);

	// ===== Generate meshlets for this group
	if (simplified_index_count > 0 && simplified_index_count != group_vertex_indices.size())
	{
		float local_scale      = meshopt_simplifyScale(&group_vertex_buffer[0].x, group_vertex_buffer.size(), sizeof(glm::vec3));
		float mesh_space_error = simplification_error * local_scale;
		float max_child_error  = 0.0f;

		glm::vec3 min{+INFINITY, +INFINITY, +INFINITY};
		glm::vec3 max{-INFINITY, -INFINITY, -INFINITY};

		// meshletgroup's bounding box
		for (auto &index : simplified_index_buffer)
		{
			const glm::vec3 vertex_pos = group_vertex_buffer[index];
			min                        = glm::min(min, vertex_pos);
			max                        = glm::max(max, vertex_pos);
		}

		glm::vec4 simplified_cluster_bounds = glm::vec4((min + max) / 2.0f, glm::distance(min, max) / 2.0f);

		for (const auto &meshlet_index : group.meshlets)
		{
			const auto &previous_meshlet = previous_level_meshlets[meshlet_index];
			max_child_error              = std::max(max_child_error, previous_meshlet.cluster_error);
		}

		mesh_space_error += max_child_error;

		assert(!group.meshlets.empty());
		uint32_t cluster_group_index = previous_level_meshlets[group.meshlets[0]].cluster_group_index;
		cluster_groups[cluster_group_index].parent_error = mesh_space_error;
		cluster_groups[cluster_group_index].parent_bounding_sphere = simplified_cluster_bounds;

		append_meshlets(meshlet_vertices, meshlet_triangles, meshlets, &group_vertex_buffer[0].x, group_vertex_buffer.size(), simplified_index_buffer, mesh_space_error, simplified_cluster_bounds, group_to_mesh_vertex_remap); // group_to_mesh_vertex_remap再映射会总体

		auto meshlet_time = meshlet_timer.stop();
		/*LOGI("meshlet time: {}", meshlet_time);*/

		return true;
	}
	return false;

#endif        // USE_MESHOPT
}

void generate_cluster_hierarchy(const MeshPrimitiveData &primitive, std::vector<uint32_t> &scene_data, std::vector<ClusterGroup> &cluster_groups, std::vector<Cluster> &clusters)
{
	LOGI("Building lod...");

	Timer timer;
	timer.start();

	static int num = 0;
	num++;
	LOGI("SubMesh {}", num);

	std::vector<uint32_t> index_data_32;
	if (primitive.index_type == vk::IndexType::eUint16)
	{
		const uint16_t *index_data_16 = reinterpret_cast<const uint16_t *>(primitive.indices.data());
		index_data_32.resize(primitive.index_count);
		for (size_t i = 0; i < primitive.index_count; ++i)
		{
			index_data_32[i] = static_cast<uint32_t>(index_data_16[i]);
		}
	}
	else if (primitive.index_type == vk::IndexType::eUint32)
	{
		index_data_32.assign(
		    reinterpret_cast<const uint32_t *>(primitive.indices.data()),
		    reinterpret_cast<const uint32_t *>(primitive.indices.data()) + primitive.index_count);
	}

	// auto vertex_positions = reinterpret_cast<const float *>(primitive.attributes.at("position").data.data());
	size_t      vertex_count = primitive.vertex_count;
	std::vector<glm::vec3> vertex_positions(vertex_count);
	std::vector<glm::vec3> vertex_normals(vertex_count);
	std::vector<glm::vec2> vertex_uvs(vertex_count);
	const float           *raw_positions = reinterpret_cast<const float *>(primitive.attributes.at("position").data.data());
	const float           *raw_normals   = reinterpret_cast<const float *>(primitive.attributes.at("normal").data.data());
	const float           *raw_uvs       = nullptr;
	if (primitive.attributes.find("texcoord_0") != primitive.attributes.end())
	{
		raw_uvs = reinterpret_cast<const float *>(primitive.attributes.at("texcoord_0").data.data());
	}

	for (size_t i = 0; i < vertex_count; ++i)
	{
		vertex_positions[i] = glm::vec3(
		    raw_positions[i * 3 + 0],
		    raw_positions[i * 3 + 1],
		    raw_positions[i * 3 + 2]);
		vertex_normals[i] = glm::vec3(
		    raw_normals[i * 3 + 0],
		    raw_normals[i * 3 + 1],
		    raw_normals[i * 3 + 2]);
		if (raw_uvs)
		{
			vertex_uvs[i] = glm::vec2(raw_uvs[i * 2 + 0], raw_uvs[i * 2 + 1]);
		}
		else
		{
			vertex_uvs[i] = glm::vec2(0.0f, 0.0f);
		}
	}

	glm::vec3 min{+INFINITY, +INFINITY, +INFINITY};
	glm::vec3 max{-INFINITY, -INFINITY, -INFINITY};

	auto         &index_buffer       = index_data_32;
	std::uint32_t unique_group_index = 0;

	for (auto &index : index_buffer)
	{
		const glm::vec3 vertex_pos = vertex_positions[index].xyz;
		min                        = glm::min(min, vertex_pos);
		max                        = glm::max(max, vertex_pos);
	}

	glm::vec4 simplified_cluster_bounds = glm::vec4((min + max) / 2.0f, glm::distance(min, max) / 2.0f);

	std::vector<uint32_t> meshlet_vertices;
	std::vector<uint32_t> meshlet_triangles;
	std::vector<Meshlet>  meshlets;

	append_meshlets(meshlet_vertices, meshlet_triangles, meshlets, &vertex_positions[0].x, primitive.vertex_count, index_buffer, 0.0f, simplified_cluster_bounds);

	LOGI("LOD {}: {} meshlets, {} vertices, {} triangles", 0, meshlets.size(), meshlet_vertices.size(), meshlet_triangles.size());

	KDTree<VertexWrapper> kdtree;

	const int max_lod = 20;

	std::vector<uint8_t>       group_vertex_indices;
	std::vector<VertexWrapper> group_vertices_pre_weld;

	size_t previous_meshlets_start       = 0;

	const float simplify_scale = SIMPLIFY_SCALE;
	// const float simplify_scale = meshopt_simplifyScale(vertex_positions, primitive.vertex_count, sizeof(glm::vec3));

	for (int lod = 0; lod < max_lod; ++lod)
	{
		Timer lod_timer;
		lod_timer.start();

		float t_lod = lod / (float) max_lod;

		std::span<Meshlet> previous_level_meshlets = std::span{meshlets.data() + previous_meshlets_start, meshlets.size() - previous_meshlets_start};
		if (previous_level_meshlets.size() <= 1 && lod != 0)
		{
			break;
		}

		std::unordered_set<size_t> meshlet_vertex_indices;
		for (const auto &meshlet : previous_level_meshlets)
		{
			auto get_vertex_index = [&](size_t index) {
				uint32_t packed_vertex_index = meshlet_triangles[index / 3 + meshlet.triangle_offset];
				uint8_t  vertex_index        = (packed_vertex_index >> ((index % 3) * 8)) & 0xFF;
				return meshlet_vertices[meshlet.vertex_offset + vertex_index];
			};

			for (size_t i = 0; i < meshlet.triangle_count * 3; i++)
			{
				meshlet_vertex_indices.insert(get_vertex_index(i));
			}
		}
		group_vertices_pre_weld.clear();
		group_vertices_pre_weld.reserve(meshlet_vertex_indices.size());
		for (const size_t i : meshlet_vertex_indices)
		{
			group_vertices_pre_weld.push_back(VertexWrapper(&vertex_positions[0].x, i));
		}

		std::span<const VertexWrapper> wrapped_vertices = group_vertices_pre_weld;

		std::vector<bool> boundary;

#ifdef USE_WELDING
		kdtree.build(wrapped_vertices);
		boundary = find_boundary_vertices(meshlet_vertices, meshlet_triangles, previous_level_meshlets, vertex_positions.size());
#endif

		const float max_distance   = (t_lod * 0.1f + (1 - t_lod) * 0.01f) * simplify_scale;
		float max_normal_distance = 0.49f;
		//if (meshlet_vertex_indices.size() > 40000)
		//{
		//	max_normal_distance = 1.0f;
		//}
		/*const float max_normal_distance = 1.0f;*/
		const float max_uv_distance = t_lod * 0.5f + (1 - t_lod) * 1.0f / 256.0f;

		Timer merge_timer;
		merge_timer.start();

		std::vector<std::int64_t> merge_vertex_remap;
		
		merge_vertex_remap = merge_by_distance(vertex_positions,  vertex_normals, vertex_uvs, boundary, group_vertices_pre_weld, max_distance, max_normal_distance, max_uv_distance, kdtree);

		auto merge_time = merge_timer.stop();
		//LOGI("Merge time is {} s", merge_time);

		const std::vector<MeshletGroup> groups = group_meshlets_remap(meshlet_vertices, meshlet_triangles, previous_level_meshlets, merge_vertex_remap);

		const std::size_t new_cluster_group_start = cluster_groups.size();
		const std::size_t new_meshlet_start       = meshlets.size();

		float target_error = 0.05f + 0.85f * t_lod;

		Timer simplify_timer;
		simplify_timer.start();

		double meshlet_group_time = 0;
		double simplify_group_time = 0;

		for (const auto &group : groups)
		{
			// meshlets vector is modified during the loop
			previous_level_meshlets = std::span{meshlets.data() + previous_meshlets_start, meshlets.size() - previous_meshlets_start};

			Timer meshlet_group_timer;
			meshlet_group_timer.start();

			append_meshlet_groups(vertex_positions, vertex_normals, vertex_uvs, scene_data, cluster_groups, clusters, meshlet_vertices, meshlet_triangles, previous_level_meshlets, group, merge_vertex_remap, lod);

			meshlet_group_time = meshlet_group_timer.stop();

			/*LOGI("meshlet group time: {}", meshlet_group_time);*/

			Timer simplify_group_timer;
			simplify_group_timer.start();

			bool is_simplified = simplify_group(vertex_positions, vertex_normals, vertex_uvs, cluster_groups, meshlet_vertices, meshlet_triangles, meshlets, previous_level_meshlets, group, merge_vertex_remap, target_error);

			simplify_group_time = simplify_group_timer.stop();

			/*LOGI("simplify group time: {}", simplify_group_time);*/
		}

		auto simplify_time = simplify_timer.stop();
		/*LOGI("meshlet group time: {} seconds, simplify group time: {} seconds, simplify time: {} seconds", meshlet_group_time, simplify_group_time, simplify_time);*/

		for (size_t i = new_cluster_group_start; i < cluster_groups.size(); i++)
		{
			cluster_groups[i].lod = lod;
		}

		auto lod_time = lod_timer.stop();
		//LOGI("lod time: {} seconds", lod_time);

		if (new_meshlet_start != meshlets.size())
		{
			LOGI("LOD {}: {} meshlets", lod + 1, meshlets.size() - new_meshlet_start);
			previous_meshlets_start = new_meshlet_start;
		}
		else
		{
			break;
		}
	}

	auto elapsed_time = timer.stop();
	LOGI("Time spent building lod: {} seconds.", xihe::to_string(elapsed_time));
}
}        // namespace xihe::sg