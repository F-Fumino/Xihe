#include "mesh_lod.h"
#include "kdtree.h"
#include "metis.h"
#include <common/timer.h>
#include <glm/gtx/norm.hpp>
#include <tbb/parallel_for.h>
#include <scene_graph/geometry_data.h>
#include <meshoptimizer.h>

#define USE_WELDING 1

#define THRESHOLD 0.75f
#define SIMPLIFY_SCALE 30.0f

namespace xihe::sg
{

static std::vector<bool> find_boundary_vertices(const MeshPrimitiveData &primitive, const std::vector<uint32_t> &meshlet_vertices, const std::vector<uint32_t> &triangles, std::span<Meshlet> meshlets)
{
	std::vector<bool> boundary_vertices;
	boundary_vertices.resize(primitive.vertex_count);

	for (size_t i = 0; i < primitive.vertex_count; i++)
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

static std::vector<std::int64_t> merge_by_distance(const MeshPrimitiveData &primitive, const std::vector<bool> &boundary, std::span<const VertexWrapper> group_vertices_pre_weld, float max_distance, float max_uv_distance, const KDTree<VertexWrapper> &kdtree)
{
	std::vector<std::int64_t> vertex_remap;
	const std::size_t         vertex_count = primitive.vertex_count;
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

	auto vertex_positions = reinterpret_cast<const float *>(primitive.attributes.at("position").data.data());

	const float *vertex_uvs = nullptr;
	if (primitive.attributes.find("texcoord_0") != primitive.attributes.end())
	{
		vertex_uvs = reinterpret_cast<const float *>(primitive.attributes.at("texcoord_0").data.data());
	}

	for (std::int64_t v = 0; v < group_vertices_pre_weld.size(); v++)
	{
		std::int64_t replacement            = -1;
		const auto  &current_vertex_wrapped = group_vertices_pre_weld[v];
		if (!boundary[current_vertex_wrapped.index])
		{
			auto            &neighbors          = neighbors_for_all_vertices[v];
			const glm::vec3 &current_vertex_pos = current_vertex_wrapped.getPosition();
			glm::vec2        current_vertex_uv  = glm::vec2(0.0, 0.0);

			if (vertex_uvs)
			{
				const float *vertex_uv = vertex_uvs + current_vertex_wrapped.index * 2;
				current_vertex_uv      = glm::vec2(vertex_uv[0], vertex_uv[1]);
			}

			float max_distance_sq    = max_distance * max_distance;
			float max_uv_distance_sq = max_uv_distance * max_uv_distance;

			for (const std::size_t &neighbor : neighbors)
			{
				if (vertex_remap[group_vertices_pre_weld[neighbor].index] == -1)
				{
					continue;
				}
				auto             other_vertex_wrapped = VertexWrapper(vertex_positions, vertex_remap[group_vertices_pre_weld[neighbor].index]);
				const glm::vec3 &other_vertex_pos     = other_vertex_wrapped.getPosition();
				const float      vertex_distance_sq   = glm::distance2(current_vertex_pos, other_vertex_pos);
				if (vertex_distance_sq <= max_distance_sq)
				{
					if (!vertex_uvs)
					{
						replacement     = vertex_remap[group_vertices_pre_weld[neighbor].index];
						max_distance_sq = vertex_distance_sq;
					}
					else
					{
						const float     *vertex_uv             = vertex_uvs + group_vertices_pre_weld[neighbor].index * 2;
						const glm::vec2 &other_vertex_uv       = glm::vec2(vertex_uv[0], vertex_uv[1]);
						const float      vertex_uv_distance_sq = glm::distance2(current_vertex_uv, other_vertex_uv);
						if (vertex_uv_distance_sq <= max_uv_distance_sq)
						{
							replacement        = vertex_remap[group_vertices_pre_weld[neighbor].index];
							max_distance_sq    = vertex_distance_sq;
							max_uv_distance_sq = vertex_uv_distance_sq;
						}
					}
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

static std::vector<MeshletGroup> group_meshlets_remap(const MeshPrimitiveData &primitive, const std::vector<uint32_t> &meshlet_vertices, const std::vector<uint32_t> &triangles, std::span<Meshlet> meshlets, std::span<const std::int64_t> vertex_remap)
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
				edges_to_meshlets[edge].push_back(meshlet_index);
				meshlets_to_edges[meshlet_index].emplace_back(edge);
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

PackedVertex get_packed_vertex(const float *vertex_positions, const float *vertex_normals, const float *vertex_texcoords, uint32_t index)
{
	PackedVertex vertex;

	vertex.pos    = glm::vec4(vertex_positions[index * 3 + 0], vertex_positions[index * 3 + 1], vertex_positions[index * 3 + 2], 0.0f);
	vertex.normal = glm::vec4(vertex_normals[index * 3 + 0], vertex_normals[index * 3 + 1], vertex_normals[index * 3 + 2], 0.0f);

	if (vertex_texcoords)
	{
		vertex.pos.w    = vertex_texcoords[index * 2 + 0];
		vertex.normal.w = vertex_texcoords[index * 2 + 1];
	}

	return vertex;
}

static void append_meshlets(const MeshPrimitiveData &primitive_data, std::vector<PackedVertex> &vertices, std::vector<uint32_t> &meshlet_vertices, std::vector<uint32_t> &triangles, std::vector<Meshlet> &meshlets, const float *vertex_positions, uint32_t vertex_positions_count, std::span<std::uint32_t> index_buffer, const glm::vec4 &cluster_bounds, float cluster_error, std::span<size_t> vertex_remap = std::span<size_t>())
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

	std::size_t vertex_offset   = vertices.size();
	std::size_t triangle_offset = triangles.size();
	std::size_t meshlet_offset  = meshlets.size();

	const meshopt_Meshlet &last           = local_meshlets[meshlet_count - 1];
	const std::size_t      vertex_count   = last.vertex_offset + last.vertex_count;
	std::size_t            triangle_count = last.triangle_offset / 3 + last.triangle_count;

	vertices.resize(vertex_offset + vertex_count);
	meshlet_vertices.resize(vertex_offset + vertex_count);
	triangles.resize(triangle_offset + triangle_count);
	meshlets.resize(meshlet_offset + meshlet_count);

	const float *mesh_vertex_positions = reinterpret_cast<const float *>(primitive_data.attributes.at("position").data.data());
	const float *mesh_vertex_normals   = reinterpret_cast<const float *>(primitive_data.attributes.at("normal").data.data());
	const float *mesh_vertex_texcoords = nullptr;

	if (primitive_data.attributes.find("texcoord_0") != primitive_data.attributes.end())
	{
		mesh_vertex_texcoords = reinterpret_cast<const float *>(primitive_data.attributes.at("texcoord_0").data.data());
	}

	if (vertex_remap.empty())
	{
		tbb::parallel_for(std::size_t(0), vertex_count, [&](std::size_t index) {
			meshlet_vertices[vertex_offset + index] = meshlet_vertex_indices[index];
			vertices[vertex_offset + index]         = get_packed_vertex(mesh_vertex_positions, mesh_vertex_normals, mesh_vertex_texcoords, meshlet_vertex_indices[index]);
		});
	}
	else
	{
		tbb::parallel_for(std::size_t(0), vertex_count, [&](std::size_t index) {
			meshlet_vertices[vertex_offset + index] = vertex_remap[meshlet_vertex_indices[index]];
			vertices[vertex_offset + index]         = get_packed_vertex(mesh_vertex_positions, mesh_vertex_normals, mesh_vertex_texcoords, vertex_remap[meshlet_vertex_indices[index]]);
		});
	}

	tbb::parallel_for(std::size_t(0), triangle_count, [&](std::size_t index) {
		uint8_t idx0 = meshlet_triangle_indices[index * 3 + 0];
		uint8_t idx1 = meshlet_triangle_indices[index * 3 + 1];
		uint8_t idx2 = meshlet_triangle_indices[index * 3 + 2];

		uint32_t packed_triangle = idx0 | (idx1 << 8) | (idx2 << 16);

		triangles[triangle_offset + index] = packed_triangle;
	});

	tbb::parallel_for(std::size_t(0), meshlet_count, [&](std::size_t index) {
		auto &local_meshlet = local_meshlets[index];
		auto &meshlet       = meshlets[meshlet_offset + index];

		meshlet.vertex_offset = vertex_offset + local_meshlet.vertex_offset;
		meshlet.vertex_count  = local_meshlet.vertex_count;

		meshlet.triangle_offset = triangle_offset + local_meshlet.triangle_offset / 3;
		meshlet.triangle_count  = local_meshlet.triangle_count;

		meshopt_Bounds meshlet_bounds = meshopt_computeMeshletBounds(
		    meshlet_vertex_indices.data() + local_meshlet.vertex_offset,
		    meshlet_triangle_indices.data() + local_meshlet.triangle_offset,
		    local_meshlet.triangle_count, vertex_positions, vertex_positions_count, sizeof(float) * 3);

		meshlet.cone_axis   = glm::vec3(meshlet_bounds.cone_axis[0], meshlet_bounds.cone_axis[1], meshlet_bounds.cone_axis[2]);
		meshlet.cone_cutoff = meshlet_bounds.cone_cutoff;

		meshlet.cone_apex = glm::vec3(meshlet_bounds.cone_apex[0], meshlet_bounds.cone_apex[1], meshlet_bounds.cone_apex[2]);

		meshlet.cluster_error = cluster_error;
		meshlet.center       = glm::vec3(cluster_bounds.x, cluster_bounds.y, cluster_bounds.z);
		meshlet.radius       = cluster_bounds.w;
	});
}

bool simplify_group(const MeshPrimitiveData &primitive, std::vector<PackedVertex> &vertices, std::vector<uint32_t> &meshlet_vertices, std::vector<uint32_t> &triangles, std::vector<Meshlet> &meshlets, std::span<Meshlet> &previous_level_meshlets, const MeshletGroup &group, const std::vector<std::int64_t> &merge_vertex_remap, float target_error)
{
	std::vector<uint32_t>                        group_vertex_indices;
	std::vector<glm::vec3>                       group_vertex_buffer;
	std::vector<std::size_t>                     group_to_mesh_vertex_remap;
	std::unordered_map<std::size_t, std::size_t> mesh_to_group_vertex_remap;

	auto vertex_positions = reinterpret_cast<const float *>(primitive.attributes.at("position").data.data());

	// add cluster vertices to this group
	for (const auto &meshlet_index : group.meshlets)
	{
		const auto &meshlet = previous_level_meshlets[meshlet_index];
		std::size_t start   = group_vertex_indices.size();
		group_vertex_indices.reserve(start + meshlet.triangle_count * 3);

		for (std::size_t j = 0; j < meshlet.triangle_count * 3; j += 3)
		{        // triangle per triangle

			auto get_vertex_index = [&](std::size_t index) {
				uint32_t packed_vertex_index = triangles[index / 3 + meshlet.triangle_offset];
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
					group_vertex_buffer.push_back(VertexWrapper(vertex_positions, vertex_index).getPosition());
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

		// 把小buffer中的index映射回总体的index
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
		for (const auto &meshlet_index : group.meshlets)
		{
			previous_level_meshlets[meshlet_index].parent_error           = mesh_space_error;
			previous_level_meshlets[meshlet_index].parent_bounding_sphere = simplified_cluster_bounds;
		}

		append_meshlets(primitive, vertices, meshlet_vertices, triangles, meshlets, &group_vertex_buffer[0].x, group_vertex_buffer.size(), simplified_index_buffer, simplified_cluster_bounds, mesh_space_error, group_to_mesh_vertex_remap);

		return true;
	}
	return false;
}

void generate_cluster_hierarchy(const MeshPrimitiveData &primitive, std::vector<PackedVertex> &vertices, std::vector<uint32_t> &triangles, std::vector<Meshlet> &meshlets)
{
	LOGI("Building lod...");

	std::vector<uint32_t> meshlet_vertices;

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

	auto vertex_positions = reinterpret_cast<const float *>(primitive.attributes.at("position").data.data());

	glm::vec3 min{+INFINITY, +INFINITY, +INFINITY};
	glm::vec3 max{-INFINITY, -INFINITY, -INFINITY};

	auto         &index_buffer       = index_data_32;
	std::uint32_t unique_group_index = 0;

	for (auto &index : index_buffer)
	{
		const glm::vec3 vertex_pos = glm::vec3{vertex_positions[3 * index], vertex_positions[3 * index + 1], vertex_positions[3 * index + 2]};
		min                        = glm::min(min, vertex_pos);
		max                        = glm::max(max, vertex_pos);
	}

	glm::vec4 simplified_cluster_bounds = glm::vec4((min + max) / 2.0f, glm::distance(min, max) / 2.0f);

	append_meshlets(primitive, vertices, meshlet_vertices, triangles, meshlets, vertex_positions, primitive.vertex_count, index_buffer, simplified_cluster_bounds, 0.0f);

	LOGI("LOD {}: {} meshlets, {} vertices, {} triangles", 0, meshlets.size(), meshlet_vertices.size(), triangles.size());

	// LOGI("{}, {}, {}, {}", meshlets[0].parent_bounding_sphere.x, meshlets[0].parent_bounding_sphere.y, meshlets[0].parent_bounding_sphere.z, meshlets[0].parent_bounding_sphere.w);

	KDTree<VertexWrapper> kdtree;

	const int max_lod = 5;

	std::vector<uint8_t>       group_vertex_indices;
	std::vector<VertexWrapper> group_vertices_pre_weld;

	size_t previous_meshlets_start       = 0;
	size_t previous_vertex_indices_start = 0;
	size_t previous_triangles_start      = 0;

	for (int lod = 0; lod < max_lod; ++lod)
	{
		Timer lod_timer;
		lod_timer.start();

		float t_lod = lod / (float) max_lod;

		std::span<Meshlet> previous_level_meshlets = std::span{meshlets.data() + previous_meshlets_start, meshlets.size() - previous_meshlets_start};
		if (previous_level_meshlets.size() <= 1)
		{
			break;
		}

		std::unordered_set<size_t> meshlet_vertex_indices;
		for (const auto &meshlet : previous_level_meshlets)
		{
			auto get_vertex_index = [&](size_t index) {
				uint32_t packed_vertex_index = triangles[index / 3 + meshlet.triangle_offset];
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
			group_vertices_pre_weld.push_back(VertexWrapper(vertex_positions, i));
		}

		std::span<const VertexWrapper> wrapped_vertices = group_vertices_pre_weld;

		std::vector<bool> boundary;

#ifdef USE_WELDING
		kdtree.build(wrapped_vertices);
		boundary = find_boundary_vertices(primitive, meshlet_vertices, triangles, previous_level_meshlets);
#endif

		float       simplify_scale = SIMPLIFY_SCALE;
		const float max_distance   = (t_lod * 0.1f + (1 - t_lod) * 0.01f) * simplify_scale;
		const float max_uv_distance  = t_lod * 0.5f + (1 - t_lod) * 1.0f / 256.0f;

		const std::vector<std::int64_t> merge_vertex_remap = merge_by_distance(primitive, boundary, group_vertices_pre_weld, max_distance, max_uv_distance, kdtree);

		const std::vector<MeshletGroup> groups = group_meshlets_remap(primitive, meshlet_vertices, triangles, previous_level_meshlets, merge_vertex_remap);

		const std::size_t new_meshlet_start = meshlets.size();

		float target_error = 0.9f * t_lod + 0.05f * (1 - t_lod);

		for (const auto &group : groups)
		{
			// meshlets vector is modified during the loop
			previous_level_meshlets = std::span{meshlets.data() + previous_meshlets_start, meshlets.size() - previous_meshlets_start};

			bool is_simplified = simplify_group(primitive, vertices, meshlet_vertices, triangles, meshlets, previous_level_meshlets, group, merge_vertex_remap, target_error);
		}

		for (std::size_t i = new_meshlet_start; i < meshlets.size(); i++)
		{
			meshlets[i].lod = lod + 1;
		}

		auto lod_time = lod_timer.stop();

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