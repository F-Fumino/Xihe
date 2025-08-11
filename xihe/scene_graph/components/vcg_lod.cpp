#include "vcg_lod.h"

#include "common/timer.h"
#include "common/logging.h"
#include "vcg_mesh.h"

#include <meshoptimizer.h>
#include <tbb/parallel_for.h>

//#define SIMPLIFY_SCALE 30.0f
#define THRESH_POINTS_ARE_SAME 0.00002f
#define THRESHOLD 0.5f

namespace xihe::sg
{
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

	const std::size_t global_vertex_offset   = meshlet_vertices.size();
	const std::size_t global_triangle_offset = meshlet_triangles.size();
	const std::size_t global_meshlet_offset  = meshlets.size();

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

		meshlet.cone_axis       = glm::vec3(meshlet_bounds.cone_axis[0], meshlet_bounds.cone_axis[1], meshlet_bounds.cone_axis[2]);
		meshlet.cone_cutoff     = meshlet_bounds.cone_cutoff;
		meshlet.cluster_error   = cluster_error;
		meshlet.bounding_sphere = bounding_sphere;
	});
}

static std::vector<MeshletGroup> group_meshlets_remap(const std::vector<uint32_t> &meshlet_vertices, const std::vector<uint32_t> &triangles, std::span<Meshlet> meshlets)
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
                meshlet_vertices[meshlet.vertex_offset + vertex_index]);
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

static void append_meshlet_groups(const std::vector<PackedVertex> &vertices, std::vector<uint32_t> &scene_data, std::vector<ClusterGroup> &cluster_groups, std::vector<Cluster> &clusters, const std::vector<uint32_t> &meshlet_vertices, const std::vector<uint32_t> &meshlet_triangles, std::span<Meshlet> &previous_level_meshlets, const MeshletGroup &group, uint32_t lod)
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

	std::unordered_map<std::size_t, std::size_t> mesh_to_group_vertex_remap;

	uint32_t cluster_index = 0;

	for (const auto &meshlet_index : group.meshlets)
	{
		uint32_t previous_vertex_indices_size = group_vertex_indices.size();
		uint32_t previous_triangles_size      = group_triangles.size();

		auto &meshlet               = previous_level_meshlets[meshlet_index];
		meshlet.cluster_group_index = cluster_groups.size();

		glm::vec3 cluster_min{+INFINITY, +INFINITY, +INFINITY};
		glm::vec3 cluster_max{-INFINITY, -INFINITY, -INFINITY};

		std::vector<int32_t> used(group.meshlets.size() * 64, -1);

		for (std::size_t j = 0; j < meshlet.triangle_count * 3; j += 3)
		{
			auto get_vertex_index = [&](std::size_t index) {
				uint32_t packed_vertex_index = meshlet_triangles[index / 3 + meshlet.triangle_offset];
				uint8_t  vertex_index        = (packed_vertex_index >> ((index % 3) * 8)) & 0xFF;
				return meshlet_vertices[meshlet.vertex_offset + vertex_index];
			};

			std::int64_t triangle[3];

			triangle[0] = get_vertex_index(j + 0);
			triangle[1] = get_vertex_index(j + 1);
			triangle[2] = get_vertex_index(j + 2);

			// remove triangles which have collapsed on themselves due to vertex merge
			if (triangle[0] == triangle[1] || triangle[0] == triangle[2] || triangle[1] == triangle[2])
			{
				continue;
			}

			uint32_t local_triangle[3];
			for (std::size_t vertex = 0; vertex < 3; vertex++)
			{
				const std::size_t vertex_index = triangle[vertex];

				glm::vec3 pos = vertices[vertex_index].pos.xyz;
				cluster_min   = glm::min(cluster_min, pos);
				cluster_max   = glm::max(cluster_max, pos);

				// 总体的index到group内index的映射
				auto [iter0, was_new0] = mesh_to_group_vertex_remap.try_emplace(vertex_index);
				if (was_new0)
				{
					iter0->second = group_vertices.size();
					group_vertices.push_back(vertices[vertex_index]);
				}

				uint32_t group_vertex_index = iter0->second;

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
			    (local_triangle[2] << 16));
		}

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
		temp_meshlet.vertex_offset   = previous_vertex_indices_size;
		temp_meshlet.vertex_count    = group_vertex_indices.size() - previous_vertex_indices_size;
		temp_meshlet.triangle_offset = previous_triangles_size;
		temp_meshlet.triangle_count  = group_triangles.size() - previous_triangles_size;
		group_meshlets.push_back(temp_meshlet);
	}

	ClusterGroup cluster_group;
	cluster_group.offset = scene_data.size();

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

	cluster_group.lod = lod;

	cluster_groups.push_back(cluster_group);
}

bool simplify_group(std::vector<glm::vec3> &vertex_positions, std::vector<PackedVertex> &vertices, std::vector<ClusterGroup> &cluster_groups, std::vector<uint32_t> &meshlet_vertices, std::vector<uint32_t> &meshlet_triangles, std::vector<Meshlet> &meshlets, std::span<Meshlet> &previous_level_meshlets, const MeshletGroup &group, float target_error)
{
	Timer meshlet_timer;
	meshlet_timer.start();

	assert(!group.meshlets.empty());
	uint32_t cluster_group_index = previous_level_meshlets[group.meshlets[0]].cluster_group_index;

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
			    get_vertex_index(j + 0),
			    get_vertex_index(j + 1),
			    get_vertex_index(j + 2),
			};

			// remove triangles which have collapsed on themselves due to vertex merge
			if (triangle[0] == triangle[1] || triangle[0] == triangle[2] || triangle[1] == triangle[2])
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
					group_vertex_buffer.push_back(vertices[vertex_index].pos.xyz);
					group_vertex_buffer_wrapped.push_back(vertices[vertex_index]);
				}
				group_vertex_indices.push_back(iter->second);
			}
		}
	}

	if (group_vertex_indices.empty())
		return false;

	bool flagX = false;
	bool flagY = false;
	bool flagZ = false;

	MyMesh mesh;
	vcg::tri::Allocator<MyMesh>::AddVertices(mesh, group_vertex_buffer.size());
	for (size_t i = 0; i < group_vertex_buffer.size(); ++i)
	{
		mesh.vert[i].P() = MyMesh::CoordType(group_vertex_buffer_wrapped[i].pos.x, group_vertex_buffer_wrapped[i].pos.y, group_vertex_buffer_wrapped[i].pos.z);
		mesh.vert[i].N() = MyMesh::CoordType(group_vertex_buffer_wrapped[i].normal.x, group_vertex_buffer_wrapped[i].normal.y, group_vertex_buffer_wrapped[i].normal.z);
		mesh.vert[i].T() = vcg::TexCoord2f(group_vertex_buffer_wrapped[i].pos.w, group_vertex_buffer_wrapped[i].normal.w);

		if (fabs(mesh.vert[i].P().X()) < 0.0001)
		{
			flagX = true;
		}
		if (fabs(mesh.vert[i].P().Y()) < 0.0001)
		{
			flagY = true;
		}
		if (fabs(mesh.vert[i].P().Z()) < 0.0001)
		{
			flagZ = true;
		}
	}

	vcg::tri::Allocator<MyMesh>::AddFaces(mesh, group_vertex_indices.size() / 3);
	for (size_t i = 0; i < group_vertex_indices.size(); i += 3)
	{
		mesh.face[i / 3].V(0) = &mesh.vert[group_vertex_indices[i]];
		mesh.face[i / 3].V(1) = &mesh.vert[group_vertex_indices[i + 1]];
		mesh.face[i / 3].V(2) = &mesh.vert[group_vertex_indices[i + 2]];
	}
	
	vcg::tri::Clean<MyMesh>::RemoveUnreferencedVertex(mesh, true);
	vcg::tri::Clean<MyMesh>::RemoveDuplicateVertexWithNormalAndUV(mesh);
	vcg::tri::Allocator<MyMesh>::CompactFaceVector(mesh);
	vcg::tri::Allocator<MyMesh>::CompactVertexVector(mesh);
	vcg::tri::UpdateNormal<MyMesh>::PerVertexNormalized(mesh);
	vcg::tri::UpdateTopology<MyMesh>::VertexFace(mesh);
	vcg::tri::UpdateBounding<MyMesh>::Box(mesh);

	vcg::Point3f min           = mesh.bbox.min;

	glm::vec3 original_min     = glm::vec3(mesh.bbox.min.X(), mesh.bbox.min.Y(), mesh.bbox.min.Z());
	glm::vec3 original_max     = glm::vec3(mesh.bbox.max.X(), mesh.bbox.max.Y(), mesh.bbox.max.Z());

	vcg::tri::Stat<MyMesh> stat;
	float                  total_area       = stat.ComputeMeshArea(mesh);
	float                  triangle_size    = sqrtf(total_area / static_cast<float>(mesh.fn));
	float                  current_size     = std::max(triangle_size, THRESH_POINTS_ARE_SAME);
	float                  desired_size     = 0.25f;
	float                  scale_factor     = desired_size / current_size;
	/*scale_factor                            = 1.0f;*/
	float inv_scale_factor                  = 1.0 / scale_factor;
	
	for (auto &v : mesh.vert)
	{
		v.P() = (v.P() - min) * scale_factor;
	}

	vcg::tri::UpdateBounding<MyMesh>::Box(mesh);

	uint32_t before_count = CountConnectedComponents(mesh);

	vcg::tri::TriEdgeCollapseQuadricParameter qparams;
	qparams.QualityThr            = 0.3;
	qparams.OptimalPlacement      = false;
	qparams.SVDPlacement          = false;
	qparams.PreserveTopology      = false;
	qparams.BoundaryQuadricWeight = 1.8f;
	qparams.PreserveBoundary      = false;
	qparams.QualityCheck          = true;
	qparams.NormalCheck           = true;

	vcg::math::Quadric<double> QZero;
	QZero.SetZero();
	vcg::tri::QuadricTemp TD(mesh.vert, QZero);
	vcg::tri::QHelper::TDp() = &TD;

	vcg::LocalOptimization<MyMesh> decimator(mesh, &qparams);
	decimator.Init<vcg::tri::MyTriEdgeCollapse>();

	const float threshold         = THRESHOLD;
	uint32_t    target_face_count = group_vertex_indices.size() / 3 * threshold;
	float       error_limit       = target_error * target_error / (scale_factor * scale_factor);
	decimator.SetTargetSimplices(target_face_count);
	decimator.SetTargetMetric(error_limit);
	decimator.SetTimeBudget(1.0f);
	decimator.SetTargetOperations(100000);

	// ====== 6. 执行简化 ======
	while (decimator.DoOptimization() &&
	       mesh.fn > target_face_count &&
	       decimator.currMetric < error_limit)
	{
	}

	decimator.Finalize<vcg::tri::MyTriEdgeCollapse>();

	for (auto &v : mesh.vert)
	{
		v.P() = v.P() * inv_scale_factor + min;
	}

	vcg::tri::Clean<MyMesh>::RemoveDegenerateFace(mesh);
	vcg::tri::Clean<MyMesh>::RemoveDuplicateFace(mesh);
	vcg::tri::Allocator<MyMesh>::CompactVertexVector(mesh);
	vcg::tri::Allocator<MyMesh>::CompactFaceVector(mesh);
	vcg::tri::UpdateNormal<MyMesh>::PerVertexNormalized(mesh);
	vcg::tri::UpdateBounding<MyMesh>::Box(mesh);

	uint32_t after_count = CountConnectedComponents(mesh);

	float weighted = 1.0f;
	if (before_count >= after_count)
	{
		weighted = glm::clamp(static_cast<float>(before_count) / static_cast<float>(after_count), 1.0f, 5.0f);
	}
	else
	{
		weighted = glm::clamp(static_cast<float>(after_count) / static_cast<float>(before_count), 1.0f, 5.0f);
	}

	double total_error = sqrtf(decimator.currMetric) * inv_scale_factor * pow(weighted, 1);

	vcg::tri::QHelper::TDp() = nullptr;

	/*LOGI("before simplify: {}, after simplify: {}, total error: {}", group_vertex_indices.size() / 3, mesh.fn, total_error);*/

	uint32_t vertex_offset = vertex_positions.size();
	uint32_t vertex_count  = mesh.vn;

	if (mesh.fn < group_vertex_indices.size() / 3)
	{

		glm::vec3 mesh_min{+INFINITY, +INFINITY, +INFINITY};
		glm::vec3 mesh_max{-INFINITY, -INFINITY, -INFINITY};

		for (size_t i = 0; i < vertex_count; i++)
		{
			auto &v = mesh.vert[i];

			if (v.P().X() > original_max.x || v.P().X() < original_min.x)
			{
				if (v.P().X() > original_max.x && (v.P().X() - original_max.x) / (original_max.x - original_min.x + 0.000001f) > 0.1)
				{
					LOGE("vertex position x is out of original range: {}", (v.P().X() - original_max.x) / (original_max.x - original_min.x + 0.000001f));
				}
				else if (v.P().X() < original_min.x && (original_min.x - v.P().X()) / (original_max.x - original_min.x + 0.000001f) > 0.1)
				{
					LOGE("vertex position x is out of original range: {}", (v.P().X() - original_min.x) / (original_max.x - original_min.x + 0.000001f));
				}
				//if (i == 0)
				//	v.P().X() = mesh.vert[i + 1].P().X() + 0.0001f; // avoid zero position
				//else
				//	v.P().X() = mesh.vert[i - 1].P().X() + 0.0001f; // avoid zero position
			}

			if (v.P().Y() > original_max.y || v.P().Y() < original_min.y)
			{
				if (v.P().Y() > original_max.y && (v.P().Y() - original_max.y) / (original_max.y - original_min.y + 0.000001f) > 0.1)
				{
					LOGE("vertex position y is out of original range: {}", (v.P().Y() - original_max.y) / (original_max.y - original_min.y + 0.000001f));
				}
				else if (v.P().Y() < original_min.y && (original_min.y - v.P().Y()) / (original_max.y - original_min.y + 0.000001f) > 0.1)
				{
					LOGE("vertex position y is out of original range: {}", (v.P().Y() - original_min.y) / (original_max.y - original_min.y + 0.000001f));
				}
				//if (i == 0)
				//	v.P().Y() = mesh.vert[i + 1].P().Y() + 0.0001f;        // avoid zero position
				//else
				//	v.P().Y() = mesh.vert[i - 1].P().Y() + 0.0001f;        // avoid zero position
			}

			if (v.P().Z() > original_max.z || v.P().Z() < original_min.z)
			{
				if (v.P().Z() > original_max.z && (v.P().Z() - original_max.z) / (original_max.z - original_min.z + 0.000001f) > 0.1)
				{
					LOGE("vertex position z is out of original range: {}", (v.P().Z() - original_max.z) / (original_max.z - original_min.z + 0.000001f));
				}
				else if (v.P().Z() < original_min.z && (original_min.z - v.P().Z()) / (original_max.z - original_min.z + 0.000001f) > 0.1)
				{
					LOGE("vertex position z is out of original range: {}", (v.P().Z() - original_min.z) / (original_max.z - original_min.z + 0.000001f));
				}
				//if (i == 0)
				//	v.P().Z() = mesh.vert[i + 1].P().Z() + 0.0001f;        // avoid zero position
				//else
				//	v.P().Z() = mesh.vert[i - 1].P().Z() + 0.0001f;        // avoid zero position
			}

			vertex_positions.push_back(glm::vec3(v.cP().X(), v.cP().Y(), v.cP().Z()));
			PackedVertex vertex;
			vertex.pos    = glm::vec4(v.cP().X(), v.cP().Y(), v.cP().Z(), v.cT().U());
			vertex.normal = glm::vec4(v.cN().X(), v.cN().Y(), v.cN().Z(), v.cT().V());
			vertices.push_back(vertex);

			mesh_min = glm::min(mesh_min, glm::vec3(vertex.pos.xyz));
			mesh_max = glm::max(mesh_max, glm::vec3(vertex.pos.xyz));
		}

		group_to_mesh_vertex_remap.resize(vertex_count);
		for (size_t i = 0; i < vertex_count; ++i)
		{
			group_to_mesh_vertex_remap[i] = vertex_offset + i;
		}

		std::vector<uint32_t> simplified_index_buffer;

		for (auto &f : mesh.face)
		{
			for (int i = 0; i < 3; ++i)
			{
				simplified_index_buffer.push_back(vcg::tri::Index(mesh, f.V(i)));
				assert(vcg::tri::Index(mesh, f.V(0)) != vcg::tri::Index(mesh, f.V(1)));
				assert(vcg::tri::Index(mesh, f.V(0)) != vcg::tri::Index(mesh, f.V(2)));
				assert(vcg::tri::Index(mesh, f.V(1)) != vcg::tri::Index(mesh, f.V(2)));
				assert(simplified_index_buffer.back() < vertex_count);
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

		cluster_groups[cluster_group_index].parent_error           = mesh_space_error;
		cluster_groups[cluster_group_index].parent_bounding_sphere = simplified_cluster_bounds;

		append_meshlets(meshlet_vertices, meshlet_triangles, meshlets, &vertex_positions[vertex_offset].x, vertex_count, simplified_index_buffer, mesh_space_error, simplified_cluster_bounds, group_to_mesh_vertex_remap);

		return true;
	}
	return false;
}

void xihe::sg::generate_lod(const MeshPrimitiveData &primitive, std::vector<uint32_t> &scene_data, std::vector<ClusterGroup> &cluster_groups, std::vector<Cluster> &clusters)
{
	LOGI("Building lod...");

	Timer timer;
	timer.start();

	static int num = 0;
	num++;
	LOGI("SubMesh {}: {} vertices, {} triangles", num, primitive.vertex_count, primitive.index_count / 3);

	const float *raw_positions = reinterpret_cast<const float *>(primitive.attributes.at("position").data.data());
	const float *raw_normals   = reinterpret_cast<const float *>(primitive.attributes.at("normal").data.data());
	const float *raw_uvs       = nullptr;
	if (primitive.attributes.find("texcoord_0") != primitive.attributes.end())
	{
		raw_uvs = reinterpret_cast<const float *>(primitive.attributes.at("texcoord_0").data.data());
	}

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

	MyMesh mesh;

	vcg::tri::Allocator<MyMesh>::AddVertices(mesh, primitive.vertex_count);
	for (size_t i = 0; i < primitive.vertex_count; ++i)
	{
		mesh.vert[i].P() = MyMesh::CoordType(raw_positions[3 * i], raw_positions[3 * i + 1], raw_positions[3 * i + 2]);
		mesh.vert[i].N() = MyMesh::CoordType(raw_normals[3 * i], raw_normals[3 * i + 1], raw_normals[3 * i + 2]);
		if (raw_uvs)
		{
			mesh.vert[i].T() = vcg::TexCoord2f(raw_uvs[2 * i], raw_uvs[2 * i + 1]);
		}
		else
		{
			mesh.vert[i].T() = vcg::TexCoord2f(0.0f, 0.0f);
		}
	}

	vcg::tri::Allocator<MyMesh>::AddFaces(mesh, primitive.index_count / 3);
	for (size_t i = 0; i < primitive.index_count; i += 3)
	{
		mesh.face[i / 3].V(0) = &mesh.vert[index_data_32[i]];
		mesh.face[i / 3].V(1) = &mesh.vert[index_data_32[i + 1]];
		mesh.face[i / 3].V(2) = &mesh.vert[index_data_32[i + 2]];
	}

	// Step 1: 清除未引用顶点
	vcg::tri::Clean<MyMesh>::RemoveUnreferencedVertex(mesh, true);

	// Step 2: 清除重复顶点（根据位置）
	vcg::tri::Clean<MyMesh>::RemoveDuplicateVertexWithNormalAndUV(mesh);

	// Step 3: 清除重复面（顶点指针相同）
	vcg::tri::Clean<MyMesh>::RemoveDuplicateFace(mesh);

	// Step 4: 清除退化面（三点共线或重复）
	vcg::tri::Clean<MyMesh>::RemoveDegenerateFace(mesh);

	// Step 5: 压缩顶点和面数组
	vcg::tri::Allocator<MyMesh>::CompactFaceVector(mesh);
	vcg::tri::Allocator<MyMesh>::CompactVertexVector(mesh);

	// Step 6: 更新包围盒
	vcg::tri::UpdateBounding<MyMesh>::Box(mesh);

	size_t vertex_count = mesh.vn;
	size_t index_count  = mesh.fn * 3;

	glm::vec3 min{+INFINITY, +INFINITY, +INFINITY};
	glm::vec3 max{-INFINITY, -INFINITY, -INFINITY};

	std::vector<glm::vec3>    vertex_positions(vertex_count);
	std::vector<PackedVertex> vertices(vertex_count);
	for (size_t i = 0; i < vertex_count; ++i)
	{
		vertex_positions[i] = glm::vec3(
		    mesh.vert[i].cP().X(),
		    mesh.vert[i].cP().Y(),
		    mesh.vert[i].cP().Z());
		vertices[i].pos = glm::vec4(
		    mesh.vert[i].cP().X(),
		    mesh.vert[i].cP().Y(),
		    mesh.vert[i].cP().Z(),
		    mesh.vert[i].cT().U());
		vertices[i].normal = glm::vec4(
		    mesh.vert[i].cN().X(),
		    mesh.vert[i].cN().Y(),
		    mesh.vert[i].cN().Z(),
		    mesh.vert[i].cT().V());
		min = glm::min(min, vertex_positions[i]);
		max = glm::max(max, vertex_positions[i]);
	}

	std::vector<uint32_t> index_buffer(index_count);
	for (size_t i = 0; i < mesh.fn; i++)
	{
		index_buffer[i * 3 + 0] = vcg::tri::Index(mesh, mesh.face[i].V(0));
		index_buffer[i * 3 + 1] = vcg::tri::Index(mesh, mesh.face[i].V(1));
		index_buffer[i * 3 + 2] = vcg::tri::Index(mesh, mesh.face[i].V(2));
	}

	glm::vec4 bounds = glm::vec4((min + max) / 2.0f, glm::distance(min, max) / 2.0f);

	std::vector<uint32_t> meshlet_vertices;
	std::vector<uint32_t> meshlet_triangles;
	std::vector<Meshlet>  meshlets;

	append_meshlets(meshlet_vertices, meshlet_triangles, meshlets, &vertex_positions[0].x, primitive.vertex_count, index_buffer, 0.0f, bounds);

	LOGI("LOD {}: {} meshlets, {} vertices, {} triangles", 0, meshlets.size(), vertex_positions.size(), meshlet_triangles.size());
	
	const int max_lod = 10;
	
	std::vector<uint8_t> group_vertex_indices;
	
	size_t previous_meshlets_start = 0;
	
	for (int lod = 0; lod < max_lod; ++lod)
	{
		Timer lod_timer;
		lod_timer.start();
	
		float t_lod = lod / (float) max_lod;
	
		std::span<Meshlet> previous_level_meshlets = std::span{meshlets.data() + previous_meshlets_start, meshlets.size() - previous_meshlets_start};
	
		const std::vector<MeshletGroup> groups = group_meshlets_remap(meshlet_vertices, meshlet_triangles, previous_level_meshlets);

		// special check: if there is only one meshlet, we needn't simplify
		if (groups.size() == 1 && groups[0].meshlets.size() == 1)
		{
			append_meshlet_groups(vertices, scene_data, cluster_groups, clusters, meshlet_vertices, meshlet_triangles, previous_level_meshlets, groups[0], lod);
			break;
		}
	
		const std::size_t new_meshlet_start       = meshlets.size();
	
		float target_error = 0.05f + 0.85f * t_lod;
	
		Timer simplify_timer;
		simplify_timer.start();
	
		double meshlet_group_time  = 0;
		double simplify_group_time = 0;
	
		for (const auto &group : groups)
		{
			// meshlets vector is modified during the loop
			previous_level_meshlets = std::span{meshlets.data() + previous_meshlets_start, meshlets.size() - previous_meshlets_start};
	
			Timer meshlet_group_timer;
			meshlet_group_timer.start();
	
			append_meshlet_groups(vertices, scene_data, cluster_groups, clusters, meshlet_vertices, meshlet_triangles, previous_level_meshlets, group, lod);
	
			meshlet_group_time = meshlet_group_timer.stop();
	
			/*LOGI("meshlet group time: {}", meshlet_group_time);*/
	
			Timer simplify_group_timer;
			simplify_group_timer.start();
	
			bool is_simplified = simplify_group(vertex_positions, vertices, cluster_groups, meshlet_vertices, meshlet_triangles, meshlets, previous_level_meshlets, group, target_error);
	
			simplify_group_time = simplify_group_timer.stop();
	
			/*LOGI("simplify group time: {}", simplify_group_time);*/
		}
	
		auto simplify_time = simplify_timer.stop();
		LOGI("meshlet group time: {} seconds, simplify group time: {} seconds, simplify time: {} seconds", meshlet_group_time, simplify_group_time, simplify_time);
	
		auto lod_time = lod_timer.stop();
		 LOGI("lod time: {} seconds", lod_time);
	
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
	LOGI("Time spent building lod: {} seconds.", elapsed_time);
}
}        // namespace xihe::sg