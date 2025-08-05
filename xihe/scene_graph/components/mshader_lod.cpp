#include "mshader_lod.h"
#include "KDTree.h"
#include <glm/gtx/norm.hpp>
#include <common/timer.h>
#include <tbb/parallel_for.h>
#include "metis.h"

#define USE_WELDING 1
namespace xihe::sg
{

// triangle_count和triangle_offset有点坑。现在triangle_count视为三角形数量，但triangle_offset视为索引的偏移，即三角形偏移量的三倍。
// 原本xihe使用的是三个uint8的index压成一个uint32，现在改为每个index都是uint32。

// 把低级别的meshlet分组，每个组包含多个meshlet，相当于比原先meshlet更高一级的cluster。后续会用meshopt_simplify来简化cluster
static std::vector<MeshletGroup> groupMeshlets(MeshPrimitiveData &primitive, const std::vector<uint32_t> &meshlet_vertices, const std::vector<uint32_t> &meshlet_triangles, std::span<Meshlet> meshlets)
{
	// ===== Build meshlet connections
	auto groupWithAllMeshlets = [&]() {
		MeshletGroup group;
		for (int i = 0; i < meshlets.size(); ++i)
		{
			group.meshlets.push_back(i);
		}
		return std::vector{group};
	};
	if (meshlets.size() < 16)
	{
		return groupWithAllMeshlets();
	}

	// meshlets represented by their index into 'meshlets'
	std::unordered_map<MeshletEdge, std::vector<std::size_t>, MeshletEdgeHasher> edges2Meshlets;
	std::unordered_map<std::size_t, std::vector<MeshletEdge>>                    meshlets2Edges;        // probably could be a vector

	// for each meshlet
	for (std::size_t meshletIndex = 0; meshletIndex < meshlets.size(); meshletIndex++)
	{
		const auto &meshlet        = meshlets[meshletIndex];
		auto        getVertexIndex = [&](std::size_t index) {
            uint32_t packed_vertex_index = meshlet_triangles[index / 3 + meshlet.triangle_offset];
            uint8_t  vertex_index        = (packed_vertex_index >> ((index % 3) * 8)) & 0xFF;
            // index % 3 == 0, 不右移；index % 3 == 1, 右移8位；index % 3 == 2, 右移16位；最后取低8位
			return meshlet_vertices[meshlet.vertex_offset + vertex_index];

            /*return primitive.meshlet_vertex_indices[primitive.meshlet_triangles[index + meshlet.triangle_offset] + meshlet.vertex_offset];*/
		};

		const std::size_t triangleCount = meshlet.triangle_count;
		// for each triangle of the meshlet
		for (std::size_t triangleIndex = 0; triangleIndex < triangleCount; triangleIndex++)
		{
			// for each edge of the triangle
			for (std::size_t i = 0; i < 3; i++)
			{
				MeshletEdge edge{getVertexIndex(i + triangleIndex * 3), getVertexIndex(((i + 1) % 3) + triangleIndex * 3)};
				edges2Meshlets[edge].push_back(meshletIndex);
				meshlets2Edges[meshletIndex].emplace_back(edge);
			}
		}
	}

	// remove edges which are not connected to 2 different meshlets
	std::erase_if(edges2Meshlets, [&](const auto &pair) {
		return pair.second.size() <= 1;
	});

	if (edges2Meshlets.empty())
	{
		return groupWithAllMeshlets();
	}

	// at this point, we have basically built a graph of meshlets, in which edges represent which meshlets are connected together

	// idx_t comes from METIS and corresponds to std::uint64_t in my build of METIS
	std::vector<MeshletGroup> groups;

	idx_t vertex_count = meshlets.size();            // vertex count, from the point of view of METIS, where Meshlet = vertex
	idx_t ncon        = 1;                          // only one constraint, minimum required by METIS
	idx_t nparts      = meshlets.size() / 8;        // groups of 4 // -> 8
	assert(nparts > 1);
	idx_t options[METIS_NOPTIONS];
	METIS_SetDefaultOptions(options);

	// edge-cut, ie minimum cost betweens groups.
	options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
	options[METIS_OPTION_CCORDER] = 1;        // identify connected components first

	// prepare storage for partition data
	// each vertex will get its partition index inside this vector after the edge-cut
	std::vector<idx_t> partition(vertex_count);

	// xadj
	std::vector<idx_t> xadjacency;
	xadjacency.reserve(vertex_count + 1);

	// adjncy
	std::vector<idx_t> edgeAdjacency;
	// weight of each edge
	std::vector<idx_t> edgeWeights;

	for (std::size_t meshletIndex = 0; meshletIndex < meshlets.size(); meshletIndex++)
	{
		std::size_t startIndexInEdgeAdjacency = edgeAdjacency.size();
		for (const auto &edge : meshlets2Edges[meshletIndex])
		{
			auto connectionsIter = edges2Meshlets.find(edge);
			if (connectionsIter == edges2Meshlets.end())
			{
				continue;
			}
			const auto &connections = connectionsIter->second;
			for (const auto &connectedMeshlet : connections)
			{
				if (connectedMeshlet != meshletIndex)
				{
					auto existingEdgeIter = std::find(edgeAdjacency.begin() + startIndexInEdgeAdjacency, edgeAdjacency.end(), connectedMeshlet);
					if (existingEdgeIter == edgeAdjacency.end())
					{
						// first time we see this connection to the other meshlet
						edgeAdjacency.emplace_back(connectedMeshlet);
						edgeWeights.emplace_back(1);
					}
					else
					{
						// not the first time! increase number of times we encountered this meshlet
						std::ptrdiff_t d = std::distance(edgeAdjacency.begin(), existingEdgeIter);
						assert(d >= 0);
						assert(d < edgeWeights.size());
						edgeWeights[d]++;
					}
				}
			}
		}
		xadjacency.push_back(startIndexInEdgeAdjacency);
	}
	xadjacency.push_back(edgeAdjacency.size());
	assert(xadjacency.size() == meshlets.size() + 1);
	assert(edgeAdjacency.size() == edgeWeights.size());
	
	for (const std::size_t &edgeAdjIndex : xadjacency)
	{
		assert(edgeAdjIndex <= edgeAdjacency.size());
	}
	for (const std::size_t &vertexIndex : edgeAdjacency)
	{
		assert(vertexIndex <= vertex_count);
	}

	idx_t edgeCut;        // final cost of the cut found by METIS
	int   result = METIS_PartGraphKway(&vertex_count,
	                                   &ncon,
	                                   xadjacency.data(),
	                                   edgeAdjacency.data(),
	                                   NULL,           /* vertex weights */
	                                   NULL,           /* vertex size */
	                                   edgeWeights.data(),
	                                   &nparts,
	                                   NULL,
	                                   NULL,
	                                   options,
	                                   &edgeCut,
	                                   partition.data());

	assert(result == METIS_OK);

	// ===== Group meshlets together
	groups.resize(nparts);
	for (std::size_t i = 0; i < meshlets.size(); i++)
	{
		idx_t partitionNumber = partition[i];
		groups[partitionNumber].meshlets.push_back(i);
	}
	//for (int i=0;i<nparts;i++)
	//	if (groups[i].meshlets.size() == 0)
	//	{
	//		int j = 0;
	//	}
	return groups;
	// end of function
}

// 仅仅改变了getVertexIndex的实现，现在返回的是vertexRemap中的索引
static std::vector<MeshletGroup> groupMeshletsRemap(const MeshPrimitiveData &primitive, const std::vector<uint32_t> &meshlet_vertices, const std::vector<uint32_t> &meshlet_triangles, std::span<Meshlet> meshlets, std::span<const std::int64_t> vertexRemap)
{
	// ===== Build meshlet connections
	auto groupWithAllMeshlets = [&]() {
		MeshletGroup group;
		for (int i = 0; i < meshlets.size(); ++i)
		{
			group.meshlets.push_back(i);
		}
		return std::vector{group};
	};
	if (meshlets.size() < 16)
	{
		return groupWithAllMeshlets();
	}

	// meshlets represented by their index into 'meshlets'
	std::unordered_map<MeshletEdge, std::vector<std::size_t>, MeshletEdgeHasher> edges2Meshlets;
	std::unordered_map<std::size_t, std::vector<MeshletEdge>>                    meshlets2Edges;        // probably could be a vector

	// for each meshlet
	for (std::size_t meshletIndex = 0; meshletIndex < meshlets.size(); meshletIndex++)
	{
		const auto &meshlet        = meshlets[meshletIndex];
		auto        getVertexIndex = [&](std::size_t index) {

           uint32_t packed_vertex_index = meshlet_triangles[index / 3 + meshlet.triangle_offset];
           uint8_t  vertex_index        = (packed_vertex_index >> ((index % 3) * 8)) & 0xFF;
                // index % 3 == 0, 不右移；index % 3 == 1, 右移8位；index % 3 == 2, 右移16位；最后取低8位
			return static_cast<std::size_t>(
				vertexRemap[meshlet_vertices[meshlet.vertex_offset + vertex_index]
				]);
		};

		const std::size_t triangleCount = meshlet.triangle_count;
		// for each triangle of the meshlet
		for (std::size_t triangleIndex = 0; triangleIndex < triangleCount; triangleIndex++)
		{
			// for each edge of the triangle
			for (std::size_t i = 0; i < 3; i++)
			{
				MeshletEdge edge{getVertexIndex(i + triangleIndex * 3), getVertexIndex(((i + 1) % 3) + triangleIndex * 3)};
				edges2Meshlets[edge].push_back(meshletIndex);
				meshlets2Edges[meshletIndex].emplace_back(edge);
			}
		}
	}

	// remove edges which are not connected to 2 different meshlets
	std::erase_if(edges2Meshlets, [&](const auto &pair) {
		return pair.second.size() <= 1;
	});

	if (edges2Meshlets.empty())
	{
		return groupWithAllMeshlets();
	}

	// at this point, we have basically built a graph of meshlets, in which edges represent which meshlets are connected together

	std::vector<MeshletGroup> groups;

	idx_t vertex_count = meshlets.size();        // vertex count, from the point of view of METIS, where Meshlet = vertex
	idx_t ncon        = 1;
	idx_t nparts      = meshlets.size() / 8;
	assert(nparts > 1);
	idx_t options[METIS_NOPTIONS];
	METIS_SetDefaultOptions(options);
	options[METIS_OPTION_OBJTYPE]   = METIS_OBJTYPE_CUT;
	options[METIS_OPTION_CCORDER]   = 1;        // identify connected components first
	options[METIS_OPTION_NUMBERING] = 0;

	std::vector<idx_t> partition(vertex_count);
	
	// xadj
	std::vector<idx_t> xadjacency;
	xadjacency.reserve(vertex_count + 1);

	// adjncy
	std::vector<idx_t> edgeAdjacency;
	// weight of each edge
	std::vector<idx_t> edgeWeights;

	for (std::size_t meshletIndex = 0; meshletIndex < meshlets.size(); meshletIndex++)
	{
		std::size_t startIndexInEdgeAdjacency = edgeAdjacency.size();
		for (const auto &edge : meshlets2Edges[meshletIndex]) // 枚举这些边
		{
			auto connectionsIter = edges2Meshlets.find(edge);
			if (connectionsIter == edges2Meshlets.end())      // 如果这条边没有连接到其他meshlet
			{
				continue;
			}
			const auto &connections = connectionsIter->second;
			for (const auto &connectedMeshlet : connections)
			{
				if (connectedMeshlet != meshletIndex)
				{
					auto existingEdgeIter = std::find(edgeAdjacency.begin() + startIndexInEdgeAdjacency, edgeAdjacency.end(), connectedMeshlet); // 看这个meshlet有没有出现过
					if (existingEdgeIter == edgeAdjacency.end()) // 没有出现过就加进去，并设置初始权重为1
					{
						// first time we see this connection to the other meshlet
						edgeAdjacency.emplace_back(connectedMeshlet);
						edgeWeights.emplace_back(1);
					}
					else // 出现过，权重++
					{
						// not the first time! increase number of times we encountered this meshlet
						std::ptrdiff_t d = std::distance(edgeAdjacency.begin(), existingEdgeIter);
						assert(d >= 0);
						assert(d < edgeWeights.size());
						edgeWeights[d]++;
					}
				}
			}
		}
		xadjacency.push_back(startIndexInEdgeAdjacency);
	}
	xadjacency.push_back(edgeAdjacency.size());
	assert(xadjacency.size() == meshlets.size() + 1);
	assert(edgeAdjacency.size() == edgeWeights.size());

	for (const std::size_t &edgeAdjIndex : xadjacency)
	{
		assert(edgeAdjIndex <= edgeAdjacency.size());
	}
	for (const std::size_t &vertexIndex : edgeAdjacency)
	{
		assert(vertexIndex <= vertex_count);
	}

	idx_t edgeCut;        // final cost of the cut found by METIS
	int   result = METIS_PartGraphKway(&vertex_count,
	                                   &ncon,
	                                   xadjacency.data(),
	                                   edgeAdjacency.data(),
	                                   NULL, /* vertex weights */
	                                   NULL, /* vertex size */
	                                   edgeWeights.data(),
	                                   &nparts,
	                                   NULL,
	                                   NULL,
	                                   options,
	                                   &edgeCut,
	                                   partition.data());

	assert(result == METIS_OK);

	// ===== Group meshlets together
	groups.resize(nparts);
	for (std::size_t i = 0; i < meshlets.size(); i++)
	{
		idx_t partitionNumber = partition[i];
		groups[partitionNumber].meshlets.push_back(i);
	}

	return groups;
}

void append_meshlets(std::vector<uint32_t> &meshlet_vertices, std::vector<uint32_t> &meshlet_triangles, std::vector<Meshlet> &meshlets, const float *vertex_buffer, uint32_t vertex_buffer_count, std::span<std::uint32_t> index_buffer, float clusterError, std::span<size_t> vertex_remap = std::span<size_t>())
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
	    vertex_buffer,
	    vertex_buffer_count,
	    sizeof(float) * 3,
	    max_vertices,
	    max_triangles,
	    cone_weight);

	local_meshlets.resize(meshlet_count);

	const std::size_t meshlet_offset  = meshlets.size();
	const std::size_t vertex_offset   = meshlet_vertices.size();
	const std::size_t triangle_offset = meshlet_triangles.size();

	const meshopt_Meshlet &last           = local_meshlets[meshlet_count - 1];
	const std::size_t      vertex_count   = last.vertex_offset + last.vertex_count;
	std::size_t            triangle_count = last.triangle_offset / 3 + last.triangle_count;

	meshlet_vertices.resize(vertex_offset + vertex_count);
	meshlet_triangles.resize(triangle_offset + triangle_count);
	meshlets.resize(meshlet_offset + meshlet_count);

	if (vertex_remap.empty())
	{
		tbb::parallel_for(std::size_t(0), vertex_count, [&](std::size_t index) {
			meshlet_vertices[vertex_offset + index] = meshlet_vertex_indices[index];
		});
	}
	else
	{
		tbb::parallel_for(std::size_t(0), vertex_count, [&](std::size_t index) {
			meshlet_vertices[vertex_offset + index] = vertex_remap[meshlet_vertex_indices[index]];
		});
	}

	// 这里还是应该将3个uint8的index压成一个uint32，因为meshlet_triangle_indices的值不会超过64
	tbb::parallel_for(std::size_t(0), triangle_count, [&](std::size_t index) {
		uint8_t idx0 = meshlet_triangle_indices[index * 3 + 0];
		uint8_t idx1 = meshlet_triangle_indices[index * 3 + 1];
		uint8_t idx2 = meshlet_triangle_indices[index * 3 + 2];

		uint32_t packed_triangle = idx0 | (idx1 << 8) | (idx2 << 16);

		meshlet_triangles[triangle_offset + index] = packed_triangle;
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
		    local_meshlet.triangle_count, vertex_buffer, vertex_buffer_count, sizeof(float) * 3);

		meshlet.center = glm::vec3(meshlet_bounds.center[0], meshlet_bounds.center[1], meshlet_bounds.center[2]);
		meshlet.radius = meshlet_bounds.radius;

		meshlet.cone_axis   = glm::vec3(meshlet_bounds.cone_axis[0], meshlet_bounds.cone_axis[1], meshlet_bounds.cone_axis[2]);
		meshlet.cone_cutoff = meshlet_bounds.cone_cutoff;

		meshlet.cone_apex = glm::vec3(meshlet_bounds.cone_apex[0], meshlet_bounds.cone_apex[1], meshlet_bounds.cone_apex[2]);

		meshlet.cluster_error = clusterError;
	});
}

//// 从最新一级别的LOD重新拆分meshlet，附加到原来的primitive数组后面
//static void appendMeshlets(MeshPrimitiveData &primitive_data, std::span<std::uint32_t> index_data_32, const glm::vec4 &clusterBounds, float clusterError)
//{	
//	constexpr std::size_t max_vertices  = 64;
//	constexpr std::size_t max_triangles = 124;
//	const float           cone_weight   = 0.0f;
//	
//	const std::size_t            max_meshlets    = meshopt_buildMeshletsBound(index_data_32.size(), max_vertices, max_triangles);
//	
//	std::vector<meshopt_Meshlet> local_meshlets(max_meshlets);
//	std::vector<unsigned int>    meshlet_vertex_indices(max_meshlets * max_vertices);
//	std::vector<unsigned char>   meshlet_triangle_indices(max_meshlets * max_triangles * 3);
//
//	auto vertex_positions = reinterpret_cast<const float *>(primitive_data.attributes.at("position").data.data());
//
//	size_t meshlet_count = meshopt_buildMeshlets(
//			local_meshlets.data(), 
//			meshlet_vertex_indices.data(), 
//			meshlet_triangle_indices.data(),
//			index_data_32.data(), 
//			index_data_32.size(),
//	        vertex_positions,
//	        primitive_data.vertex_count,
//	        sizeof(float) * 3,
//			max_vertices, 
//		    max_triangles,
//			cone_weight
//	);
//
//	local_meshlets.resize(meshlet_count);
//
//	// 这个是上一个LoD的offset
//
//	 const std::size_t meshlet_offset  = primitive_data.meshlets.size();
//	 const std::size_t vertex_offset   = primitive_data.meshlet_vertices.size();
//	 const std::size_t triangle_offset = primitive_data.meshlet_triangles.size();
//
//	// 这个是当前LoD的vertex和triangle的数量
//
//	const meshopt_Meshlet &last           = local_meshlets[meshlet_count - 1];
//	const std::size_t      vertex_count   = last.vertex_offset + last.vertex_count;
//	std::size_t            triangle_count = last.triangle_offset / 3 + last.triangle_count;
//
//	primitive_data.meshlet_vertices.resize(vertex_offset + vertex_count);
//	primitive_data.meshlet_triangles.resize(triangle_offset + triangle_count);
//	primitive_data.meshlets.resize(meshlet_offset + meshlet_count);
//
//	// 改写成parallel for
//	tbb::parallel_for(std::size_t(0), vertex_count, [&](std::size_t index)
//	{
//		primitive_data.meshlet_vertices[vertex_offset + index] = meshlet_vertex_indices[index];
//	});
//
//	// 这里还是应该将3个uint8的index压成一个uint32，因为meshlet_triangle_indices的值不会超过64
//	tbb::parallel_for(std::size_t(0), triangle_count, [&](std::size_t index)
//	{
//		uint8_t idx0 = meshlet_triangle_indices[index * 3 + 0];
//		uint8_t idx1 = meshlet_triangle_indices[index * 3 + 1];
//		uint8_t idx2 = meshlet_triangle_indices[index * 3 + 2];
//
//		uint32_t packed_triangle = idx0 | (idx1 << 8) | (idx2 << 16);
//
//		primitive_data.meshlet_triangles[triangle_offset + index] = packed_triangle;
//	});
//
//	tbb::parallel_for(std::size_t(0), meshlet_count, [&](std::size_t index)
//	{
//		auto &local_meshlet = local_meshlets[index];
//		auto &meshlet       = primitive_data.meshlets[meshlet_offset + index];
//
//		meshlet.vertex_offset = vertex_offset + local_meshlet.vertex_offset;
//		meshlet.vertex_count  = local_meshlet.vertex_count;
//
//		meshlet.triangle_offset = triangle_offset + local_meshlet.triangle_offset / 3;
//		meshlet.triangle_count  = local_meshlet.triangle_count;
//
//		meshopt_Bounds meshlet_bounds = meshopt_computeMeshletBounds(
//		    meshlet_vertex_indices.data() + local_meshlet.vertex_offset,
//		    meshlet_triangle_indices.data() + local_meshlet.triangle_offset,
//		    local_meshlet.triangle_count, vertex_positions, primitive_data.vertex_count, sizeof(float) * 3);
//
//		meshlet.center = glm::vec3(meshlet_bounds.center[0], meshlet_bounds.center[1], meshlet_bounds.center[2]);
//		meshlet.radius = meshlet_bounds.radius;
//
//		meshlet.cone_axis   = glm::vec3(meshlet_bounds.cone_axis[0], meshlet_bounds.cone_axis[1], meshlet_bounds.cone_axis[2]);
//		meshlet.cone_cutoff = meshlet_bounds.cone_cutoff;
//
//		meshlet.cone_apex = glm::vec3(meshlet_bounds.cone_apex[0], meshlet_bounds.cone_apex[1], meshlet_bounds.cone_apex[2]);
//
//		meshlet.clusterError = clusterError;
//		meshlet.center       = glm::vec3(clusterBounds.x, clusterBounds.y, clusterBounds.z);
//		meshlet.radius       = clusterBounds.w;
//
//	});
//}

/**
 * Find which vertices are part of meshlet boundaries. These should not be merged to avoid cracks between LOD levels
 */
static std::vector<bool> findBoundaryVertices(const MeshPrimitiveData &primitive, const std::vector<uint32_t> &meshlet_vertices, const std::vector<uint32_t> &meshlet_triangles, std::span<Meshlet> meshlets)
{
	std::vector<bool> boundaryVertices;
	boundaryVertices.resize(primitive.vertex_count);

	// 初始化boundaryVertices为false
	for (std::size_t i = 0; i < primitive.vertex_count; i++)
	{
		boundaryVertices[i] = false;
	}

	// meshlets represented by their index into 'previousLevelMeshlets'
	std::unordered_map<MeshletEdge, std::unordered_set<std::size_t>, MeshletEdgeHasher> edges2Meshlets;

	// for each meshlet
	for (std::size_t meshletIndex = 0; meshletIndex < meshlets.size(); meshletIndex++)
	{
		const auto &meshlet        = meshlets[meshletIndex];
		auto        getVertexIndex = [&](std::size_t index) {
            uint32_t packed_vertex_index = meshlet_triangles[index / 3 + meshlet.triangle_offset];
            uint8_t  vertex_index        = (packed_vertex_index >> ((index % 3) * 8)) & 0xFF;
            // index % 3 == 0, 不右移；index % 3 == 1, 右移8位；index % 3 == 2, 右移16位；最后取低8位
			return meshlet_vertices[meshlet.vertex_offset + vertex_index];
		};

		const std::size_t triangleCount = meshlet.triangle_count;
		// for each triangle of the meshlet
		for (std::size_t triangleIndex = 0; triangleIndex < triangleCount; triangleIndex++)
		{
			// for each edge of the triangle
			for (std::size_t i = 0; i < 3; i++)
			{
				MeshletEdge edge{getVertexIndex(i + triangleIndex * 3), getVertexIndex(((i + 1) % 3) + triangleIndex * 3)};
				if (edge.first != edge.second)
				{
					edges2Meshlets[edge].insert(meshletIndex);
				}
			}
		}
	}

	for (const auto &[edge, meshlets] : edges2Meshlets)
	{
		if (meshlets.size() > 1)
		{
			boundaryVertices[edge.first]  = true;
			boundaryVertices[edge.second] = true;
		}
	}

	return boundaryVertices;
}

std::vector<std::int64_t> mergeByDistance(const MeshPrimitiveData &primitive, const std::vector<bool> &boundary, std::span<const VertexWrapper> groupVerticesPreWeld, float maxDistance, float maxUVDistance, const KDTree<VertexWrapper> &kdtree)
{
	// merge vertices which are close enough
	std::vector<std::int64_t> vertexRemap;

	const std::size_t vertex_count = primitive.vertex_count;
	vertexRemap.resize(vertex_count);

#ifndef USE_WELDING

	for (int i = 0; i < vertex_count; i++)
	{
		vertexRemap[i] = i;
	}

#else
	for (auto &v : vertexRemap)
	{
		v = -1;
	}

	std::vector<std::vector<std::size_t>> neighborsForAllVertices;
	neighborsForAllVertices.resize(groupVerticesPreWeld.size());



	tbb::parallel_for(std::size_t(0), groupVerticesPreWeld.size(), [&](std::size_t v) {
		const VertexWrapper &currentVertexWrapped = groupVerticesPreWeld[v];
		if (boundary[currentVertexWrapped.index])
		{
			return;        // no need to compute neighbors
		}
		kdtree.getNeighbors(neighborsForAllVertices[v], currentVertexWrapped, maxDistance);
	});

	//for (std::size_t v = 0; v < groupVerticesPreWeld.size(); v++)
	//{
	//	const VertexWrapper &currentVertexWrapped = groupVerticesPreWeld[v];
	//	if (boundary[currentVertexWrapped.index])
	//	{
	//		continue;        // no need to compute neighbors
	//	}
	//	kdtree.getNeighbors(neighborsForAllVertices[v], currentVertexWrapped, maxDistance);
	//}

	auto vertex_positions = reinterpret_cast<const float *>(primitive.attributes.at("position").data.data());

	const float *vertex_uvs = nullptr;
	if (primitive.attributes.find("texcoord_0") != primitive.attributes.end())
	{
		vertex_uvs = reinterpret_cast<const float *>(primitive.attributes.at("texcoord_0").data.data());
	}

	for (std::int64_t v = 0; v < groupVerticesPreWeld.size(); v++)
	{
		std::int64_t replacement          = -1;
		const auto  &currentVertexWrapped = groupVerticesPreWeld[v];
		if (!boundary[currentVertexWrapped.index])
		{        // boundary vertices must not be merged with others (to avoid cracks)
			auto                 &neighbors     = neighborsForAllVertices[v];
			const glm::vec3 &currentVertexPos = currentVertexWrapped.getPosition(); 
			glm::vec2      currentVertexUV  = glm::vec2(0.0, 0.0);

			if (vertex_uvs)
			{
				const float *vertexUV = vertex_uvs + currentVertexWrapped.index * 2;
				currentVertexUV       = glm::vec2(vertexUV[0], vertexUV[1]);
			}

			float maxDistanceSq   = maxDistance * maxDistance;
			float maxUVDistanceSq = maxUVDistance * maxUVDistance;

			for (const std::size_t &neighbor : neighbors)
			{
				if (vertexRemap[groupVerticesPreWeld[neighbor].index] == -1)
				{
					// due to the way we iterate, all indices starting from v will not be remapped yet
					continue;
				}
				auto otherVertexWrapped            = VertexWrapper(vertex_positions, vertexRemap[groupVerticesPreWeld[neighbor].index]);
				const glm::vec3 &otherVertexPos     = otherVertexWrapped.getPosition();
				const float vertexDistanceSq = glm::distance2(currentVertexPos, otherVertexPos);
				if (vertexDistanceSq <= maxDistanceSq)
				{
					if (!vertex_uvs)
					{
						replacement   = vertexRemap[groupVerticesPreWeld[neighbor].index];
						maxDistanceSq = vertexDistanceSq;
					}
					else
					{
						const float *vertexUV = vertex_uvs + groupVerticesPreWeld[neighbor].index * 2;
						const glm::vec2 &otherVertexUV = glm::vec2(vertexUV[0], vertexUV[1]);
						const float      vertexUVDistanceSq = glm::distance2(currentVertexUV, otherVertexUV);
						if (vertexUVDistanceSq <= maxUVDistanceSq)
						{
							replacement     = vertexRemap[groupVerticesPreWeld[neighbor].index];
							maxDistanceSq   = vertexDistanceSq;
							maxUVDistanceSq = vertexUVDistanceSq;
						}
					}
				}
			}
		}

		if (replacement == -1)
		{
			vertexRemap[currentVertexWrapped.index] = currentVertexWrapped.index;
		}
		else
		{
			vertexRemap[currentVertexWrapped.index] = replacement;
		}
	}
#endif        // USE_WELDING

	return vertexRemap;
	
}

bool simplifyGroup(const MeshPrimitiveData &primitive, std::vector<uint32_t> &meshlet_vertices, std::vector<uint32_t> &meshlet_triangles, std::vector<Meshlet> &meshlets, std::span<Meshlet> &previousLevelMeshlets, const MeshletGroup &group, const std::vector<std::int64_t> &mergeVertexRemap, float targetError)
{
	std::vector<uint32_t> groupVertexIndices;

	std::vector<glm::vec3> groupVertexBuffer;

	std::vector<std::size_t>                     group2meshVertexRemap;
	std::unordered_map<std::size_t, std::size_t> mesh2groupVertexRemap;

	auto vertex_positions = reinterpret_cast<const float *>(primitive.attributes.at("position").data.data());

	// add cluster vertices to this group
	for (const auto &meshletIndex : group.meshlets)
	{
		const auto &meshlet = previousLevelMeshlets[meshletIndex];
		std::size_t start   = groupVertexIndices.size();
		groupVertexIndices.reserve(start + meshlet.triangle_count * 3);
		for (std::size_t j = 0; j < meshlet.triangle_count * 3; j += 3)
		{        // triangle per triangle

			auto getVertexIndex = [&](std::size_t index) {
				uint32_t packed_vertex_index = meshlet_triangles[index / 3 + meshlet.triangle_offset];
				uint8_t  vertex_index        = (packed_vertex_index >> ((index % 3) * 8)) & 0xFF;
				// index % 3 == 0, 不右移；index % 3 == 1, 右移8位；index % 3 == 2, 右移16位；最后取低8位
				return meshlet_vertices[meshlet.vertex_offset + vertex_index];

				/*return primitive.meshlet_vertex_indices[primitive.meshlet_triangles[index + meshlet.triangle_offset] + meshlet.vertex_offset];*/
			};

			std::int64_t triangle[3] = {
				mergeVertexRemap[getVertexIndex(j + 0)],
				mergeVertexRemap[getVertexIndex(j + 1)],
				mergeVertexRemap[getVertexIndex(j + 2)],
			};

			// remove triangles which have collapsed on themselves due to vertex merge
			if (triangle[0] == triangle[1] && triangle[0] == triangle[2])
			{
				continue;
			}
			for (std::size_t vertex = 0; vertex < 3; vertex++)
			{
				const std::size_t vertexIndex = triangle[vertex];

				// 总体的index到group内index的映射
				auto [iter, bWasNew] = mesh2groupVertexRemap.try_emplace(vertexIndex);
				if (bWasNew)
				{
					iter->second = groupVertexBuffer.size();
					groupVertexBuffer.push_back(VertexWrapper(vertex_positions, vertexIndex).getPosition());
				}
				groupVertexIndices.push_back(iter->second);
			}
		}
	}

	// group vertex buffer和group index buffer已经准备好了

	if (groupVertexIndices.empty())
		return false;

	// create reverse mapping from group to mesh-wide vertex indices
	group2meshVertexRemap.resize(groupVertexBuffer.size());

	for (const auto &[meshIndex, groupIndex] : mesh2groupVertexRemap)
	{
		assert(groupIndex < group2meshVertexRemap.size());
		group2meshVertexRemap[groupIndex] = meshIndex;
	}

	// simplify this group
	const float threshold        = 0.75f;
	std::size_t targetIndexCount = groupVertexIndices.size() * threshold;
	// unsigned int options     = meshopt_SimplifyErrorAbsolute;
	unsigned int options = meshopt_SimplifyLockBorder;        // we want all group borders to be locked (because they are shared between groups)

	std::vector<uint32_t> simplifiedIndexBuffer;
	simplifiedIndexBuffer.resize(groupVertexIndices.size());
	float simplificationError = 0.f;

	std::size_t simplifiedIndexCount = meshopt_simplify(
		simplifiedIndexBuffer.data(),   // output
		groupVertexIndices.data(),
		groupVertexIndices.size(),      // index buffer
		&groupVertexBuffer[0].x,        // pointer to position data
		groupVertexBuffer.size(),
		sizeof(glm::vec3),
		targetIndexCount,
		targetError,
		options,
		&simplificationError
	);
	simplifiedIndexBuffer.resize(simplifiedIndexCount);

	// ===== Generate meshlets for this group
	// TODO: if cluster is not simplified, use it for next LOD
	if (simplifiedIndexCount > 0 && simplifiedIndexCount != groupVertexIndices.size())
	// 等于就相当于没简化，以后就不用对该group进行简化了
	{
		float localScale = meshopt_simplifyScale(&groupVertexBuffer[0].x, groupVertexBuffer.size(), sizeof(glm::vec3));
		//// TODO: numerical stability
		float meshSpaceError = simplificationError * localScale;
		float maxChildError  = 0.0f;

		glm::vec3 min{+INFINITY, +INFINITY, +INFINITY};
		glm::vec3 max{-INFINITY, -INFINITY, -INFINITY};

		// 把小buffer中的index映射回总体的index
		for (auto &index : simplifiedIndexBuffer)
		{
			const glm::vec3 vertexPos = glm::vec3{vertex_positions[3 * index], vertex_positions[3 * index + 1], vertex_positions[3 * index + 2]};
			min                       = glm::min(min, vertexPos);
			max                       = glm::max(max, vertexPos);
		}

		glm::vec4 simplifiedClusterBounds = glm::vec4((min + max) / 2.0f, glm::distance(min, max) / 2.0f);

		for (const auto &meshletIndex : group.meshlets)
		{
			const auto &previousMeshlet = previousLevelMeshlets[meshletIndex];
			// ensure parent(this) error >= child(members of group) error
			maxChildError = std::max(maxChildError, previousMeshlet.cluster_error);
		}

		meshSpaceError += maxChildError;
		for (const auto &meshletIndex : group.meshlets)
		{
			previousLevelMeshlets[meshletIndex].parent_error          = meshSpaceError;
			previousLevelMeshlets[meshletIndex].parent_bounding_sphere = simplifiedClusterBounds;
		}

		append_meshlets(meshlet_vertices, meshlet_triangles, meshlets, &groupVertexBuffer[0].x, groupVertexBuffer.size(), simplifiedIndexBuffer, meshSpaceError, group2meshVertexRemap);

		return true;
	}
	return false;
}

void generateClusterHierarchy(const MeshPrimitiveData &primitive, std::vector<uint32_t> &meshlet_vertices, std::vector<uint32_t> &meshlet_triangles, std::vector<Meshlet> &meshlets, uint32_t &vertices_offset_last_lod, uint32_t &triangles_offset_last_lod, uint32_t &meshlets_offset_last_lod)
{
	LOGI("Building lod...");

	Timer timer;
	timer.start();

	static int num = 0;
	num++;
	LOGI("SubMesh {}", num);

	// level 0
	// tell meshoptimizer to generate meshlets
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

	auto         &indexBuffer           = index_data_32;
	std::uint32_t uniqueGroupIndex      = 0;

	append_meshlets(meshlet_vertices, meshlet_triangles, meshlets, vertex_positions, primitive.vertex_count, indexBuffer, 0.0f);

	LOGI("LOD {}: {} meshlets, {} vertices, {} triangles", 0, meshlets.size(), meshlet_vertices.size(), meshlet_triangles.size());

	KDTree<VertexWrapper> kdtree;

	// level n+1
	int maxLOD = 0;

	// 把每个group用到的vertex放到一个小buffer里，然后用meshopt_simplify来简化这个group
	std::vector<uint8_t> groupVertexIndices;
	std::vector<VertexWrapper> groupVerticesPreWeld;

	std::size_t previousMeshletsStart = 0;
	std::size_t previousVertexIndicesStart = 0;
	std::size_t previousTrianglesStart = 0;

	for (int lod = 0; lod < maxLOD; ++lod)
	{
		Timer lod_timer;
		lod_timer.start();

		float tLod = lod / (float) maxLOD;

		// find out the meshlets of the LOD n
		std::span<Meshlet> previousLevelMeshlets = std::span{meshlets.data() + previousMeshletsStart, meshlets.size() - previousMeshletsStart};
		if (previousLevelMeshlets.size() <= 1)
		{
			break;        // we have reached the end
		}

		// 作为代替
		std::unordered_set<std::size_t> meshlet_vertex_indices;
		// 把同一个meshlet的所有vertex放到一起
		for (const auto &meshlet : previousLevelMeshlets)
		{
			auto getVertexIndex = [&](std::size_t index) {

				uint32_t packed_vertex_index = meshlet_triangles[index / 3 + meshlet.triangle_offset];
				uint8_t vertex_index        = (packed_vertex_index >> ((index % 3) * 8)) & 0xFF;
				// index % 3 == 0, 不右移；index % 3 == 1, 右移8位；index % 3 == 2, 右移16位；最后取低8位
				return meshlet_vertices[meshlet.vertex_offset + vertex_index];

				/*return primitive.meshlet_vertex_indices[primitive.meshlet_triangles[index + meshlet.triangle_offset] + meshlet.vertex_offset];*/
			};

			// for each triangle of the meshlet
			for (std::size_t i = 0; i < meshlet.triangle_count * 3; i++)
			{
				meshlet_vertex_indices.insert(getVertexIndex(i));
			}
		}
		groupVerticesPreWeld.clear();
		groupVerticesPreWeld.reserve(meshlet_vertex_indices.size());
		for (const std::size_t i : meshlet_vertex_indices)
		{
			groupVerticesPreWeld.push_back(VertexWrapper(vertex_positions, i));
		}

		std::span<const VertexWrapper> wrappedVertices = groupVerticesPreWeld;

		Timer kdtree_timer;
		kdtree_timer.start();

		kdtree.build(wrappedVertices);

		float simplifyScale = 30;
		const float maxDistance = (tLod * 0.1f + (1 - tLod) * 0.01f) * simplifyScale;
		const float maxUVDistance = tLod * 0.5f + (1 - tLod) * 1.0f / 256.0f;

		// 合并足够近的vertex
		std::vector<bool> boundary = findBoundaryVertices(primitive, meshlet_vertices, meshlet_triangles, previousLevelMeshlets);

		const std::vector<std::int64_t> mergeVertexRemap = mergeByDistance(primitive, boundary, groupVerticesPreWeld, maxDistance, maxUVDistance, kdtree);

		const std::vector<MeshletGroup> groups = groupMeshletsRemap(primitive, meshlet_vertices, meshlet_triangles, previousLevelMeshlets, mergeVertexRemap);

		auto kdtree_time = kdtree_timer.stop();
		LOGI("KDTree time: {} ms", kdtree_time);

		// ===== Simplify groups
		const std::size_t newMeshletStart       = meshlets.size();
		const std::size_t newVertexIndicesStart = meshlet_vertices.size();
		const std::size_t newTrianglesStart     = meshlet_triangles.size();

		float targetError = 0.9f * tLod + 0.05f * (1 - tLod);

		for (const auto &group : groups)
		{
			// meshlets vector is modified during the loop
			previousLevelMeshlets = std::span{meshlets.data() + previousMeshletsStart, meshlets.size() - previousMeshletsStart};

			bool isSimplified = simplifyGroup(primitive, meshlet_vertices, meshlet_triangles, meshlets, previousLevelMeshlets, group, mergeVertexRemap, targetError);

			//if (!isSimplified)
			//{
			//	meshlets.resize(newMeshletStart);
			//	meshlet_vertices.resize(newVertexIndicesStart);
			//	meshlet_triangles.resize(newTrianglesStart);
			//	break;
			//}
		}

		for (std::size_t i = newMeshletStart; i < meshlets.size(); i++)
		{
			meshlets[i].lod = lod + 1;
		}

		auto lod_time = lod_timer.stop();
		LOGI("Lod Time: {} ms", lod_time);

		if (newMeshletStart != meshlets.size())
		{
			// 此处meshlets个数相比于上一级LoD没变也是正常的，因为顶点数和三角数减少了
			LOGI("LOD {}: {} meshlets", lod + 1, meshlets.size() - newMeshletStart);
			previousMeshletsStart      = newMeshletStart;
			previousVertexIndicesStart = newVertexIndicesStart;
			previousTrianglesStart     = newTrianglesStart;
		}
		else
		{
			break;
		}
	}

	meshlets_offset_last_lod  = previousMeshletsStart;
	vertices_offset_last_lod  = previousVertexIndicesStart;
	triangles_offset_last_lod = previousTrianglesStart;

	auto elapsed_time = timer.stop();
	LOGI("Time spent building lod: {} seconds.", xihe::to_string(elapsed_time));


	//return primitive;
}
}        // namespace xihe::sg
