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
static std::vector<MeshletGroup> groupMeshlets(MeshPrimitiveData &primitive, std::span<Meshlet> meshlets)
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
            return primitive.meshletVertexIndices[primitive.meshletIndices[index + meshlet.triangle_offset] + meshlet.vertex_offset];
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

	idx_t vertexCount = meshlets.size();            // vertex count, from the point of view of METIS, where Meshlet = vertex
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
	std::vector<idx_t> partition(vertexCount);

	// xadj
	std::vector<idx_t> xadjacency;
	xadjacency.reserve(vertexCount + 1);

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
		assert(vertexIndex <= vertexCount);
	}

	idx_t edgeCut;        // final cost of the cut found by METIS
	int   result = METIS_PartGraphKway(&vertexCount,
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
static std::vector<MeshletGroup> groupMeshletsRemap(MeshPrimitiveData &primitive, std::span<Meshlet> meshlets, std::span<const std::int64_t> vertexRemap)
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
            std::size_t vertexIndex = primitive.meshletVertexIndices[primitive.meshletIndices[index + meshlet.triangle_offset] + meshlet.vertex_offset];
            return static_cast<std::size_t>(vertexRemap[vertexIndex]);
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

	idx_t vertexCount = meshlets.size();        // vertex count, from the point of view of METIS, where Meshlet = vertex
	idx_t ncon        = 1;
	idx_t nparts      = meshlets.size() / 8;
	assert(nparts > 1);
	idx_t options[METIS_NOPTIONS];
	METIS_SetDefaultOptions(options);
	options[METIS_OPTION_OBJTYPE]   = METIS_OBJTYPE_CUT;
	options[METIS_OPTION_CCORDER]   = 1;        // identify connected components first
	options[METIS_OPTION_NUMBERING] = 0;

	std::vector<idx_t> partition(vertexCount);
	
	// xadj
	std::vector<idx_t> xadjacency;
	xadjacency.reserve(vertexCount + 1);

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
		assert(vertexIndex <= vertexCount);
	}

	idx_t edgeCut;        // final cost of the cut found by METIS
	int   result = METIS_PartGraphKway(&vertexCount,
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

// 从最新一级别的LOD重新拆分meshlet，附加到原来的primitive数组后面
//static void appendMeshlets(MeshPrimitiveData &primitive, std::span<std::uint32_t> indexBuffer, std::optional<MeshPrimitiveData> &maxLodPrimitive = std::nullopt) 
static void appendMeshlets(MeshPrimitiveData &primitive, std::span<std::uint32_t> indexBuffer, const glm::vec4 &clusterBounds, float clusterError)        //, MeshPrimitiveData *maxLodPrimitive = nullptr)
{
	constexpr std::size_t maxVertices  = 64;
	constexpr std::size_t maxTriangles = 124;
	const float           coneWeight   = 0.0f;        // for occlusion culling, currently unused

	const std::size_t            meshletOffset = primitive.meshlets.size();
	const std::size_t            vertexOffset  = primitive.meshletVertexIndices.size();
	const std::size_t            indexOffset   = primitive.meshletIndices.size();
	const std::size_t            maxMeshlets   = meshopt_buildMeshletsBound(indexBuffer.size(), maxVertices, maxTriangles);
	
	std::vector<meshopt_Meshlet> meshoptMeshlets(maxMeshlets);
	std::vector<unsigned int>  meshletVertexIndices(maxMeshlets * maxVertices);
	std::vector<unsigned char> meshletTriangles(maxMeshlets * maxTriangles * 3);

	auto vertex_positions = reinterpret_cast<const float *>(primitive.attributes.at("position").data.data());

	const std::size_t meshletCount = meshopt_buildMeshlets(
			meshoptMeshlets.data(), 
			meshletVertexIndices.data(), 
			meshletTriangles.data(),       // meshlet outputs
            indexBuffer.data(), 
			indexBuffer.size(),            // original index buffer
	        vertex_positions,              // pointer to position data
			primitive.vertex_count,        // vertex count of original mesh
	        sizeof(float) * 3,
	        maxVertices, 
		    maxTriangles,
		    coneWeight
	);

	const meshopt_Meshlet &last         = meshoptMeshlets[meshletCount - 1];
	const std::size_t      vertexCount  = last.vertex_offset + last.vertex_count;
	const std::size_t      indexCount   = last.triangle_offset + last.triangle_count * 3;
	primitive.meshletVertexIndices.resize(vertexOffset + vertexCount);
	primitive.meshletIndices.resize(indexOffset + indexCount);
	primitive.meshlets.resize(meshletOffset + meshletCount);        // remove over-allocated meshlets

    //for (std::size_t index = 0; index < vertexCount; ++index) {
    //    primitive.meshletVertexIndices[vertexOffset + index] = meshletVertexIndices[index];
    //}
	
    /*for (std::size_t index = 0; index < indexCount; ++index) {
        primitive.meshletIndices[indexOffset + index] = meshletTriangles[index];
    }*/

    /*for (std::size_t index = 0; index < meshletCount; ++index) {
        auto &meshoptMeshlet = meshoptMeshlets[index];
        auto &meshlet  = primitive.meshlets[meshletOffset + index];

        meshlet.vertex_offset      = vertexOffset + meshoptMeshlet.vertex_offset;
		meshlet.vertex_count	   = meshoptMeshlet.vertex_count;

        meshlet.triangle_offset    = indexOffset + meshoptMeshlet.triangle_offset;
		meshlet.triangle_count	   = meshoptMeshlet.triangle_count;

		meshopt_Bounds meshlet_bounds = meshopt_computeMeshletBounds(
		    meshletVertexIndices.data() + meshoptMeshlet.vertex_offset,
		    meshletTriangles.data() + meshoptMeshlet.triangle_offset,
		    meshoptMeshlet.triangle_count, vertex_positions, primitive.vertex_count, sizeof(float) * 3);

		meshlet.center = glm::vec3(meshlet_bounds.center[0], meshlet_bounds.center[1], meshlet_bounds.center[2]);
		meshlet.radius = meshlet_bounds.radius;

		meshlet.cone_axis   = glm::vec3(meshlet_bounds.cone_axis[0], meshlet_bounds.cone_axis[1], meshlet_bounds.cone_axis[2]);
		meshlet.cone_cutoff = meshlet_bounds.cone_cutoff;

		meshlet.cone_apex = glm::vec3(meshlet_bounds.cone_apex[0], meshlet_bounds.cone_apex[1], meshlet_bounds.cone_apex[2]);
    }*/

	// 改写成parallel for
	tbb::parallel_for(std::size_t(0), vertexCount, [&](std::size_t index) {
		primitive.meshletVertexIndices[vertexOffset + index] = meshletVertexIndices[index];
	});
	tbb::parallel_for(std::size_t(0), indexCount, [&](std::size_t index) {
		primitive.meshletIndices[indexOffset + index] = meshletTriangles[index];
	});
	tbb::parallel_for(std::size_t(0), meshletCount, [&](std::size_t index) {
		auto &meshoptMeshlet = meshoptMeshlets[index];
		auto &meshlet        = primitive.meshlets[meshletOffset + index];

		meshlet.vertex_offset = vertexOffset + meshoptMeshlet.vertex_offset;
		meshlet.vertex_count  = meshoptMeshlet.vertex_count;

		meshlet.triangle_offset = indexOffset + meshoptMeshlet.triangle_offset;
		meshlet.triangle_count  = meshoptMeshlet.triangle_count;

		meshopt_Bounds meshlet_bounds = meshopt_computeMeshletBounds(
		    meshletVertexIndices.data() + meshoptMeshlet.vertex_offset,
		    meshletTriangles.data() + meshoptMeshlet.triangle_offset,
		    meshoptMeshlet.triangle_count, vertex_positions, primitive.vertex_count, sizeof(float) * 3);

		meshlet.center = glm::vec3(meshlet_bounds.center[0], meshlet_bounds.center[1], meshlet_bounds.center[2]);
		meshlet.radius = meshlet_bounds.radius;

		meshlet.cone_axis   = glm::vec3(meshlet_bounds.cone_axis[0], meshlet_bounds.cone_axis[1], meshlet_bounds.cone_axis[2]);
		meshlet.cone_cutoff = meshlet_bounds.cone_cutoff;

		meshlet.cone_apex = glm::vec3(meshlet_bounds.cone_apex[0], meshlet_bounds.cone_apex[1], meshlet_bounds.cone_apex[2]);

		meshlet.clusterError = clusterError;
		meshlet.center       = glm::vec3(clusterBounds.x, clusterBounds.y, clusterBounds.z);
		meshlet.radius       = clusterBounds.w;

	});
}

/**
 * Find which vertices are part of meshlet boundaries. These should not be merged to avoid cracks between LOD levels
 */
static std::vector<bool> findBoundaryVertices(MeshPrimitiveData &primitive, std::span<Meshlet> meshlets)
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
		auto getVertexIndex = [&](std::size_t index) {
			return primitive.meshletVertexIndices[primitive.meshletIndices[index + meshlet.triangle_offset] + meshlet.vertex_offset];
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

std::vector<std::int64_t> mergeByDistance(const MeshPrimitiveData &primitive, const std::vector<bool> &boundary, std::span<const VertexWrapper> groupVerticesPreWeld, float maxDistance, const KDTree<VertexWrapper> &kdtree)
{
	// merge vertices which are close enough
	std::vector<std::int64_t> vertexRemap;

	const std::size_t vertexCount = primitive.vertex_count;
	vertexRemap.resize(vertexCount);

#ifndef USE_WELDING

	for (int i = 0; i < vertexCount; i++)
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

	for (std::int64_t v = 0; v < groupVerticesPreWeld.size(); v++)
	{
		std::int64_t replacement          = -1;
		const auto  &currentVertexWrapped = groupVerticesPreWeld[v];
		if (!boundary[currentVertexWrapped.index])
		{        // boundary vertices must not be merged with others (to avoid cracks)
			auto                 &neighbors     = neighborsForAllVertices[v];
			const glm::vec3 &currentVertexPos = currentVertexWrapped.getPosition(); 

			float maxDistanceSq   = maxDistance * maxDistance;

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
					replacement     = vertexRemap[groupVerticesPreWeld[neighbor].index];
					maxDistanceSq   = vertexDistanceSq;
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


void generateClusterHierarchy(MeshPrimitiveData &primitive)
{
	LOGI("Building lod...");

	Timer timer;
	timer.start();

	MeshPrimitiveData maxLodPrimitive;

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
	std::size_t   previousMeshletsStart = 0;
	std::uint32_t uniqueGroupIndex      = 0;
	{
		glm::vec3 min{+INFINITY, +INFINITY, +INFINITY};
		glm::vec3 max{-INFINITY, -INFINITY, -INFINITY};

		// remap simplified index buffer to mesh-wide vertex indices
		for (auto &index : indexBuffer)
		{
			const glm::vec3 vertexPos = glm::vec3{vertex_positions[3 * index], vertex_positions[3 * index + 1], vertex_positions[3 * index + 2]};
			min                       = glm::min(min, vertexPos);
			max                       = glm::max(max, vertexPos);
		}

		glm::vec4 simplifiedClusterBounds = glm::vec4((min + max) / 2.0f, glm::distance(min, max) / 2.0f);

		appendMeshlets(primitive, indexBuffer, simplifiedClusterBounds, 0.0f);
	}
	//appendMeshlets(primitive, index_data_32);

	KDTree<VertexWrapper> kdtree;


	// level n+1
	const int maxLOD = 25;        // I put a hard limit, but 25 might already be too high for some models

	// 把每个group用到的vertex放到一个小buffer里，然后用meshopt_simplify来简化这个group
	std::vector<uint8_t> groupVertexIndices;
	std::vector<VertexWrapper> groupVerticesPreWeld;


	for (int lod = 0; lod < maxLOD; ++lod)
	{
		float tLod = lod / (float) maxLOD;

		// find out the meshlets of the LOD n
		std::span<Meshlet> previousLevelMeshlets = std::span{primitive.meshlets.data() + previousMeshletsStart, primitive.meshlets.size() - previousMeshletsStart};
		if (previousLevelMeshlets.size() <= 1)
		{
			return;        // we have reached the end
		}

		// 作为代替
		std::unordered_set<std::size_t> meshletVertexIndices;
		// 把同一个meshlet的所有vertex放到一起
		for (const auto &meshlet : previousLevelMeshlets)
		{
			auto getVertexIndex = [&](std::size_t index) {
				return primitive.meshletVertexIndices[primitive.meshletIndices[index + meshlet.triangle_offset] + meshlet.vertex_offset];
			};

			// for each triangle of the meshlet
			for (std::size_t i = 0; i < meshlet.triangle_count * 3; i++)
			{
				meshletVertexIndices.insert(getVertexIndex(i));
			}
		}
		groupVerticesPreWeld.clear();
		groupVerticesPreWeld.reserve(meshletVertexIndices.size());
		for (const std::size_t i : meshletVertexIndices)
		{
			groupVerticesPreWeld.push_back(VertexWrapper(vertex_positions, i));
		}

		std::span<const VertexWrapper> wrappedVertices = groupVerticesPreWeld;
		kdtree.build(wrappedVertices);

		float simplifyScale = 10;
		const float maxDistance = (tLod * 0.1f + (1 - tLod) * 0.01f) * simplifyScale;

		// 合并足够近的vertex
		std::vector<bool> boundary = findBoundaryVertices(primitive, previousLevelMeshlets);

		const std::vector<std::int64_t> mergeVertexRemap = mergeByDistance(primitive, boundary, groupVerticesPreWeld, maxDistance, kdtree);

		const std::vector<MeshletGroup> groups = groupMeshletsRemap(primitive, previousLevelMeshlets, mergeVertexRemap);


		// ===== Simplify groups
		const std::size_t newMeshletStart = primitive.meshlets.size();
		for (const auto &group : groups)
		{
			// meshlets vector is modified during the loop
			previousLevelMeshlets = std::span{primitive.meshlets.data() + previousMeshletsStart, primitive.meshlets.size() - previousMeshletsStart};
			std::vector<uint32_t> groupVertexIndices;

			std::vector<glm::vec3> groupVertexBuffer;

			std::vector<std::size_t> group2meshVertexRemap;
			std::unordered_map<std::size_t, std::size_t> mesh2groupVertexRemap;


			// add cluster vertices to this group
			for (const auto &meshletIndex : group.meshlets)
			{
				const auto &meshlet = previousLevelMeshlets[meshletIndex];
				std::size_t start   = groupVertexIndices.size();
				groupVertexIndices.reserve(start + meshlet.triangle_count * 3);
				for (std::size_t j = 0; j < meshlet.triangle_count * 3; j += 3)
				{        // triangle per triangle
					std::int64_t triangle[3] = {
					    mergeVertexRemap[primitive.meshletVertexIndices[primitive.meshletIndices[meshlet.triangle_offset + j + 0] + meshlet.vertex_offset]],
					    mergeVertexRemap[primitive.meshletVertexIndices[primitive.meshletIndices[meshlet.triangle_offset + j + 1] + meshlet.vertex_offset]],
					    mergeVertexRemap[primitive.meshletVertexIndices[primitive.meshletIndices[meshlet.triangle_offset + j + 2] + meshlet.vertex_offset]],
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
							//groupVertexBuffer.push_back(glm::vec3(vertex_positions[iter->second], vertex_positions[iter->second + 1], vertex_positions[iter->second + 2]));
							groupVertexBuffer.push_back(VertexWrapper(vertex_positions, vertexIndex).getPosition());
							//groupVertexBuffer.push_back(VertexWrapper(vertex_positions, iter->second).getPosition());
						}
						groupVertexIndices.push_back(iter->second);
					}
				}
			}

			if (groupVertexIndices.empty())
				continue;

			// create reverse mapping from group to mesh-wide vertex indices
			group2meshVertexRemap.resize(groupVertexBuffer.size());

			for (const auto &[meshIndex, groupIndex] : mesh2groupVertexRemap)
			{
				assert(groupIndex < group2meshVertexRemap.size());
				group2meshVertexRemap[groupIndex] = meshIndex;
			}

			// simplify this group
			const float  threshold        = 0.5f;
			std::size_t  targetIndexCount = groupVertexIndices.size() * threshold;
			float        targetError      = 0.9f * tLod + 0.01f * (1 - tLod);
			//unsigned int options     = meshopt_SimplifyErrorAbsolute;
			unsigned int options     = meshopt_SimplifyLockBorder;        // we want all group borders to be locked (because they are shared between groups)

			std::vector<uint32_t> simplifiedIndexBuffer;
			simplifiedIndexBuffer.resize(groupVertexIndices.size());
			float simplificationError = 0.f;

			auto vertex_positions = reinterpret_cast<const float *>(primitive.attributes.at("position").data.data());

			std::size_t simplifiedIndexCount = meshopt_simplify(simplifiedIndexBuffer.data(),                                                           // output
			                                                    groupVertexIndices.data(), groupVertexIndices.size(),                                   // index buffer
			                                                    &groupVertexBuffer[0].x,                                        // pointer to position data
			                                                    groupVertexBuffer.size(),
			                                                    sizeof(glm::vec3),
			                                                    targetIndexCount, targetError, options, &simplificationError);
			simplifiedIndexBuffer.resize(simplifiedIndexCount);



			// ===== Generate meshlets for this group
			// TODO: if cluster is not simplified, use it for next LOD
			if (simplifiedIndexCount > 0 && simplifiedIndexCount != groupVertexIndices.size())
			{
				float localScale = meshopt_simplifyScale(&groupVertexBuffer[0].x, groupVertexBuffer.size(), sizeof(glm::vec3));
				//// TODO: numerical stability
				float meshSpaceError = simplificationError * localScale;
				float maxChildError    = 0.0f;

				glm::vec3 min{+INFINITY, +INFINITY, +INFINITY};
				glm::vec3 max{-INFINITY, -INFINITY, -INFINITY};

				// 把小buffer中的index映射回总体的index
				for (auto &index : simplifiedIndexBuffer)
				{
					index = group2meshVertexRemap[index];

					const glm::vec3 vertexPos = glm::vec3{vertex_positions[3 * index], vertex_positions[3 * index + 1], vertex_positions[3 * index + 2]};
					min                       = glm::min(min, vertexPos);
					max                       = glm::max(max, vertexPos);
				}

				glm::vec4 simplifiedClusterBounds = glm::vec4((min + max) / 2.0f, glm::distance(min, max) / 2.0f);

				for (const auto &meshletIndex : group.meshlets)
				{
					const auto &previousMeshlet = previousLevelMeshlets[meshletIndex];
					// ensure parent(this) error >= child(members of group) error
					maxChildError = std::max(maxChildError, previousMeshlet.clusterError);
				}

				meshSpaceError += maxChildError;
				for (const auto &meshletIndex : group.meshlets)
				{
					previousLevelMeshlets[meshletIndex].parentError          = meshSpaceError;
					previousLevelMeshlets[meshletIndex].parentBoundingSphere = simplifiedClusterBounds;
				}

				// group index is replaced on next iteration of loop (if there is one) to group the meshlets together based on partitionning
				//appendMeshlets(primitive, simplifiedIndexBuffer);
				appendMeshlets(primitive, simplifiedIndexBuffer,
				               simplifiedClusterBounds,        // use same group bounds for all meshlets
				               meshSpaceError                  // use same error for all meshlets
				);
			}
		}
		for (std::size_t i = newMeshletStart; i < primitive.meshlets.size(); i++)
		{
			primitive.meshlets[i].lod = lod + 1;
		}

		// 把primitive中所有与最后一级lod有关的信息拷贝到maxLodPrimitive中，并且要求从0开始编号
		/*if (lod == maxLOD - 1)
		{
			maxLodPrimitive.meshlets.resize(primitive.meshlets.size() - newMeshletStart);
			for (int i = 0; i < primitive.meshlets.size() - newMeshletStart; i++)
			{
				maxLodPrimitive.meshlets[i] = primitive.meshlets[i + newMeshletStart];
			}
			maxLodPrimitive.meshletVertexIndices
		}*/


		previousMeshletsStart = newMeshletStart;
	}

	auto elapsed_time = timer.stop();
	LOGI("Time spent building lod: {} seconds.", xihe::to_string(elapsed_time));


	//return primitive;
}
}        // namespace xihe::sg
