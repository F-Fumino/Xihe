#pragma once

#include "metis.h"
#include "scene_graph/geometry_data.h"

namespace xihe::sg
{
void generate_lod(const MeshPrimitiveData &primitive, std::vector<uint32_t> &scene_data, std::vector<ClusterGroup> &cluster_groups, std::vector<Cluster> &clusters);
}        // namespace xihe::sg