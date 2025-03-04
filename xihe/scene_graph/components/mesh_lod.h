#pragma once

#include "mshader_lod.h"

namespace xihe::sg
{
void generateClusterHierarchy(const MeshPrimitiveData &primitive, std::vector<PackedVertex> &vertices, std::vector<uint32_t> &triangles, std::vector<Meshlet> &meshlets);
}        // namespace xihe::sg