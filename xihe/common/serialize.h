#pragma once

#include <filesystem>
#include <glm/glm.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>

namespace cereal
{

// ���л� glm::vec3
template <class Archive>
void serialize(Archive &archive, glm::vec3 &v)
{
	archive(v.x, v.y, v.z);
}

// ���л� glm::vec4
template <class Archive>
void serialize(Archive &archive, glm::vec4 &v)
{
	archive(v.x, v.y, v.z, v.w);
}

// ���л� glm::uvec3
template <class Archive>
void serialize(Archive &archive, glm::uvec3 &v)
{
	archive(v.x, v.y, v.z);
}

// ���л� glm::uvec4
template <class Archive>
void serialize(Archive &archive, glm::uvec4 &v)
{
	archive(v.x, v.y, v.z, v.w);
}

// ���л� glm::mat4
template <class Archive>
void serialize(Archive &archive, glm::mat4 &v)
{
	archive(v[0], v[1], v[2], v[3]);
}

}        // namespace cereal