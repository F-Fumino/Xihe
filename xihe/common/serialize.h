#pragma once

#include <filesystem>
#include <glm/glm.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>

namespace cereal
{

// 序列化 glm::vec3
template <class Archive>
void serialize(Archive &archive, glm::vec3 &v)
{
	archive(v.x, v.y, v.z);
}

// 序列化 glm::vec4
template <class Archive>
void serialize(Archive &archive, glm::vec4 &v)
{
	archive(v.x, v.y, v.z, v.w);
}

// 序列化 glm::uvec3
template <class Archive>
void serialize(Archive &archive, glm::uvec3 &v)
{
	archive(v.x, v.y, v.z);
}

// 序列化 glm::uvec4
template <class Archive>
void serialize(Archive &archive, glm::uvec4 &v)
{
	archive(v.x, v.y, v.z, v.w);
}

// 序列化 glm::mat4
template <class Archive>
void serialize(Archive &archive, glm::mat4 &v)
{
	archive(v[0], v[1], v[2], v[3]);
}

}        // namespace cereal