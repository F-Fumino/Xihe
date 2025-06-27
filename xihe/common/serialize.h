#pragma once

#include <filesystem>
#include <glm/glm.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>

namespace cereal
{
template <class Archive>
void serialize(Archive &archive, glm::vec2 &v)
{
	archive(v.x, v.y);
}

template <class Archive>
void serialize(Archive &archive, glm::uvec2 &v)
{
	archive(v.x, v.y);
}

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

//// ���л� std::vector<std::vector<uint32_t>>
//template <class Archive>
//void serialize(Archive &archive, std::vector<std::vector<uint32_t>> &data)
//{
//	for (auto &vec : data)
//	{
//		archive(cereal::make_size_tag(static_cast<cereal::size_type>(vec.size())));
//		for (auto &elem : vec)
//		{
//			archive(elem);
//		}
//	}
//}

}        // namespace cereal