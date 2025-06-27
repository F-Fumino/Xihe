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

//// 序列化 std::vector<std::vector<uint32_t>>
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