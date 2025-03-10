#include "gpu_lod_scene.h"

#include "meshoptimizer.h"

#include "common/timer.h"

#include "scene_graph/components/material.h"
#include "scene_graph/components/mesh.h"
#include "scene_graph/components/mesh_lod.h"
#include "scene_graph/node.h"
#include "scene_graph/scene.h"

namespace
{
glm::vec4 convert_to_vec4(const std::vector<uint8_t> &data, uint32_t offset, float padding = 1.0f)
{
	if (data.size() < offset + 3 * sizeof(float))
		throw std::runtime_error("Data size is too small for conversion to vec4.");

	float x, y, z;
	std::memcpy(&x, &data[offset], sizeof(float));
	std::memcpy(&y, &data[offset + sizeof(float)], sizeof(float));
	std::memcpy(&z, &data[offset + 2 * sizeof(float)], sizeof(float));

	return {x, y, z, padding};
}

glm::vec4 calculate_bounds(const float *vertex_positions, uint32_t vertex_count)
{
	if (vertex_count == 0)
	{
		return glm::vec4(0.0f); 
	}

	glm::vec3 min_point(FLT_MAX);
	glm::vec3 max_point(-FLT_MAX);

	for (uint32_t i = 0; i < vertex_count; ++i)
	{
		const float *vertex = vertex_positions + i * 3;

		min_point.x = std::min(min_point.x, vertex[0]);
		min_point.y = std::min(min_point.y, vertex[1]);
		min_point.z = std::min(min_point.z, vertex[2]);

		max_point.x = std::max(max_point.x, vertex[0]);
		max_point.y = std::max(max_point.y, vertex[1]);
		max_point.z = std::max(max_point.z, vertex[2]);
	}

	glm::vec3 center = (min_point + max_point) * 0.5f;

	float max_dist_sq = 0.0f;
	for (uint32_t i = 0; i < vertex_count; ++i)
	{
		const float *vertex = vertex_positions + i * 3;
		glm::vec3    pos(vertex[0], vertex[1], vertex[2]);

		float dist_sq = glm::length2(pos - center);
		max_dist_sq   = std::max(max_dist_sq, dist_sq);
	}
	return glm::vec4(center, std::sqrt(max_dist_sq));
}

}        // namespace

namespace xihe
{
MeshLoDData::MeshLoDData(const MeshPrimitiveData &primitive_data)
{
	auto pos_it    = primitive_data.attributes.find("position");
	auto normal_it = primitive_data.attributes.find("normal");
	auto uv_it     = primitive_data.attributes.find("texcoord_0");

	if (pos_it == primitive_data.attributes.end() || normal_it == primitive_data.attributes.end())
	{
		throw std::runtime_error("Position or Normal attribute not found.");
	}

	const VertexAttributeData &pos_attr    = pos_it->second;
	const VertexAttributeData &normal_attr = normal_it->second;
	
	const bool                 has_uv      = (uv_it != primitive_data.attributes.end());
	const VertexAttributeData *uv_attr_ptr = has_uv ? &uv_it->second : nullptr;

	if (pos_attr.stride == 0 || normal_attr.stride == 0)
	{
		throw std::runtime_error("Stride for position or normal attribute is zero.");
	}
	uint32_t vertex_count = primitive_data.vertex_count;

	vertices.reserve(vertex_count);

	for (size_t i = 0; i < vertex_count; i++)
	{
		uint32_t pos_offset    = i * pos_attr.stride;
		uint32_t normal_offset = i * normal_attr.stride;
		
		float u = 0.0f;
		float v = 0.0f;

		if (has_uv)
		{
			const VertexAttributeData &uv_attr   = *uv_attr_ptr;
			uint32_t                   uv_offset = i * uv_attr.stride;
			std::memcpy(&u, &uv_attr.data[uv_offset], sizeof(float));
			std::memcpy(&v, &uv_attr.data[uv_offset + sizeof(float)], sizeof(float));	
		}

		glm::vec4 pos    = convert_to_vec4(pos_attr.data, pos_offset, u);
		glm::vec4 normal = convert_to_vec4(normal_attr.data, normal_offset, v);
		vertices.push_back({pos, normal});
	}

	prepare_meshlets(primitive_data);
}

void MeshLoDData::prepare_meshlets(const MeshPrimitiveData &primitive_data)
{
	auto vertex_positions = reinterpret_cast<const float *>(primitive_data.attributes.at("position").data.data());
	bounds                = calculate_bounds(vertex_positions, primitive_data.vertex_count);

	xihe::sg::generateClusterHierarchy(primitive_data, vertices, triangles, meshlets);

	meshlet_count = meshlets.size();
}

GpuLoDScene::GpuLoDScene(backend::Device &device) :
    device_{device}
{
}

void GpuLoDScene::initialize(sg::Scene &scene)
{
	auto meshes = scene.get_components<sg::Mesh>();

	std::vector<MeshLoDDraw> mesh_draws;

	std::vector<glm::vec4> mesh_bounds;

	std::vector<MeshInstanceDraw> instance_draws;

	std::vector<PackedVertex> global_vertices;
	std::vector<uint32_t>     global_triangles;
	std::vector<Meshlet>      global_meshlets;

	Timer initialize_timer;
	initialize_timer.start();

	int num = 0;

	for (const auto &mesh : meshes)
	{
		num++;
		if (num != 19)
		{
			continue;
		}
		for (const auto &submesh_data : mesh->get_submeshes_data())
		{
			Timer      submesh_timer;
			submesh_timer.start();

			auto      &primitive_data = submesh_data.primitive_data;
			const auto pbr_material   = dynamic_cast<const sg::PbrMaterial *>(submesh_data.material);

			MeshLoDDraw mesh_draw;
			mesh_draw.texture_indices                     = pbr_material->texture_indices;
			mesh_draw.base_color_factor                   = pbr_material->base_color_factor;
			mesh_draw.metallic_roughness_occlusion_factor = glm::vec4(pbr_material->metallic_factor, pbr_material->roughness_factor, 0.0f, 0.0f);

			mesh_draw.mesh_vertex_offset                  = static_cast<uint32_t>(global_vertices.size());
			mesh_draw.mesh_triangle_offset                = static_cast<uint32_t>(global_triangles.size());
			mesh_draw.meshlet_offset                      = static_cast<uint32_t>(global_meshlets.size());

			MeshLoDData mesh_data{primitive_data};

			global_vertices.insert(global_vertices.end(), mesh_data.vertices.begin(), mesh_data.vertices.end());

			global_triangles.insert(global_triangles.end(), mesh_data.triangles.begin(), mesh_data.triangles.end());

			std::ranges::for_each(mesh_data.meshlets, [mesh_draw_index = static_cast<uint32_t>(mesh_draws.size())](Meshlet &meshlet) { meshlet.mesh_draw_index = mesh_draw_index; });

			global_meshlets.insert(global_meshlets.end(), mesh_data.meshlets.begin(), mesh_data.meshlets.end());

			for (const auto &node : mesh->get_nodes())
			{
				MeshInstanceDraw instance_draw;
				auto node_transform = node->get_transform().get_world_matrix();
				instance_draw.model         = node_transform;
				instance_draw.model_inverse = glm::inverse(node_transform);
				instance_draw.mesh_draw_id  = static_cast<uint32_t>(mesh_draws.size());

				instance_draws.push_back(instance_draw);
			}

			mesh_draw.meshlet_count = static_cast<uint32_t>(mesh_data.meshlets.size());
			mesh_draws.push_back(mesh_draw);

			mesh_bounds.push_back(mesh_data.bounds);

			//auto submesh_time = submesh_timer.stop();
			//LOGI("Submesh time: {} s", submesh_time);
		}
	}

	auto initialize_time = initialize_timer.stop();
	LOGI("Initialize time: {} s", initialize_time);

	instance_count_ = static_cast<uint32_t>(instance_draws.size());

	{
		global_vertex_buffer_ = std::make_unique<backend::Buffer>(backend::Buffer::create_gpu_buffer(device_, global_vertices, vk::BufferUsageFlagBits::eStorageBuffer));
		global_vertex_buffer_->set_debug_name("global vertex buffer");

		LOGI("Global vertex buffer size: {} bytes", global_vertices.size() * sizeof(PackedVertex));
	}
	{
		global_meshlet_buffer_ = std::make_unique<backend::Buffer>(backend::Buffer::create_gpu_buffer(device_, global_meshlets, vk::BufferUsageFlagBits::eStorageBuffer));
		global_meshlet_buffer_->set_debug_name("global meshlet buffer");

		LOGI("Global meshlet buffer size: {} bytes", global_meshlets.size() * sizeof(Meshlet));
	}
	{
		global_triangle_buffer_ = std::make_unique<backend::Buffer>(backend::Buffer::create_gpu_buffer(device_, global_triangles, vk::BufferUsageFlagBits::eStorageBuffer));
		global_triangle_buffer_->set_debug_name("global triangle buffer");

		LOGI("Global packed meshlet indices buffer size: {} bytes", global_triangles.size() * sizeof(uint32_t));
	}
	{
		mesh_draws_buffer_ = std::make_unique<backend::Buffer>(backend::Buffer::create_gpu_buffer(device_, mesh_draws, vk::BufferUsageFlagBits::eStorageBuffer));
		mesh_draws_buffer_->set_debug_name("mesh draws buffer");

		LOGI("Mesh draws buffer size: {} bytes", mesh_draws.size() * sizeof(MeshDraw));
	}
	{
		assert(mesh_bounds.size() == mesh_draws.size());

		mesh_bounds_buffer_ = std::make_unique<backend::Buffer>(backend::Buffer::create_gpu_buffer(device_, mesh_bounds, vk::BufferUsageFlagBits::eStorageBuffer));
		mesh_bounds_buffer_->set_debug_name("mesh bounds buffer");

		LOGI("Mesh bounds buffer size: {} bytes", mesh_bounds.size() * sizeof(glm::vec4));
	}
	{
		backend::BufferBuilder buffer_builder{sizeof(uint32_t)};
		buffer_builder.with_usage(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer)
		    .with_vma_usage(VMA_MEMORY_USAGE_CPU_TO_GPU);
		draw_counts_buffer_ = std::make_unique<backend::Buffer>(device_, buffer_builder);
		draw_counts_buffer_->set_debug_name("draw counts buffer");
		draw_counts_buffer_->update(std::vector<uint32_t>{0});
	}
	{
		instance_buffer_ = std::make_unique<backend::Buffer>(backend::Buffer::create_gpu_buffer(device_, instance_draws, vk::BufferUsageFlagBits::eStorageBuffer));
		instance_buffer_->set_debug_name("instance buffer");
		
		LOGI("Instance buffer size: {} bytes", instance_draws.size() * sizeof(MeshInstanceDraw));
	}
	{
		draw_command_buffer_ = std::make_unique<backend::Buffer>(backend::Buffer::create_gpu_buffer(device_, std::vector<MeshDrawCommand>(instance_draws.size()), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer));
		draw_command_buffer_->set_debug_name("draw command buffer");

		LOGI("Draw command buffer size: {} bytes", instance_draws.size() * sizeof(MeshDrawCommand));
	}
}

backend::Buffer &GpuLoDScene::get_instance_buffer() const
{
	if (!instance_buffer_)
	{
		throw std::runtime_error("Instance buffer is not initialized.");
	}
	return *instance_buffer_;
}

backend::Buffer &GpuLoDScene::get_mesh_draws_buffer() const
{
	if (!mesh_draws_buffer_)
	{
		throw std::runtime_error("Mesh draws buffer is not initialized.");
	}
	return *mesh_draws_buffer_;
}

backend::Buffer &GpuLoDScene::get_mesh_bounds_buffer() const
{
	return *mesh_bounds_buffer_;
}

backend::Buffer &GpuLoDScene::get_draw_command_buffer() const
{
	if (!draw_command_buffer_)
	{
		throw std::runtime_error("Draw command buffer is not initialized.");
	}
	return *draw_command_buffer_;
}

backend::Buffer &GpuLoDScene::get_draw_counts_buffer() const
{
	if (!draw_counts_buffer_)
	{
		throw std::runtime_error("Draw counts buffer is not initialized.");
	}
	return *draw_counts_buffer_;
}

backend::Buffer &GpuLoDScene::get_global_vertex_buffer() const
{
	if (!global_vertex_buffer_)
	{
		throw std::runtime_error("Global vertex buffer is not initialized.");
	}
	return *global_vertex_buffer_;
}

backend::Buffer &GpuLoDScene::get_global_meshlet_buffer() const
{
	if (!global_meshlet_buffer_)
	{
		throw std::runtime_error("Global meshlet buffer is not initialized.");
	}
	return *global_meshlet_buffer_;
}

backend::Buffer &GpuLoDScene::get_global_triangle_buffer() const
{
	if (!global_triangle_buffer_)
	{
		throw std::runtime_error("Global packed meshlet indices buffer is not initialized.");
	}
	return *global_triangle_buffer_;
}

uint32_t GpuLoDScene::get_instance_count() const
{
	if (instance_count_ == 0)
	{
		throw std::runtime_error("Instance count is not initialized.");
	}
	return instance_count_;
}

backend::Device &GpuLoDScene::get_device() const
{
	return device_;
}
}        // namespace xihe
