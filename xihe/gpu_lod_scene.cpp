#include "gpu_lod_scene.h"

#include "common/serialize.h"
#include "common/timer.h"

#include "platform/filesystem.h"

#include "scene_graph/components/material.h"
#include "scene_graph/components/mesh.h"
#include "scene_graph/components/mesh_lod.h"
#include "scene_graph/node.h"
#include "scene_graph/scene.h"

//#define USE_SERIALIZE

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

	xihe::sg::generate_cluster_hierarchy(primitive_data, vertices, triangles, meshlets);

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

	bool exist_scene = false;

#ifdef USE_SERIALIZE
	fs::Path    scene_path = fs::path::get(fs::path::Type::kStorage) / scene.get_name() / ("gpu_lod_scene.bin");
	if (std::filesystem::exists(scene_path))
	{
		std::ifstream              is(scene_path, std::ios::binary);
		cereal::BinaryInputArchive archive(is);
		archive(global_vertices, global_triangles, global_meshlets, mesh_draws, mesh_bounds, instance_draws);
		exist_scene = true;
	}
#endif

	Timer initialize_timer;
	initialize_timer.start();

	int num = 0;

	for (const auto &mesh : meshes)
	{
		num++;
		if (exist_scene)
		{
			break;
		}
		/*if (num != 26)
		{
			continue;
		}*/
		//if (num == 5 || num == 12 || num == 15 || num == 18)           // 只有LOD 0
		//{
		//	continue;
		//}
		//if (num == 6 || num == 7 || num == 8 || num == 9 || num == 11) // 由于范围过大永远都是LOD 0
		//{
		//	continue;
		//}
		// num == 14 永远选不到LOD 0
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

			std::ranges::for_each(mesh_data.meshlets, 
				[
					mesh_draw_index = static_cast<uint32_t>(mesh_draws.size()), 
					global_vertex_offset = global_vertices.size(),
			        global_triangle_offset = global_triangles.size()
				]
				(Meshlet &meshlet)
			{
				meshlet.mesh_draw_index = mesh_draw_index;
				meshlet.vertex_page_index1 = (global_vertex_offset + meshlet.vertex_offset) * sizeof(PackedVertex) / PAGE_SIZE;
				meshlet.vertex_page_index2 = (global_vertex_offset + meshlet.vertex_offset + meshlet.vertex_count) * sizeof(PackedVertex) / PAGE_SIZE;
				meshlet.triangle_page_index1 = (global_triangle_offset + meshlet.triangle_offset) * sizeof(uint32_t) / PAGE_SIZE;
				meshlet.triangle_page_index2 = (global_triangle_offset + meshlet.triangle_offset + meshlet.triangle_count) * sizeof(uint32_t) / PAGE_SIZE;
			});

			//if (num == 281)
			//{
			//	LOGI("Vertex data from page {} to page {}", mesh_data.meshlets[0].vertex_page_index1, mesh_data.meshlets[mesh_data.meshlet_count - 1].vertex_page_index2);
			//	LOGI("Triangle data from page {} to page {}", mesh_data.meshlets[0].triangle_page_index1, mesh_data.meshlets[mesh_data.meshlet_count - 1].triangle_page_index2);
			//}

			global_vertices.insert(global_vertices.end(), mesh_data.vertices.begin(), mesh_data.vertices.end());

			global_triangles.insert(global_triangles.end(), mesh_data.triangles.begin(), mesh_data.triangles.end());

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
		}
	}

	if (!exist_scene)
	{
		fs::Path                    scene_path = fs::path::get(fs::path::Type::kStorage) / scene.get_name() / ("gpu_lod_scene.bin");

		fs::Path dir_path = scene_path.parent_path();
		if (!std::filesystem::exists(dir_path))
		{
			std::filesystem::create_directories(dir_path);
		}

		std::ofstream               os(scene_path, std::ios::binary);
		cereal::BinaryOutputArchive archive(os);
		archive(global_vertices, global_triangles, global_meshlets, mesh_draws, mesh_bounds, instance_draws);
	}

	LOGI("vertex data size: {}", global_vertices.size() * sizeof(PackedVertex));
	LOGI("triangle data size: {}", global_triangles.size() * sizeof(uint32_t));

	auto initialize_time = initialize_timer.stop();
	LOGI("Initialize time: {} s", initialize_time);

	instance_count_ = static_cast<uint32_t>(instance_draws.size());

	size_t   total_vertex_buffer_page_count = (global_vertices.size() * sizeof(PackedVertex) + PAGE_SIZE - 1) / PAGE_SIZE;
	size_t   vertex_table_page_count        = std::min(total_vertex_buffer_page_count, MAX_VERTEX_TABLE_PAGE);
	size_t   vertex_buffer_page_count       = std::min(total_vertex_buffer_page_count, MAX_BUFFER_PAGE);
	size_t   vertex_buffer_count            = (total_vertex_buffer_page_count + MAX_BUFFER_PAGE - 1) / MAX_BUFFER_PAGE;

	LOGI("Vertex Data Page: {}", total_vertex_buffer_page_count);
	LOGI("Vertex Table Page: {}", vertex_table_page_count);

	{
		vertex_page_table_ = std::make_unique<PageTable<PackedVertex>>(device_, vertex_table_page_count, PAGE_SIZE);
		vertex_page_table_->init(vertex_buffer_count, total_vertex_buffer_page_count);
		
		std::vector<uint64_t> vertex_buffer_addresses;

		for (size_t i = 0; i < vertex_buffer_count; i++)
		{
			uint32_t count = vertex_buffer_page_count;
			if (i == vertex_buffer_count - 1)
			{
				count = total_vertex_buffer_page_count - vertex_buffer_page_count * (vertex_buffer_count - 1);
			}
			backend::BufferBuilder buffer_builder{count * PAGE_SIZE};
			buffer_builder.with_usage(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eTransferDst).
				with_flags(vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency).
				with_vma_usage(VMA_MEMORY_USAGE_GPU_ONLY);

			vertex_page_table_->buffers_[i] = buffer_builder.build_unique(device_);

			vertex_buffer_addresses.push_back(vertex_page_table_->buffers_[i]->get_device_address());
		}

		vertex_page_table_->allocate_pages();

		vertex_buffer_address_ = std::make_unique<backend::Buffer>(backend::Buffer::create_gpu_buffer(device_, vertex_buffer_addresses, vk::BufferUsageFlagBits::eStorageBuffer));
		vertex_buffer_address_->set_debug_name("vertex buffer address");

		for (size_t i = 0; i < total_vertex_buffer_page_count; i++)
		{
			size_t start = i * PAGE_SIZE / sizeof(PackedVertex);
			size_t end   = std::min(start + PAGE_SIZE / sizeof(PackedVertex), global_vertices.size());
			vertex_page_table_->data_[i].assign(global_vertices.data() + start, global_vertices.data() + end);
		}

		size_t last = vertex_page_table_->data_[total_vertex_buffer_page_count - 1].size();
		vertex_page_table_->data_[total_vertex_buffer_page_count - 1].resize(PAGE_SIZE / sizeof(PackedVertex));
		for (size_t i = last; i < PAGE_SIZE / sizeof(PackedVertex); i++)
		{
			vertex_page_table_->data_[total_vertex_buffer_page_count - 1][i] = {{0, 0, 0, 0},
			                                                      {0, 0, 0, 0}};
		}

		LOGI("Global vertex buffer size: {} bytes", vertex_table_page_count * PAGE_SIZE);
	}
	{
		global_meshlet_buffer_ = std::make_unique<backend::Buffer>(backend::Buffer::create_gpu_buffer(device_, global_meshlets, vk::BufferUsageFlagBits::eStorageBuffer));
		global_meshlet_buffer_->set_debug_name("global meshlet buffer");

		LOGI("Global meshlet buffer size: {} bytes", global_meshlets.size() * sizeof(Meshlet));
	}

	size_t   total_triangle_buffer_page_count = (global_triangles.size() * sizeof(uint32_t) + PAGE_SIZE - 1) / PAGE_SIZE;
	size_t   triangle_table_page_count        = std::min(total_triangle_buffer_page_count, MAX_INDEX_TABLE_PAGE);
	size_t   triangle_buffer_page_count       = std::min(total_triangle_buffer_page_count, MAX_BUFFER_PAGE);
	size_t   triangle_buffer_count            = (total_triangle_buffer_page_count + MAX_BUFFER_PAGE - 1) / MAX_BUFFER_PAGE;

	{
		triangle_page_table_ = std::make_unique<PageTable<uint32_t>>(device_, triangle_table_page_count, PAGE_SIZE);
		triangle_page_table_->init(triangle_buffer_count, total_triangle_buffer_page_count);

		std::vector<uint64_t> triangle_buffer_addresses;

		for (size_t i = 0; i < triangle_buffer_count; i++)
		{
			backend::BufferBuilder buffer_builder{triangle_buffer_page_count * PAGE_SIZE};
			buffer_builder.with_usage(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eTransferDst).with_flags(vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency).with_vma_usage(VMA_MEMORY_USAGE_GPU_ONLY);

			triangle_page_table_->buffers_[i] = buffer_builder.build_unique(device_);

			triangle_buffer_addresses.push_back(triangle_page_table_->buffers_[i]->get_device_address());
		}

		triangle_page_table_->allocate_pages();

		triangle_buffer_address_ = std::make_unique<backend::Buffer>(backend::Buffer::create_gpu_buffer(device_, triangle_buffer_addresses, vk::BufferUsageFlagBits::eStorageBuffer));
		triangle_buffer_address_->set_debug_name("index buffer address");

		for (size_t i = 0; i < total_triangle_buffer_page_count; i++)
		{
			size_t start = i * PAGE_SIZE / sizeof(uint32_t);
			size_t end   = std::min(start + PAGE_SIZE / sizeof(uint32_t), global_triangles.size());
			triangle_page_table_->data_[i].assign(global_triangles.data() + start, global_triangles.data() + end);
		}

		size_t last = triangle_page_table_->data_[total_triangle_buffer_page_count - 1].size();
		triangle_page_table_->data_[total_triangle_buffer_page_count - 1].resize(PAGE_SIZE / sizeof(uint32_t));
		for (size_t i = last; i < PAGE_SIZE / sizeof(uint32_t); i++)
		{
			triangle_page_table_->data_[total_triangle_buffer_page_count - 1][i] = 0;
		}

		LOGI("Global index buffer size: {} bytes", triangle_table_page_count * PAGE_SIZE);
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
	{
		backend::BufferBuilder buffer_builder{total_vertex_buffer_page_count * sizeof(uint8_t)};
		//backend::BufferBuilder buffer_builder{total_vertex_buffer_page_count * sizeof(uint32_t)};
		buffer_builder.with_usage(vk::BufferUsageFlagBits::eStorageBuffer).with_vma_usage(VMA_MEMORY_USAGE_CPU_ONLY);

		vertex_page_state_buffer_ = buffer_builder.build_unique(device_);
		vertex_page_state_buffer_->set_debug_name("vertex page state buffer");
		memset(vertex_page_state_buffer_->map(), 0, vertex_page_state_buffer_->get_size());

		LOGI("Vertex page state buffer size: {} bytes", total_vertex_buffer_page_count * sizeof(uint8_t));
	}
	{
		backend::BufferBuilder buffer_builder{total_triangle_buffer_page_count * sizeof(uint8_t)};
		//backend::BufferBuilder buffer_builder{total_triangle_buffer_page_count * sizeof(uint32_t)};
		buffer_builder.with_usage(vk::BufferUsageFlagBits::eStorageBuffer).with_vma_usage(VMA_MEMORY_USAGE_CPU_ONLY);

		triangle_page_state_buffer_ = buffer_builder.build_unique(device_);
		triangle_page_state_buffer_->set_debug_name("triangle page state buffer");
		memset(triangle_page_state_buffer_->map(), 0, triangle_page_state_buffer_->get_size());

		LOGI("Triangle page state buffer size: {} bytes", total_triangle_buffer_page_count * sizeof(uint8_t));
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

backend::Buffer &GpuLoDScene::get_vertex_page_state_buffer() const
{
	if (!vertex_page_state_buffer_)
	{
		throw std::runtime_error("Vertex page state buffer is not initialized.");
	}
	return *vertex_page_state_buffer_;
}

backend::Buffer &GpuLoDScene::get_triangle_page_state_buffer() const
{
	if (!triangle_page_state_buffer_)
	{
		throw std::runtime_error("Triangle page state buffer is not initialized.");
	}
	return *triangle_page_state_buffer_;
}

backend::Buffer &GpuLoDScene::get_global_meshlet_buffer() const
{
	if (!global_meshlet_buffer_)
	{
		throw std::runtime_error("Global meshlet buffer is not initialized.");
	}
	return *global_meshlet_buffer_;
}

backend::Buffer &GpuLoDScene::get_vertex_buffer_address() const
{
	if (!vertex_buffer_address_)
	{
		throw std::runtime_error("Vertex buffer address is not initialized.");
	}
	return *vertex_buffer_address_;
}

backend::Buffer &GpuLoDScene::get_triangle_buffer_address() const
{
	if (!triangle_buffer_address_)
	{
		throw std::runtime_error("Triangle buffer address is not initialized.");
	}
	return *triangle_buffer_address_;
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

void GpuLoDScene::streaming(backend::CommandBuffer &command_buffer)
{
	//device_.get_fence_pool().wait();
	//device_.get_fence_pool().reset();
	//device_.wait_idle();
	// sparse bind
	uint8_t *vertex_state = reinterpret_cast<uint8_t *>(vertex_page_state_buffer_->map());
	//uint32_t *vertex_state = reinterpret_cast<uint32_t *>(vertex_page_state_buffer_->map());
	vertex_page_table_->execute(command_buffer, vertex_state);

	uint8_t *triangle_state = reinterpret_cast<uint8_t *>(triangle_page_state_buffer_->map());
	//uint32_t *triangle_state = reinterpret_cast<uint32_t *>(triangle_page_state_buffer_->map());
	triangle_page_table_->execute(command_buffer, triangle_state);
}
}        // namespace xihe
