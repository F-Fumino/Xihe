#include "gpu_lod_scene.h"

#include "common/serialize.h"
#include "common/timer.h"

#include "platform/filesystem.h"

#include "scene_graph/components/material.h"
#include "scene_graph/components/mesh.h"
#include "scene_graph/components/vcg_lod.h"
#include "scene_graph/node.h"
#include "scene_graph/scene.h"

#define USE_SERIALIZE
#define MAX_LOD_THRESHOLD 8.0f

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
	prepare_meshlets(primitive_data);
}

void MeshLoDData::prepare_meshlets(const MeshPrimitiveData &primitive_data)
{
	auto vertex_positions = reinterpret_cast<const float *>(primitive_data.attributes.at("position").data.data());
	bounds                = calculate_bounds(vertex_positions, primitive_data.vertex_count);

	xihe::sg::generate_lod(primitive_data, scene_data, cluster_groups, clusters);
	/*xihe::sg::generate_cluster_hierarchy(primitive_data, scene_data, cluster_groups, clusters);*/

	meshlet_count = clusters.size();
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

	std::vector<ClusterGroup> global_cluster_groups;
	std::vector<Cluster>      global_clusters;

	bool exist_scene = false;
	uint32_t face_num    = 0;

	scene_data_page_table_ = std::make_unique<PageTable<uint32_t>>(device_, MAX_TABLE_PAGE, PAGE_SIZE);

#ifdef USE_SERIALIZE
	fs::Path    scene_path = fs::path::get(fs::path::Type::kStorage) / scene.get_name() / ("gpu_lod_scene.bin");
	if (std::filesystem::exists(scene_path))
	{
		std::ifstream              is(scene_path, std::ios::binary);
		cereal::BinaryInputArchive archive(is);

		archive(scene_data_page_table_->data_, global_cluster_groups, global_clusters, mesh_draws, mesh_bounds, instance_draws);
		exist_scene = true;
	}
#endif

	Timer initialize_timer;
	initialize_timer.start();

	int num = 0;
	size_t current_page_index = 0;
	size_t current_page_size  = PAGE_SIZE;
	// page 0
	scene_data_page_table_->data_.push_back(std::vector<uint32_t>());

	for (const auto &mesh : meshes)
	{
		num++;

		/*if (num != 3)
		{
			continue;
		}*/

		/*if (num != 6 && num != 8 && num != 10 && num != 12 && num != 23 && num != 24 && num != 32)
		{
			continue;
		}*/

		/*if (num != 35)
		{
			continue;
		}*/

		if (exist_scene)
		{
			break;
		}
		for (const auto &submesh_data : mesh->get_submeshes_data())
		{
			Timer      submesh_timer;
			submesh_timer.start();

			auto      &primitive_data = submesh_data.primitive_data;
			const auto pbr_material   = dynamic_cast<const sg::PbrMaterial *>(submesh_data.material);

			face_num += primitive_data.index_count / 3;

			MeshLoDDraw mesh_draw;
			mesh_draw.texture_indices                     = pbr_material->texture_indices;
			mesh_draw.base_color_factor                   = pbr_material->base_color_factor;
			mesh_draw.metallic_roughness_occlusion_factor = glm::vec4(pbr_material->metallic_factor, pbr_material->roughness_factor, 0.0f, 0.0f);

			mesh_draw.cluster_offset = static_cast<uint32_t>(global_clusters.size());

			MeshLoDData mesh_data{primitive_data};

			std::ranges::for_each(mesh_data.clusters, 
				[
					cluster_group_offset = static_cast<uint32_t>(global_cluster_groups.size()),
					mesh_draw_index = static_cast<uint32_t>(mesh_draws.size())
				]
				(Cluster &cluster)
			{
				cluster.mesh_draw_index = mesh_draw_index;
				cluster.cluster_group_index = cluster_group_offset + cluster.cluster_group_index;
			});

			for (auto& cluster_group : mesh_data.cluster_groups)
			{
				if (cluster_group.size * sizeof(uint32_t) > current_page_size)
				{
					scene_data_page_table_->data_[current_page_index].resize(PAGE_SIZE / sizeof(uint32_t), 0);
					scene_data_page_table_->data_.push_back(std::vector<uint32_t>());
					scene_data_page_table_->data_[++current_page_index].reserve(PAGE_SIZE / sizeof(uint32_t));
					cluster_group.page_index  = current_page_index;
					cluster_group.page_offset = 0;
					current_page_size         = PAGE_SIZE - cluster_group.size * sizeof(uint32_t);
				}
				else
				{
					cluster_group.page_index  = current_page_index;
					cluster_group.page_offset = (PAGE_SIZE - current_page_size) / sizeof(uint32_t);
					current_page_size         = current_page_size - cluster_group.size * sizeof(uint32_t);
				}
				scene_data_page_table_->data_[current_page_index].insert(scene_data_page_table_->data_[current_page_index].end(), mesh_data.scene_data.begin() + cluster_group.offset, mesh_data.scene_data.begin() + cluster_group.offset + cluster_group.size);
			}

			global_cluster_groups.insert(global_cluster_groups.end(), mesh_data.cluster_groups.begin(), mesh_data.cluster_groups.end());
			global_clusters.insert(global_clusters.end(), mesh_data.clusters.begin(), mesh_data.clusters.end());

			for (const auto &node : mesh->get_nodes())
			{
				MeshInstanceDraw instance_draw;
				auto         node_transform = node->get_transform().get_world_matrix();
				instance_draw.model         = node_transform;
				instance_draw.model_inverse = glm::inverse(node_transform);
				instance_draw.mesh_draw_id  = static_cast<uint32_t>(mesh_draws.size());

				instance_draws.push_back(instance_draw);
			}

			mesh_draw.cluster_count = static_cast<uint32_t>(mesh_data.clusters.size());
			mesh_draws.push_back(mesh_draw);

			mesh_bounds.push_back(mesh_data.bounds);
		}
	}

	LOGI("Total face count: {}", face_num);

	scene_data_page_table_->data_[current_page_index].resize(PAGE_SIZE / sizeof(uint32_t), 0);

	if (!exist_scene)
	{
		fs::Path scene_path = fs::path::get(fs::path::Type::kStorage) / scene.get_name() / ("gpu_lod_scene.bin");

		fs::Path dir_path = scene_path.parent_path();
		if (!std::filesystem::exists(dir_path))
		{
			std::filesystem::create_directories(dir_path);
		}

		std::ofstream               os(scene_path, std::ios::binary);
		cereal::BinaryOutputArchive archive(os);

		archive(scene_data_page_table_->data_, global_cluster_groups, global_clusters, mesh_draws, mesh_bounds, instance_draws);
	}

	instance_count_ = static_cast<uint32_t>(instance_draws.size());
	cluster_count_  = static_cast<uint32_t>(global_clusters.size());

	size_t sum_size = 0;

	size_t total_scene_data_buffer_page_count = scene_data_page_table_->data_.size();
	size_t scene_data_buffer_page_count       = std::min(total_scene_data_buffer_page_count, MAX_BUFFER_PAGE);
	size_t scene_data_buffer_count            = (total_scene_data_buffer_page_count + MAX_BUFFER_PAGE - 1) / MAX_BUFFER_PAGE;
	size_t table_page_count                   = std::min(total_scene_data_buffer_page_count, MAX_TABLE_PAGE);

	scene_data_page_table_->set_page_num(table_page_count);

	LOGI("Vertex Data Page: {}", total_scene_data_buffer_page_count);
	LOGI("Vertex Table Page: {}", table_page_count);

	{
		scene_data_page_table_->init(scene_data_buffer_count, total_scene_data_buffer_page_count);
		
		std::vector<uint64_t> vertex_buffer_addresses;

		for (size_t i = 0; i < scene_data_buffer_count; i++)
		{
			uint32_t count = scene_data_buffer_page_count;
			if (i == scene_data_buffer_count - 1)
			{
				count = total_scene_data_buffer_page_count - scene_data_buffer_page_count * (scene_data_buffer_count - 1);
			}
			backend::BufferBuilder buffer_builder{count * PAGE_SIZE};
			buffer_builder.with_usage(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eTransferDst).
				with_flags(vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency).
				with_vma_usage(VMA_MEMORY_USAGE_GPU_ONLY);

			scene_data_page_table_->buffers_[i] = buffer_builder.build_unique(device_);

			vertex_buffer_addresses.push_back(scene_data_page_table_->buffers_[i]->get_device_address());
		}

		scene_data_page_table_->allocate_pages();

		scene_data_buffer_address_ = std::make_unique<backend::Buffer>(backend::Buffer::create_gpu_buffer(device_, vertex_buffer_addresses, vk::BufferUsageFlagBits::eStorageBuffer));
		scene_data_buffer_address_->set_debug_name("vertex buffer address");

		sum_size += table_page_count * PAGE_SIZE;

		LOGI("Global scene data buffer size: {} bytes", table_page_count * PAGE_SIZE);
	}
	{
		cluster_group_buffer_ = std::make_unique<backend::Buffer>(backend::Buffer::create_gpu_buffer(device_, global_cluster_groups, vk::BufferUsageFlagBits::eStorageBuffer));
		cluster_group_buffer_->set_debug_name("cluster group buffer");

		sum_size += global_cluster_groups.size() * sizeof(ClusterGroup);

		LOGI("Global cluster group buffer size: {} bytes", global_cluster_groups.size() * sizeof(ClusterGroup));
	}
	{
		cluster_buffer_ = std::make_unique<backend::Buffer>(backend::Buffer::create_gpu_buffer(device_, global_clusters, vk::BufferUsageFlagBits::eStorageBuffer));
		cluster_buffer_->set_debug_name("cluster buffer");

		sum_size += global_clusters.size() * sizeof(Cluster);

		LOGI("Global cluster buffer size: {} bytes", global_clusters.size() * sizeof(Cluster));
	}
	{
		mesh_draws_buffer_ = std::make_unique<backend::Buffer>(backend::Buffer::create_gpu_buffer(device_, mesh_draws, vk::BufferUsageFlagBits::eStorageBuffer));
		mesh_draws_buffer_->set_debug_name("mesh draws buffer");

		sum_size += mesh_draws.size() * sizeof(MeshDraw);

		LOGI("Mesh draws buffer size: {} bytes", mesh_draws.size() * sizeof(MeshDraw));
	}
	{
		assert(mesh_bounds.size() == mesh_draws.size());

		mesh_bounds_buffer_ = std::make_unique<backend::Buffer>(backend::Buffer::create_gpu_buffer(device_, mesh_bounds, vk::BufferUsageFlagBits::eStorageBuffer));
		mesh_bounds_buffer_->set_debug_name("mesh bounds buffer");

		sum_size += mesh_bounds.size() * sizeof(glm::vec4);

		LOGI("Mesh bounds buffer size: {} bytes", mesh_bounds.size() * sizeof(glm::vec4));
	}
	{
		backend::BufferBuilder buffer_builder{sizeof(uint32_t)};
		buffer_builder.with_usage(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer)
		    .with_vma_usage(VMA_MEMORY_USAGE_CPU_TO_GPU);
		draw_counts_buffer_ = std::make_unique<backend::Buffer>(device_, buffer_builder);
		draw_counts_buffer_->set_debug_name("draw counts buffer");
		draw_counts_buffer_->update(std::vector<uint32_t>{0});

		sum_size += sizeof(uint32_t);
	}
	{
		instance_buffer_ = std::make_unique<backend::Buffer>(backend::Buffer::create_gpu_buffer(device_, instance_draws, vk::BufferUsageFlagBits::eStorageBuffer));
		instance_buffer_->set_debug_name("instance buffer");
		
		sum_size += instance_draws.size() * sizeof(MeshInstanceDraw);
		
		LOGI("Instance buffer size: {} bytes", instance_draws.size() * sizeof(MeshInstanceDraw));
	}
	{
		draw_command_buffer_ = std::make_unique<backend::Buffer>(backend::Buffer::create_gpu_buffer(device_, std::vector<MeshDrawCommand>(instance_draws.size()), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer));
		draw_command_buffer_->set_debug_name("draw command buffer");

		sum_size += instance_draws.size() * sizeof(MeshDrawCommand);

		LOGI("Draw command buffer size: {} bytes", instance_draws.size() * sizeof(MeshDrawCommand));
	}
	{
		backend::BufferBuilder buffer_builder{total_scene_data_buffer_page_count * sizeof(uint8_t)};
		//backend::BufferBuilder buffer_builder{total_vertex_buffer_page_count * sizeof(uint32_t)};
		buffer_builder.with_usage(vk::BufferUsageFlagBits::eStorageBuffer).with_vma_usage(VMA_MEMORY_USAGE_CPU_ONLY);

		page_state_buffer_ = buffer_builder.build_unique(device_);
		page_state_buffer_->set_debug_name("page state buffer");
		memset(page_state_buffer_->map(), 0, page_state_buffer_->get_size());

		sum_size += total_scene_data_buffer_page_count * sizeof(uint8_t);

		LOGI("Page state buffer size: {} bytes", total_scene_data_buffer_page_count * sizeof(uint8_t));
	}
	{
		backend::BufferBuilder buffer_builder{sizeof(uint32_t)};
		buffer_builder.with_usage(vk::BufferUsageFlagBits::eStorageBuffer)
		    .with_vma_usage(VMA_MEMORY_USAGE_CPU_ONLY);
		valid_data_size_buffer_ = buffer_builder.build_unique(device_);
		valid_data_size_buffer_->set_debug_name("valid data size buffer");
		memset(valid_data_size_buffer_->map(), 0, valid_data_size_buffer_->get_size());

		sum_size += sizeof(uint32_t);
	}
	{
		backend::BufferBuilder buffer_builder{sizeof(uint32_t)};
		buffer_builder.with_usage(vk::BufferUsageFlagBits::eStorageBuffer)
		    .with_vma_usage(VMA_MEMORY_USAGE_CPU_TO_GPU);
		recheck_counts_buffer_ = std::make_unique<backend::Buffer>(device_, buffer_builder);
		recheck_counts_buffer_->set_debug_name("recheck counts buffer");
		recheck_counts_buffer_->update(std::vector<uint32_t>{0});

		sum_size += sizeof(uint32_t);
	}
	{
		recheck_list_buffer_ = std::make_unique<backend::Buffer>(backend::Buffer::create_gpu_buffer(device_, std::vector<uint32_t>(global_clusters.size() * 2), vk::BufferUsageFlagBits::eStorageBuffer));
		recheck_list_buffer_->set_debug_name("recheck list buffer");

		sum_size += global_clusters.size() * 2 * sizeof(uint32_t);
	}
	{
		occlusion_command_buffer_ = std::make_unique<backend::Buffer>(backend::Buffer::create_gpu_buffer(device_, std::vector<OcclusionCommand>(global_clusters.size() / 4096 + 1), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer));
		occlusion_command_buffer_->set_debug_name("occlusion command buffer");

		sum_size += sizeof(OcclusionCommand);
	}
	{
		backend::BufferBuilder buffer_builder{sizeof(uint32_t)};
		buffer_builder.with_usage(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer)
		    .with_vma_usage(VMA_MEMORY_USAGE_CPU_TO_GPU);
		occlusion_counts_buffer_ = std::make_unique<backend::Buffer>(device_, buffer_builder);
		occlusion_counts_buffer_->set_debug_name("occlusion counts buffer");
		occlusion_counts_buffer_->update(std::vector<uint32_t>{0});

		sum_size += sizeof(uint32_t);
	}

	LOGI("Total gpu size: {} MB", double(sum_size) / 1024 / 1024);

	auto initialize_time = initialize_timer.stop();
	LOGI("Initialize time: {} s", initialize_time);
}

float GpuLoDScene::get_lod_threshold() const
{
	return lod_threshold_;
}

backend::Buffer &GpuLoDScene::get_scene_data_buffer_address() const
{
	if (!scene_data_buffer_address_)
	{
		throw std::runtime_error("Scene data buffer address is not initialized.");
	}
	return *scene_data_buffer_address_;
}

backend::Buffer &GpuLoDScene::get_cluster_group_buffer() const
{
	if (!cluster_group_buffer_)
	{
		throw std::runtime_error("Cluster group buffer is not initialized.");
	}
	return *cluster_group_buffer_;
}

backend::Buffer &GpuLoDScene::get_cluster_buffer() const
{
	if (!cluster_buffer_)
	{
		throw std::runtime_error("Cluster buffer is not initialized.");
	}
	return *cluster_buffer_;
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

backend::Buffer &GpuLoDScene::get_page_state_buffer() const
{
	if (!page_state_buffer_)
	{
		throw std::runtime_error("Page state buffer is not initialized.");
	}
	return *page_state_buffer_;
}

backend::Buffer &GpuLoDScene::get_valid_data_size_buffer() const
{
	if (!valid_data_size_buffer_)
	{
		throw std::runtime_error("Valid data size buffer is not initialized.");
	}
	return *valid_data_size_buffer_;
}

backend::Buffer &GpuLoDScene::get_recheck_counts_buffer() const
{
	if (!recheck_counts_buffer_)
	{
		throw std::runtime_error("Draw counts buffer is not initialized.");
	}
	return *recheck_counts_buffer_;
}

backend::Buffer &GpuLoDScene::get_recheck_list_buffer() const
{
	if (!recheck_list_buffer_)
	{
		throw std::runtime_error("Recheck list buffer is not initialized.");
	}
	return *recheck_list_buffer_;
}

backend::Buffer &GpuLoDScene::get_occlusion_command_buffer() const
{
	if (!occlusion_command_buffer_)
	{
		throw std::runtime_error("Occlusion command buffer is not initialized.");
	}
	return *occlusion_command_buffer_;
}

backend::Buffer &GpuLoDScene::get_occlusion_counts_buffer() const
{
	if (!occlusion_counts_buffer_)
	{
		throw std::runtime_error("Occlusion count buffer is not initialized.");
	}
	return *occlusion_counts_buffer_;
}

uint32_t GpuLoDScene::get_instance_count() const
{
	if (instance_count_ == 0)
	{
		throw std::runtime_error("Instance count is not initialized.");
	}
	return instance_count_;
}

uint32_t GpuLoDScene::get_cluster_count() const
{
	if (cluster_count_ == 0)
	{
		throw std::runtime_error("Instance count is not initialized.");
	}
	return cluster_count_;
}

double GpuLoDScene::get_page_table_time() const
{
	double time_avg = scene_data_page_table_->sum_page_table_time_ / scene_data_page_table_->frame_count_;
	return time_avg;
}

double GpuLoDScene::get_bind_time() const
{
	double time_avg = scene_data_page_table_->sum_bind_time_ / scene_data_page_table_->frame_count_;
	return time_avg;
}

double GpuLoDScene::get_page_table_hit_probability() const
{
	double probability_avg = 1.0 * scene_data_page_table_->sum_hit_ / scene_data_page_table_->sum_request_;
	return probability_avg;
}

double GpuLoDScene::get_memory_utilization() const
{
	double memory_utilization = sum_utilization_ / scene_data_page_table_->frame_count_;
	return memory_utilization;
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
	uint8_t *vertex_state = reinterpret_cast<uint8_t *>(page_state_buffer_->map());
	//uint32_t *vertex_state = reinterpret_cast<uint32_t *>(vertex_page_state_buffer_->map());
	PageTableState vertex_table_state = scene_data_page_table_->execute(command_buffer, vertex_state);

	const float max_lod_threshold = MAX_LOD_THRESHOLD;

	/*if (vertex_table_state == PageTableState::FULL || triangle_table_state == PageTableState::FULL)
	{
		if (lod_threshold_ < max_lod_threshold)
		{
			LOGW("LOD threshold changes from {} to {}", lod_threshold_, lod_threshold_ + 1);
			lod_threshold_ += 1;
		}
	}
	else if (vertex_table_state == PageTableState::EMPTY && triangle_table_state == PageTableState::EMPTY && lod_threshold_ >= 2)
	{
		LOGW("LOD threshold changes from {} to {}", lod_threshold_, lod_threshold_ - 1);
		lod_threshold_ += 1;
	}*/

	uint32_t *valid_data = reinterpret_cast<uint32_t *>(valid_data_size_buffer_->map());
	uint32_t  valid_data_size = valid_data[0];

	if (scene_data_page_table_->request_count_ == 0)
	{
		sum_utilization_ += 0;
	}
	else
	{
		sum_utilization_ += 1.0 * valid_data_size / (uint64_t(scene_data_page_table_->request_count_) * PAGE_SIZE);
	}

	memset(valid_data_size_buffer_->map(), 0, valid_data_size_buffer_->get_size());
}
}        // namespace xihe
