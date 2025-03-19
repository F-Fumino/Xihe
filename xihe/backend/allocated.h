#pragma once

#include <string>

#include <vulkan/vulkan.hpp>

#include "backend/vulkan_resource.h"
#include "common/error.h"

namespace xihe::backend
{
class Device;

namespace allocated
{
template <typename BuilderType,
          typename CreateInfoType>
struct Builder
{
	VmaAllocationCreateInfo allocation_create_info{};
	std::string             debug_name;
	CreateInfoType          create_info;

  protected:
	Builder(const Builder &other) = delete;
	Builder(const CreateInfoType &create_info) :
	    create_info(create_info)
	{
		allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO;
	}

  public:
	BuilderType &with_debug_name(const std::string &name)
	{
		debug_name = name;
		return static_cast<BuilderType &>(*this);
	}

	BuilderType &with_vma_usage(VmaMemoryUsage usage)
	{
		allocation_create_info.usage = usage;
		return static_cast<BuilderType &>(*this);
	}

	BuilderType &with_vma_flags(VmaAllocationCreateFlags flags)
	{
		allocation_create_info.flags = flags;
		return static_cast<BuilderType &>(*this);
	}

	BuilderType &with_vma_required_flags(const vk::MemoryPropertyFlags &flags)
	{
		allocation_create_info.requiredFlags = flags.operator VkMemoryPropertyFlags();
		return static_cast<BuilderType &>(*this);
	}
	BuilderType &with_vma_preferred_flags(const vk::MemoryPropertyFlags &flags)
	{
		allocation_create_info.preferredFlags = flags.operator VkMemoryPropertyFlags();
		return static_cast<BuilderType &>(*this);
	}
	BuilderType &with_vma_type_bits(uint32_t type_bits)
	{
		allocation_create_info.memoryTypeBits = type_bits;
		return static_cast<BuilderType &>(*this);
	}
	BuilderType &with_vma_pool(VmaPool pool)
	{
		allocation_create_info.pool = pool;
		return static_cast<BuilderType &>(*this);
	}
	BuilderType &with_queue_families(uint32_t count, const uint32_t *family_indices)
	{
		create_info.queueFamilyIndexCount = count;
		create_info.pQueueFamilyIndices   = family_indices;
		return static_cast<BuilderType &>(*this);
	}
	BuilderType &with_sharing_mode(vk::SharingMode mode)
	{
		create_info.sharingMode = mode;
		return static_cast<BuilderType &>(*this);
	}
	BuilderType &with_implicit_sharing_mode()
	{
		if (create_info.queueFamilyIndexCount == 0)
		{
			create_info.sharingMode = vk::SharingMode::eExclusive;
		}
		else
		{
			create_info.sharingMode = vk::SharingMode::eConcurrent;
		}
		return static_cast<BuilderType &>(*this);
	}
	BuilderType &with_queue_families(const std::vector<uint32_t> &queue_families)
	{
		return with_queue_families(static_cast<uint32_t>(queue_families.size()), queue_families.data());
	}
};

void init(const VmaAllocatorCreateInfo &create_info);

void init(const backend::Device &device);

VmaAllocator &get_memory_allocator();

void shutdown();

class AllocatedBase
{
  public:
	AllocatedBase() = default;
	AllocatedBase(const VmaAllocationCreateInfo &alloc_create_info);
	AllocatedBase(AllocatedBase &&other) noexcept;

	AllocatedBase &operator=(AllocatedBase &&other)      = delete;
	AllocatedBase &operator=(const AllocatedBase &other) = delete;


	const uint8_t   *get_data() const;
	vk::DeviceMemory get_memory() const;
	vk::DeviceMemory get_memory(uint32_t page_index) const;
	vk::DeviceSize   get_memory_offset(uint32_t page_index) const;
	/**
	 * @brief Flushes memory if it is HOST_VISIBLE and not HOST_COHERENT
	 */
	void flush(vk::DeviceSize offset = 0, vk::DeviceSize size = VK_WHOLE_SIZE);

	bool mapped() const;

	uint8_t *map();

	void unmap();

	size_t update(const uint8_t *data, size_t size, size_t offset = 0);

	size_t update(const void *data, size_t size, size_t offset = 0);

	template <typename T>
	size_t update(const std::vector<T> &data, size_t offset = 0)
	{
		return update(data.data(), data.size() * sizeof(T), offset);
	}

	template <typename T, size_t N>
	size_t update(std::array<T, N> &data, size_t offset = 0)
	{
		return update(data.data(), data.size() * sizeof(T), offset);
	}

	template <class T>
	size_t convert_and_update(const T &object, size_t offset = 0)
	{
		return update(reinterpret_cast<const uint8_t *>(&object), sizeof(T), offset);
	}

	size_t update(uint32_t page_num, const void *data, size_t size, size_t offset = 0);

	void allocate_page(uint32_t page_index);
	void free_page(uint32_t page_index);

  protected:
	virtual void             post_create(VmaAllocationInfo const &allocation_info);
	[[nodiscard]] vk::Buffer create_buffer(vk::BufferCreateInfo const &create_info);

	[[nodiscard]] vk::Buffer create_sparse_buffer(Device &device, vk::BufferCreateInfo const &create_info, uint32_t page_num, vk::DeviceSize page_size);

	[[nodiscard]] vk::Image  create_image(vk::ImageCreateInfo const &create_info);
	
	void                     destroy_buffer(Device *device, vk::Buffer buffer);
	void                     destroy_image(vk::Image image);
	void                     clear();

	VmaAllocationCreateInfo alloc_create_info_{};
	VmaAllocation           allocation_{VK_NULL_HANDLE};
	uint8_t                *mapped_data_{nullptr};
	bool                    coherent_{false};
	bool                    persistent_{false};        // Whether the buffer is persistently mapped or not

	// for sparse resources
	std::vector<VmaAllocation> allocations_{VK_NULL_HANDLE};
	VkMemoryRequirements       memory_requirements_{};
	uint32_t                   total_page_num_{0};
	vk::DeviceSize             page_size_{0};
	std::vector<uint8_t *>     sparse_data_{nullptr};
};

template <typename HandleType,
          typename ParentType = VulkanResource<HandleType>>
class Allocated : public ParentType, public AllocatedBase
{
  public:
	using ParentType::ParentType;

	Allocated()                  = delete;
	Allocated(const Allocated &) = delete;


	template <typename... Args>
	Allocated(const VmaAllocationCreateInfo &alloc_create_info, Args &&...args) :
	    backend::VulkanResource<HandleType>{std::forward<Args>(args)...},
	    AllocatedBase(alloc_create_info)
	{
	}

	Allocated(Allocated &&other) noexcept :
	    backend::VulkanResource<HandleType>{std::move(other)},
	    AllocatedBase(std::move(other))
	{
	}

	const HandleType &get() const
	{
		return backend::VulkanResource<HandleType>::get_handle();
	}
};
}        // namespace allocated

}        // namespace xihe::backend
