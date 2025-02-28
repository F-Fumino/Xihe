#pragma once

#include <string>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

#include "backend/vulkan_resource.h"
#include "common/error.h"

namespace xihe::backend
{
class Device;

namespace allocated
{

class SparseAllocatedBase
{
  public:
	SparseAllocatedBase() = default;
	SparseAllocatedBase(backend::Device &device, const VmaAllocationCreateInfo &alloc_create_info);
	SparseAllocatedBase(SparseAllocatedBase &&other) noexcept;

	SparseAllocatedBase &operator=(SparseAllocatedBase &&other) = delete;
	SparseAllocatedBase &operator=(const SparseAllocatedBase &other) = delete;

  protected:
	[[nodiscard]] vk::Buffer create_buffer(vk::BufferCreateInfo const &create_info, uint32_t block_num, uint32_t block_size);
	void                     destroy_buffer(vk::Buffer buffer);

	void					 swap_in(uint32_t block_num, void *data);
	void					 swap_out(uint32_t block_num);

	backend::Device &device_;

	VmaAllocationCreateInfo alloc_create_info_{};
	std::vector<VmaAllocation> allocations_;
};

template <typename HandleType,
          typename ParentType = VulkanResource<HandleType>>
class SparseAllocated : public ParentType, public SparseAllocatedBase
{
  public:
	using ParentType::ParentType;

	SparseAllocated()                  = delete;
	SparseAllocated(const SparseAllocated &) = delete;

	template <typename... Args>
	SparseAllocated(const VmaAllocationCreateInfo &alloc_create_info, Args &&...args) :
	    backend::VulkanResource<HandleType>{std::forward<Args>(args)...},
	    SparseAllocatedBase(alloc_create_info)
	{
	}

	SparseAllocated(SparseAllocated &&other) noexcept :
	    backend::VulkanResource<HandleType>{std::move(other)},
	    SparseAllocatedBase(std::move(other))
	{
	}

	const HandleType &get() const
	{
		return backend::VulkanResource<HandleType>::get_handle();
	}
};

}        // namespace allocated

}        // namespace xihe::backend
