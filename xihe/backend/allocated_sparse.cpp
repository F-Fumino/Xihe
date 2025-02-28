#include "allocated_sparse.h"

#include "backend/device.h"
#include "common/error.h"

namespace xihe::backend::allocated
{
SparseAllocatedBase::SparseAllocatedBase(backend::Device &device, const VmaAllocationCreateInfo &alloc_create_info) : 
	device_(device),
    alloc_create_info_(alloc_create_info)
{}

SparseAllocatedBase::SparseAllocatedBase(SparseAllocatedBase &&other) noexcept : 
	device_(other.device_), 
    alloc_create_info_(std::exchange(other.alloc_create_info_, {})),
    allocations_(std::exchange(other.allocations_, {}))
{}

vk::Buffer SparseAllocatedBase::create_buffer(vk::BufferCreateInfo const &create_info, uint32_t block_num, uint32_t block_size)
{
	VkBufferCreateInfo const &create_info_c = create_info.operator VkBufferCreateInfo const &();
	VkBuffer                  buffer;

	assert(create_info_c.size == block_num * block_size);

	VmaAllocationInfo allocation_info{};

	VK_CHECK(vkCreateBuffer(static_cast<VkDevice>(device_.get_handle()), &create_info_c, nullptr, &buffer));

	VkMemoryRequirements memory_requirements;
	vkGetBufferMemoryRequirements(static_cast<VkDevice>(device_.get_handle()), buffer, &memory_requirements);
	memory_requirements.size = block_size;

	allocations_.resize(block_num);

	return buffer;
}

void SparseAllocatedBase::destroy_buffer(vk::Buffer buffer)
{
	if (buffer != VK_NULL_HANDLE)
	{
		for (auto &allocation : allocations_)
		{
			if (allocation != VK_NULL_HANDLE)
			{
				vmaFreeMemory(get_memory_allocator(), allocation);
			}
		}
		vkDestroyBuffer(static_cast<VkDevice>(device_.get_handle()), buffer, nullptr);
	}
}

void SparseAllocatedBase::swap_in(uint32_t block_num, void *data)
{

}


}
