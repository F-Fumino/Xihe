#include "allocated.h"

#include "backend/device.h"
#include "common/error.h"

namespace xihe::backend::allocated
{
VmaAllocator &get_memory_allocator()
{
	static VmaAllocator memory_allocator = VK_NULL_HANDLE;
	return memory_allocator;
}

void shutdown()
{
	auto &allocator = get_memory_allocator();
	if (allocator != VK_NULL_HANDLE)
	{
		VmaTotalStatistics stats;
		vmaCalculateStatistics(allocator, &stats);
		LOGI("Total device memory leaked: {} bytes.", stats.total.statistics.allocationBytes);
		vmaDestroyAllocator(allocator);
		allocator = VK_NULL_HANDLE;
	}
}

AllocatedBase::AllocatedBase(const VmaAllocationCreateInfo &alloc_create_info) :
	alloc_create_info_(alloc_create_info)
{}

AllocatedBase::AllocatedBase(AllocatedBase &&other) noexcept :
    alloc_create_info_(std::exchange(other.alloc_create_info_, {})),
    allocation_(std::exchange(other.allocation_, {})),
    mapped_data_(std::exchange(other.mapped_data_, {})),
    coherent_(std::exchange(other.coherent_, {})),
    persistent_(std::exchange(other.persistent_, {})),
    allocations_(std::exchange(other.allocations_, {})),
    total_block_num_(std::exchange(other.total_block_num_, {})),
    block_size_(std::exchange(other.block_size_, {}))
{}


const uint8_t *AllocatedBase::get_data() const
{
	return mapped_data_;
}

vk::DeviceMemory AllocatedBase::get_memory() const
{
	VmaAllocationInfo alloc_info;
	vmaGetAllocationInfo(get_memory_allocator(), allocation_, &alloc_info);
	return alloc_info.deviceMemory;
}

void AllocatedBase::flush(vk::DeviceSize offset, vk::DeviceSize size)
{
	if (!coherent_)
	{
		vmaFlushAllocation(get_memory_allocator(), allocation_, offset, size);
	}
}

bool AllocatedBase::mapped() const
{
	return mapped_data_ != nullptr;
}

bool AllocatedBase::mapped(uint32_t block_num) const
{
	return sparse_data_[block_num] != nullptr;
}

uint8_t *AllocatedBase::map()
{
	if (!persistent_ && !mapped())
	{
		VK_CHECK(vmaMapMemory(get_memory_allocator(), allocation_, reinterpret_cast<void **>(&mapped_data_)));
		assert(mapped_data_);
	}
	return mapped_data_;
}

uint8_t *AllocatedBase::map(uint32_t block_num)
{
	if (!mapped(block_num))
	{
		VK_CHECK(vmaMapMemory(get_memory_allocator(), allocations_[block_num], reinterpret_cast<void **>(&sparse_data_[block_num])));
		assert(sparse_data_[block_num]);
	}
	return sparse_data_[block_num];
}

void AllocatedBase::unmap()
{
	if (!persistent_ && mapped())
	{
		vmaUnmapMemory(get_memory_allocator(), allocation_);
		mapped_data_ = nullptr;
	}
}

void AllocatedBase::unmap(uint32_t block_num)
{
	if (mapped(block_num))
	{
		vmaUnmapMemory(get_memory_allocator(), allocations_[block_num]);
		sparse_data_[block_num] = nullptr;
	}
}

size_t AllocatedBase::update(const uint8_t *data, size_t size, size_t offset)
{
	if (persistent_)
	{
		std::copy_n(data, size, mapped_data_ + offset);
		flush();
	}
	else
	{
		map();
		std::copy_n(data, size, mapped_data_ + offset);
		flush();
		unmap();
	}
	return size;
}

size_t AllocatedBase::update(const void *data, size_t size, size_t offset)
{
	return update(reinterpret_cast<const uint8_t *>(data), size, offset);
}

size_t AllocatedBase::update(uint32_t block_num, const void *data, size_t size, size_t offset)
{
	assert(block_num < total_block_num_);

	map(block_num);
	std::copy_n(static_cast<const uint8_t *>(data), size, sparse_data_[block_num] + offset);
	unmap(block_num);

	return size;
}

void AllocatedBase::post_create(VmaAllocationInfo const &allocation_info)
{
	VkMemoryPropertyFlags memory_properties;
	vmaGetAllocationMemoryProperties(get_memory_allocator(), allocation_, &memory_properties);

	coherent_    = (memory_properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	mapped_data_ = static_cast<uint8_t *>(allocation_info.pMappedData);
	persistent_  = mapped();
}

vk::Buffer AllocatedBase::create_buffer(vk::BufferCreateInfo const &create_info)
{
	VkBufferCreateInfo const &create_info_c = create_info.operator VkBufferCreateInfo const &();
	VkBuffer                  buffer;

	VmaAllocationInfo allocation_info{};

	VK_CHECK(vmaCreateBuffer(get_memory_allocator(), &create_info_c, &alloc_create_info_, &buffer, &allocation_, &allocation_info));

	post_create(allocation_info);
	return buffer;
}

vk::Buffer AllocatedBase::create_sparse_buffer(Device &device, vk::BufferCreateInfo const &create_info, uint32_t total_block_num, vk::DeviceSize block_size)
{
	VkBufferCreateInfo const &create_info_c = create_info.operator VkBufferCreateInfo const &();
	VkBuffer                  buffer;

	total_block_num_ = total_block_num;
	block_size_ = block_size;
	allocations_.resize(total_block_num_);
	sparse_data_.resize(total_block_num_);

	assert(create_info_c.size == total_block_num_ * block_size_);

	VK_CHECK(vkCreateBuffer(static_cast<VkDevice>(device.get_handle()), &create_info_c, nullptr, &buffer));

	VkMemoryRequirements memory_requirements;
	vkGetBufferMemoryRequirements(static_cast<VkDevice>(device.get_handle()), buffer, &memory_requirements);
	memory_requirements.size = block_size_;
	//memory_requirements.memoryTypeBits = 0xFFFFFFFF;

	//VmaAllocationCreateInfo alloc_create_info{};
	//alloc_create_info.usage = VMA_MEMORY_USAGE_CPU_ONLY;

	//vmaAllocateMemoryPages(get_memory_allocator(), &memory_requirements, &alloc_create_info, total_block_num_, allocations_.data(), nullptr);
	VK_CHECK(vmaAllocateMemoryPages(get_memory_allocator(), &memory_requirements, &alloc_create_info_, total_block_num_, allocations_.data(), nullptr));

	return buffer;
}

vk::Image AllocatedBase::create_image(vk::ImageCreateInfo const &create_info)
{
	assert(0 < create_info.mipLevels && "Images should have at least one level");
	assert(0 < create_info.arrayLayers && "Images should have at least one layer");
	assert(create_info.usage && "Images should have at least one usage type");

	VkImageCreateInfo const &create_info_c = create_info.operator VkImageCreateInfo const &();
	VkImage                  image;

	VmaAllocationInfo allocation_info{};

	VK_CHECK(vmaCreateImage(get_memory_allocator(), &create_info_c, &alloc_create_info_, &image, &allocation_, &allocation_info));

	post_create(allocation_info);
	return image;
}

void AllocatedBase::destroy_buffer(Device *device, vk::Buffer buffer)
{
	if (buffer != VK_NULL_HANDLE && allocation_ != VK_NULL_HANDLE)
	{
		unmap();
		vmaDestroyBuffer(get_memory_allocator(), buffer.operator VkBuffer(), allocation_);
		clear();
	}
	else if (device != nullptr && buffer != VK_NULL_HANDLE && !allocations_.empty())
	{
		for (auto &allocation : allocations_)
		{
			if (allocation != VK_NULL_HANDLE)
			{
				vmaFreeMemory(get_memory_allocator(), allocation);
			}
		}
		vkDestroyBuffer(static_cast<VkDevice>(device->get_handle()), buffer, nullptr);
	}
}

void AllocatedBase::destroy_image(vk::Image image)
{
	if (image != VK_NULL_HANDLE && allocation_ != VK_NULL_HANDLE)
	{
		unmap();
		vmaDestroyImage(get_memory_allocator(), image.operator VkImage(), allocation_);
		clear();
	}
}

void AllocatedBase::clear()
{
	mapped_data_       = nullptr;
	persistent_        = false;
	alloc_create_info_ = {};
}

void init(const VmaAllocatorCreateInfo &create_info)
{
	VkResult result = vmaCreateAllocator(&create_info, &get_memory_allocator());
	if (result != VK_SUCCESS)
	{
		throw VulkanException{result, "Cannot create allocator"};
	}
}

void init(const backend::Device &device)
{
	VmaVulkanFunctions vma_vulkan_func{};
	vma_vulkan_func.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
	vma_vulkan_func.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo allocator_info{};
	allocator_info.pVulkanFunctions = &vma_vulkan_func;
	allocator_info.physicalDevice   = static_cast<VkPhysicalDevice>(device.get_gpu().get_handle());
	allocator_info.device           = static_cast<VkDevice>(device.get_handle());
	allocator_info.instance         = static_cast<VkInstance>(device.get_gpu().get_instance().get_handle());
	//allocator_info.preferredLargeHeapBlockSize = 128ull * 1024 * 1024;

	bool can_get_memory_requirements = device.is_extension_supported(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
	bool has_dedicated_allocation    = device.is_extension_supported(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
	if (can_get_memory_requirements && has_dedicated_allocation)
	{
		allocator_info.flags |= VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT;
	}

	if (device.is_extension_supported(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) && device.is_enabled(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME))
	{
		allocator_info.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	}

	if (device.is_extension_supported(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) && device.is_enabled(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME))
	{
		allocator_info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	}

	if (device.is_extension_supported(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME) && device.is_enabled(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME))
	{
		allocator_info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;
	}

	if (device.is_extension_supported(VK_KHR_BIND_MEMORY_2_EXTENSION_NAME) && device.is_enabled(VK_KHR_BIND_MEMORY_2_EXTENSION_NAME))
	{
		allocator_info.flags |= VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT;
	}

	if (device.is_extension_supported(VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME) && device.is_enabled(VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME))
	{
		allocator_info.flags |= VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT;
	}

	init(allocator_info);
}
}        // namespace xihe::backend::allocated
