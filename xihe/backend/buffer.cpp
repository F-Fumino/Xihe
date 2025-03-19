#include "buffer.h"

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include "backend/device.h"

namespace xihe::backend
{
Buffer BufferBuilder::build(Device &device) const
{
	return Buffer(device, *this);
}

Buffer BufferBuilder::build(Device &device, uint32_t page_num, vk::DeviceSize page_size) const
{
	return Buffer(device, *this, page_num, page_size);
}

BufferPtr BufferBuilder::build_unique(Device &device) const
{
	return std::make_unique<Buffer>(device, *this);
}

BufferPtr BufferBuilder::build_unique(Device &device, uint32_t page_num, vk::DeviceSize page_size) const
{
	return std::make_unique<Buffer>(device, *this, page_num, page_size);
}

Buffer Buffer::create_staging_buffer(Device &device, vk::DeviceSize size, const void *data)
{
	BufferBuilder builder{size};
	builder.with_usage(vk::BufferUsageFlagBits::eTransferSrc);
	builder.with_vma_flags(VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
	auto staging_buffer = builder.build(device);

	if (data != nullptr)
	{
		staging_buffer.update(static_cast<const uint8_t *>(data), size);
	}
	return staging_buffer;
}

Buffer Buffer::create_gpu_buffer(Device &device, vk::DeviceSize size, const void *data, vk::BufferUsageFlags usage)
{
	backend::CommandBuffer &command_buffer = device.request_command_buffer();

	command_buffer.begin(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

	Buffer staging_buffer = create_staging_buffer(device, size, data);

	BufferBuilder buffer_builder{size};
	buffer_builder.with_usage(usage | vk::BufferUsageFlagBits::eTransferDst).with_vma_usage(VMA_MEMORY_USAGE_GPU_ONLY);

	backend::Buffer buffer{device, buffer_builder};

	command_buffer.copy_buffer(staging_buffer, buffer, size);

	command_buffer.end();

	const auto &queue = device.get_queue_by_flags(vk::QueueFlagBits::eGraphics, 0);
	queue.submit(command_buffer, device.request_fence());

	device.get_fence_pool().wait();
	device.get_fence_pool().reset();
	device.get_command_pool().reset_pool();
	device.wait_idle();

	return buffer;
}

void Buffer::sparse_bind(Device &device, uint32_t page_index)
{
	assert(page_index < total_page_num_);

	// bind memory

	VkSparseMemoryBind bind = {
	    .resourceOffset = page_index * page_size_,
	    .size           = page_size_,
	    .memory         = get_memory(page_index),
	    .memoryOffset   = get_memory_offset(page_index),
	    .flags          = 0
	};

	VkSparseBufferMemoryBindInfo bind_info = {
	    .buffer    = get_handle(),
	    .bindCount = 1,
	    .pBinds    = &bind
	};

	VkBindSparseInfo sparse_bind_info = {
	    .sType           = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
	    .bufferBindCount = 1,
	    .pBufferBinds    = &bind_info
	};

	const auto &queue = device.get_queue_by_flags(vk::QueueFlagBits::eSparseBinding, 0);

	VK_CHECK(vkQueueBindSparse(queue.get_handle(), 1, &sparse_bind_info, device.request_fence()));

	device.get_fence_pool().wait();
	device.get_fence_pool().reset();
}

void Buffer::sparse_unbind(Device &device, uint32_t page_index)
{
	assert(page_index < total_page_num_);

	// bind memory

	VkSparseMemoryBind bind = {
	    .resourceOffset = page_index * page_size_,
	    .size           = page_size_,
	    .memory         = VK_NULL_HANDLE,
	    .memoryOffset   = 0,
	    .flags          = 0
	};

	VkSparseBufferMemoryBindInfo bind_info = {
	    .buffer    = get_handle(),
	    .bindCount = 1,
	    .pBinds    = &bind
	};

	VkBindSparseInfo sparse_bind_info = {
	    .sType           = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
	    .bufferBindCount = 1,
	    .pBufferBinds    = &bind_info
	};

	const auto &queue = device.get_queue_by_flags(vk::QueueFlagBits::eSparseBinding, 0);

	VK_CHECK(vkQueueBindSparse(queue.get_handle(), 1, &sparse_bind_info, device.request_fence()));

	device.get_fence_pool().wait();
	device.get_fence_pool().reset();
}

Buffer::Buffer(Device &device, BufferBuilder const &builder) :
    Parent{builder.allocation_create_info, nullptr, &device},
size_{builder.create_info.size}
{
	get_handle() = create_buffer(builder.create_info);

	if (!builder.debug_name.empty())
	{
		set_debug_name(builder.debug_name);
	}
}

Buffer::Buffer(Device &device, BufferBuilder const &builder, uint32_t page_num, vk::DeviceSize page_size) :
    Parent{builder.allocation_create_info, nullptr, &device},
    size_{page_num * page_size}
{
	assert(builder.create_info.flags & vk::BufferCreateFlagBits::eSparseBinding);

	get_handle() = create_sparse_buffer(device, builder.create_info, page_num, page_size);

	if (!builder.debug_name.empty())
	{
		set_debug_name(builder.debug_name);
	}
}

Buffer::Buffer(Buffer &&other) noexcept:
    Allocated{static_cast<Allocated &&>(other)},
    size_(std::exchange(other.size_, {}))
{}

Buffer::~Buffer()
{
	destroy_buffer(get_device_ptr(), get_handle());
}

uint64_t Buffer::get_device_address() const
{
	return get_device().get_handle().getBufferAddressKHR({get_handle()});
}

vk::DeviceSize Buffer::get_size() const
{
	return size_;
}
}
