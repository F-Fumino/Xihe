#include "buffer.h"

#include "backend/device.h"

namespace xihe::backend
{
Buffer BufferBuilder::build(Device &device) const
{
	return Buffer(device, *this);
}

BufferPtr BufferBuilder::build_unique(Device &device) const
{
	return std::make_unique<Buffer>(device, *this);
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

Buffer::Buffer(Buffer &&other) noexcept:
    Allocated{static_cast<Allocated &&>(other)},
    size_(std::exchange(other.size_, {}))
{}

Buffer::~Buffer()
{
	destroy_buffer(get_handle());
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
