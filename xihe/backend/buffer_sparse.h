#pragma once

#include "backend/allocated.h"
#include "backend/allocated_sparse.h"
#include "backend/vulkan_resource.h"

namespace xihe::backend
{
class Device;
class SparseBuffer;
using SparseBufferPtr = std::unique_ptr<SparseBuffer>;

struct SparseBufferBuilder : public allocated::Builder<SparseBufferBuilder, vk::BufferCreateInfo>
{
  private:
	using Parent = Builder<SparseBufferBuilder, vk::BufferCreateInfo>;

  public:
	SparseBufferBuilder(vk::DeviceSize size) :
	    Builder(vk::BufferCreateInfo({}, size))
	{
	}
	SparseBufferBuilder &with_usage(vk::BufferUsageFlags usage)
	{
		create_info.usage = usage;
		return *this;
	}
	SparseBufferBuilder &with_flags(vk::BufferCreateFlags flags)
	{
		create_info.flags = flags;
		return *this;
	}
	SparseBuffer    build(Device &device) const;
	SparseBufferPtr build_unique(Device &device) const;
};

class SparseBuffer : public allocated::SparseAllocated<vk::Buffer>
{
	using Parent = SparseAllocated<vk::Buffer>;

  public:
	SparseBuffer(Device &device, SparseBufferBuilder const &builder);

	SparseBuffer(const SparseBuffer &) = delete;
	SparseBuffer(SparseBuffer &&other) noexcept;
	~SparseBuffer() override;

	SparseBuffer &operator=(const SparseBuffer &) = delete;
	SparseBuffer &operator=(SparseBuffer &&)      = delete;

  private:
	vk::DeviceSize size_{0};
};

}