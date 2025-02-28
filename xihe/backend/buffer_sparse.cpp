#include "buffer_sparse.h"

namespace xihe::backend
{
SparseBuffer SparseBufferBuilder::build(Device &device) const
{
	return SparseBuffer(device, *this);
}

SparseBufferPtr SparseBufferBuilder::build_unique(Device &device) const
{
	return std::make_unique<SparseBuffer>(device, *this);
}

SparseBuffer::SparseBuffer(Device &device, SparseBufferBuilder const &builder) :
    Parent{builder.allocation_create_info, nullptr, &device}, size_{builder.create_info.size}
{
	get_handle() = create_buffer(builder.create_info);
	if (!builder.debug_name.empty())
	{
		set_debug_name(builder.debug_name);
	}
}

SparseBuffer::SparseBuffer(SparseBuffer &&other) noexcept :
    SparseAllocated{static_cast<SparseAllocated &&>(other)}, size_(std::exchange(other.size_, {}))
{}

SparseBuffer::~SparseBuffer()
{
	destroy_buffer(get_handle());
}

}        // namespace xihe::backend