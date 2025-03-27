#include "page_table.h"

#include "scene_graph/geometry_data.h"

namespace xihe
{
template class xihe::PageTable<PackedVertex>;
template class xihe::PageTable<uint32_t>;

template <typename DataType>
PageTable<DataType>::PageTable(backend::Device &device, uint32_t table_page_num, vk::DeviceSize page_size) :
    SparseResources(table_page_num, page_size),
    device_{device}
{
	sparse_queue_ = &device.get_queue_by_flags(vk::QueueFlagBits::eSparseBinding, vk::QueueFlagBits::eGraphics, 0);

	for (size_t i = 0; i < table_page_num; i++)
	{
		free_list_.push_back(i);
	}

	table_to_buffer_.resize(table_page_num);
}

template <typename DataType>
void PageTable<DataType>::init(uint32_t buffer_count, uint32_t buffer_page_count)
{
	buffer_page_count_ = buffer_page_count;
	buffer_count_      = buffer_count;

	buffers_.resize(buffer_count);

	data_.resize(buffer_page_count);
	staging_buffers_.resize(buffer_page_count);
	page_swapped_in_.resize(buffer_page_count);
	buffer_to_table_.resize(buffer_page_count);

	for (size_t i = 0; i < buffer_page_count; i++)
	{
		buffer_to_table_[i] = -1;
	}
}

template <typename DataType>
void PageTable<DataType>::allocate_pages()
{
	vkGetBufferMemoryRequirements(static_cast<VkDevice>(device_.get_handle()), buffers_[0]->get_handle(), &memory_requirements_);
	alloc_create_info_ = buffers_[0]->get_allocation_create_info();
	SparseResources::allocate_pages();
}

template <typename DataType>
void PageTable<DataType>::execute(backend::CommandBuffer &command_buffer, uint16_t *page_state)
{
	const uint32_t max_binds = 1000;
	uint32_t       num_binds = 0;

	std::vector<std::vector<VkSparseMemoryBind>> binds;
	binds.resize(buffer_count_);

	for (size_t i = 0; i < buffer_page_count_; i++)
	{
		// first 1 means the page is in the table, second 1 means the page is requested
		if (page_state[i] & 1)
		{
			//LOGI("Access buffer page: {}", i);
			int32_t table_page_index = swap_in(page_state, i);
			page_state[i]            = page_state[i] & ~1;        // clear page request

			if (table_page_index == -2)
			{
				continue; // hit
			}
			if (table_page_index == -1)
			{
				LOGW("Page Table is full");
				for (size_t j = i + 1; j < buffer_page_count_; j++)
				{
					page_state[j] = page_state[j] & ~1;
				}
				break;
			}

			if (++num_binds > max_binds)
			{
				break;
			}

			table_to_buffer_[table_page_index] = i;
			buffer_to_table_[i]                = table_page_index;

			page_state[i] |= 0b10;

			VkSparseMemoryBind bind = {
			    .resourceOffset = (i % MAX_BUFFER_PAGE) * PAGE_SIZE,
			    .size           = PAGE_SIZE,
			    .memory         = get_memory(table_page_index),
			    .memoryOffset   = get_memory_offset(table_page_index),
			    .flags          = 0};
			binds[i / MAX_BUFFER_PAGE].push_back(bind);

			if (staging_buffers_[i] == nullptr)
			{
				staging_buffers_[i] = std::make_unique<backend::Buffer>(backend::Buffer::create_staging_buffer(device_, PAGE_SIZE, data_[i].data()));
			}

			command_buffer.copy_buffer(*staging_buffers_[i], *buffers_[i / MAX_BUFFER_PAGE], PAGE_SIZE, 0, (i % MAX_BUFFER_PAGE) * PAGE_SIZE);
		}
	}

	std::vector<VkSparseBufferMemoryBindInfo> binds_info;

	for (size_t i = 0; i < buffer_count_; i++)
	{
		if (!binds[i].empty())
		{
			VkSparseBufferMemoryBindInfo bind_info = {
			    .buffer    = buffers_[i]->get_handle(),
			    .bindCount = static_cast<uint32_t>(binds[i].size()),
			    .pBinds    = binds[i].data()};

			binds_info.push_back(bind_info);
			/*VkBindSparseInfo sparse_bind_info = {
			    .sType           = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
			    .bufferBindCount = 1,
			    .pBufferBinds    = &bind_info
			};

			VK_CHECK(vkQueueBindSparse(sparse_queue_->get_handle(), 1, &sparse_bind_info, device_.request_fence()));

			device_.get_fence_pool().wait();
			device_.get_fence_pool().reset();*/
		}
	}

	if (!binds_info.empty())
	{
		VkBindSparseInfo sparse_bind_info = {
		    .sType           = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
		    .bufferBindCount = static_cast<uint32_t>(binds_info.size()),
		    .pBufferBinds    = binds_info.data()};

		// device_.wait_idle();

		VkResult result = vkQueueBindSparse(sparse_queue_->get_handle(), 1, &sparse_bind_info, device_.request_fence());

		// if (result != VK_SUCCESS)
		//{
		//	// Query number of available results
		//	VkDeviceFaultCountsEXT faultCounts{};
		//	faultCounts.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT;

		//	vkGetDeviceFaultInfoEXT(device_.get_handle(), &faultCounts, NULL);

		//	// Allocate output arrays and query fault data
		//	VkDeviceFaultInfoEXT faultInfo{};
		//	faultInfo.sType             = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT;
		//	faultInfo.pAddressInfos     = (VkDeviceFaultAddressInfoEXT *) malloc(sizeof(VkDeviceFaultAddressInfoEXT) * faultCounts.addressInfoCount);
		//	faultInfo.pVendorInfos      = (VkDeviceFaultVendorInfoEXT *) malloc(sizeof(VkDeviceFaultVendorInfoEXT) * faultCounts.vendorInfoCount);
		//	faultInfo.pVendorBinaryData = malloc(faultCounts.vendorBinarySize);

		//	vkGetDeviceFaultInfoEXT(device_.get_handle(), &faultCounts, &faultInfo);
		//
		//	for (uint32_t i = 0; i < faultCounts.addressInfoCount; i++)
		//	{
		//		VkDeviceFaultAddressInfoEXT &addressInfo = faultInfo.pAddressInfos[i];

		//		LOGE("Fault Address Type: {}", int(addressInfo.addressType));
		//		LOGE("Reported Address: {}", addressInfo.reportedAddress);
		//		LOGE("Address Precision: {}", addressInfo.addressPrecision);

		//		VkDeviceAddress lower_address = (addressInfo.reportedAddress & ~(addressInfo.addressPrecision - 1));
		//		VkDeviceAddress upper_address = (addressInfo.reportedAddress | (addressInfo.addressPrecision - 1));

		//		LOGE("Possible Fault Address Range: [{}, {}]", lower_address, upper_address);
		//	}

		//	for (uint32_t i = 0; i < faultCounts.vendorInfoCount; i++)
		//	{
		//		VkDeviceFaultVendorInfoEXT &vendorInfo = faultInfo.pVendorInfos[i];
		//		LOGE("Vendor Error Code: {}", vendorInfo.vendorFaultCode);
		//		LOGE("Vendor Description: {}", vendorInfo.description);
		//	}
		//}

		device_.get_fence_pool().wait();
		device_.get_fence_pool().reset();
		// device_.wait_idle();
	}

	// device_.wait_idle();

	// for (size_t i = 0; i < buffer_page_count_; i++)
	//{
	//	// first 1 means the page is in the table, second 1 means the page is requested
	//	if ((page_request[i] & 0b11) == 0b01)
	//	{
	//		uint16_t table_page_index = swap_in(page_request);
	//		page_request[i]           = page_request[i] & ~1;        // clear page request

	//		if (table_page_index == -1)
	//		{
	//			LOGW("Page Table is full");
	//			for (size_t j = i; j < buffer_page_count_; j++)
	//			{
	//				page_request[j] = page_request[j] & ~1;
	//			}
	//			break;
	//		}

	//		table_to_buffer_[table_page_index] = i;
	//		buffer_to_table_[i]                = table_page_index;

	//		LOGI("Swap in page: {}", i);

	//		page_request[i] |= 0b10;

	//		VkSparseMemoryBind bind = {
	//		    .resourceOffset = (i % MAX_BUFFER_PAGE) * PAGE_SIZE,
	//		    .size           = PAGE_SIZE,
	//		    .memory         = get_memory(table_page_index),
	//		    .memoryOffset   = get_memory_offset(table_page_index),
	//		    .flags          = 0
	//		};

	//		VkSparseBufferMemoryBindInfo bind_info = {
	//		    .buffer    = buffers_[i / MAX_BUFFER_PAGE]->get_handle(),
	//		    .bindCount = 1,
	//		    .pBinds    = &bind
	//		};

	//		VkBindSparseInfo sparse_bind_info = {
	//		    .sType           = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
	//		    .bufferBindCount = 1,
	//		    .pBufferBinds    = &bind_info
	//		};

	//		VK_CHECK(vkQueueBindSparse(sparse_queue_->get_handle(), 1, &sparse_bind_info, device_.request_fence()));

	//		device_.get_fence_pool().wait();
	//		device_.get_fence_pool().reset();

	//		if (staging_buffers_[i] == nullptr)
	//		{
	//			staging_buffers_[i] = std::make_unique<backend::Buffer>(backend::Buffer::create_staging_buffer(device_, PAGE_SIZE, data_[i].data()));
	//		}

	//		command_buffer.copy_buffer(*staging_buffers_[i], *buffers_[i / MAX_BUFFER_PAGE], PAGE_SIZE, 0, (i % MAX_BUFFER_PAGE) * PAGE_SIZE);
	//	}
	//}
}

template <typename DataType>
int32_t PageTable<DataType>::swap_in(uint16_t *page_state, uint32_t buffer_page_index)
{
	switch (replacement_policy_)
	{
		case ReplacementPolicy::LRU:
			return swap_in_lru(page_state, buffer_page_index);
		case ReplacementPolicy::RANDOM:
			return swap_in_random(page_state, buffer_page_index);
		default:
			return -1;
	}
}

template <typename DataType>
int32_t PageTable<DataType>::swap_in_random(uint16_t *page_state, uint32_t buffer_page_index)
{	
	if (buffer_to_table_[buffer_page_index] != -1)
	//if (page_state[buffer_page_index] & 0b10)
	{
		return -2; // hit
	}

	int32_t table_page_index = -1;
	if (!free_list_.empty())
	{
		table_page_index = free_list_.front();
		free_list_.pop_front();
	}
	else
	{
		for (size_t i = 0; i < total_page_num_; i++)
		{
			uint32_t swapped_buffer_index = table_to_buffer_[i];

			assert(swapped_buffer_index < buffer_page_count_ && swapped_buffer_index >= 0);
			
			if (!(page_state[swapped_buffer_index] & 1))
			{
				table_page_index = i;

				// swap out
				page_state[swapped_buffer_index] &= ~(1 << 1);
				buffer_to_table_[swapped_buffer_index] = -1;
				LOGI("Swap out buffer page: {}", swapped_buffer_index);
				
				break;
			}
		}
	}

	LOGI("Swap in buffer page: {}", buffer_page_index);

	return table_page_index;
}

template <typename DataType>
int32_t PageTable<DataType>::swap_in_lru(uint16_t *page_state, uint32_t buffer_page_index)
{
	// hit
	if (lru_page_table_.count(buffer_page_index))
	{
		lru_list_.splice(lru_list_.begin(), lru_list_, lru_page_table_[buffer_page_index]);
		return -2;
	}

	int32_t table_page_index = -1;

	if (!free_list_.empty())
	{
		table_page_index = free_list_.front();
		free_list_.pop_front();
	}
	else
	{
		uint32_t swapped_buffer_index = lru_list_.back();
		
		if ((page_state[swapped_buffer_index] & 1) && swapped_buffer_index < buffer_page_index)
		{
			return -1;
		}

		table_page_index = buffer_to_table_[swapped_buffer_index];

		lru_list_.pop_back();
		lru_page_table_.erase(swapped_buffer_index);
		page_state[swapped_buffer_index] &= ~(1 << 1);
		buffer_to_table_[swapped_buffer_index] = -1;
		LOGI("Swap out buffer page: {}", swapped_buffer_index);
	}

	LOGI("Swap in buffer page: {}", buffer_page_index);

	lru_list_.push_front(buffer_page_index);
	lru_page_table_[buffer_page_index] = lru_list_.begin();

	return table_page_index;
}

}