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
PageTableState PageTable<DataType>::execute(backend::CommandBuffer &command_buffer, uint8_t *page_state)
{
	PageTableState state = PageTableState::NORMAL;

	uint32_t       num_request     = 0;
	const uint32_t empty_threshold = total_page_num_ / 4;
	const uint32_t full_threshold  = total_page_num_;

	const uint32_t max_binds = 1000;
	uint32_t       num_binds = 0;

	std::vector<std::vector<VkSparseMemoryBind>> binds;
	binds.resize(buffer_count_);

	for (size_t i = 0; i < buffer_page_count_; i++)
	{
		if (page_state[i] == 0b11) // request and already in page table
		{
			assert(buffer_to_table_[i] != -1);
			access(i);
			num_request++;
		}
	}

	for (size_t i = 0; i < buffer_page_count_; i++)
	{
		if (num_binds + 1 > max_binds)
		{
			break;
		}
		
		if (page_state[i] == 0b01) // request and not in page table
		{
			assert(buffer_to_table_[i] == -1);

			num_request++;

			int32_t table_page_index = swap_in(page_state, i);

			//if (table_page_index == -2)
			//{
			//	continue; // hit
			//}
			if (table_page_index == -1)
			{
				LOGW("Page Table is full");
				break;
			}

			++num_binds;

			table_to_buffer_[table_page_index] = i;
			buffer_to_table_[i]                = table_page_index;
			page_state[i] |= 0b10;

			VkSparseMemoryBind bind = {
			    .resourceOffset = (i % MAX_BUFFER_PAGE) * PAGE_SIZE,
			    .size           = PAGE_SIZE,
			    .memory         = get_memory(table_page_index),
			    .memoryOffset   = get_memory_offset(table_page_index),
			    .flags          = 0
			};
			binds[i / MAX_BUFFER_PAGE].push_back(bind);

			if (staging_buffers_[i] == nullptr)
			{
				staging_buffers_[i] = std::make_unique<backend::Buffer>(backend::Buffer::create_staging_buffer(device_, PAGE_SIZE, data_[i].data()));
			}

			command_buffer.copy_buffer(*staging_buffers_[i], *buffers_[i / MAX_BUFFER_PAGE], PAGE_SIZE, 0, (i % MAX_BUFFER_PAGE) * PAGE_SIZE);
		}
	}

	for (size_t i = 0; i < buffer_page_count_; i++)
	{
		page_state[i] &= ~1;        // clear page request
	}

	std::vector<VkSparseBufferMemoryBindInfo> binds_info;

	for (size_t i = 0; i < buffer_count_; i++)
	{
		if (!binds[i].empty())
		{
			VkSparseBufferMemoryBindInfo bind_info = {
			    .buffer    = buffers_[i]->get_handle(),
			    .bindCount = static_cast<uint32_t>(binds[i].size()),
			    .pBinds    = binds[i].data()
			};

			binds_info.push_back(bind_info);
		}
	}

	if (!binds_info.empty())
	{
		LOGI("Swap in {} pages", num_binds);
		LOGI("Free Page: {}", free_list_.size());

		VkBindSparseInfo sparse_bind_info = {
		    .sType           = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
		    .bufferBindCount = static_cast<uint32_t>(binds_info.size()),
		    .pBufferBinds    = binds_info.data()
		};

		VkResult result = vkQueueBindSparse(sparse_queue_->get_handle(), 1, &sparse_bind_info, device_.request_fence());

		device_.get_fence_pool().wait();
		device_.get_fence_pool().reset();
	}

	if (num_request < empty_threshold)
	{
		state = PageTableState::EMPTY;	
	}
	else if (num_request >= full_threshold && total_page_num_ != buffer_page_count_)
	{
		state = PageTableState::FULL;
	}

	return state;
}

template <typename DataType>
void PageTable<DataType>::test()
{
	std::vector<uint16_t> page_state{1, 1, 0, 1, 0, 1, 0, 1};
}

template <typename DataType>
void PageTable<DataType>::access(uint32_t buffer_page_index)
{
	switch (replacement_policy_)
	{
		case ReplacementPolicy::LRU:
			access_lru(buffer_page_index);
			break;
		case ReplacementPolicy::RANDOM:
			break;
		default:
			break;
	}
}

template <typename DataType>
void PageTable<DataType>::access_lru(uint32_t buffer_page_index)
{
	lru_list_.splice(lru_list_.begin(), lru_list_, lru_page_table_[buffer_page_index]);
}

template <typename DataType>
int32_t PageTable<DataType>::swap_in(uint8_t *page_state, uint32_t buffer_page_index)
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
int32_t PageTable<DataType>::swap_in_random(uint8_t *page_state, uint32_t buffer_page_index)
{	
	//if (buffer_to_table_[buffer_page_index] != -1)
	//{
	//	return -2; // hit
	//}

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
			
			if (page_state[swapped_buffer_index] == 2)
			{
				table_page_index = i;
				// swap out
				page_state[swapped_buffer_index] = 0;
				buffer_to_table_[swapped_buffer_index] = -1;
				//LOGI("Swap out buffer page: {}", swapped_buffer_index);
				
				break;
			}
		}
	}

	return table_page_index;
}

template <typename DataType>
int32_t PageTable<DataType>::swap_in_lru(uint8_t *page_state, uint32_t buffer_page_index)
{
	// hit
	//if (lru_page_table_.count(buffer_page_index))
	//{
	//	lru_list_.splice(lru_list_.begin(), lru_list_, lru_page_table_[buffer_page_index]);
	//	return -2;
	//}

	int32_t table_page_index = -1;

	if (!free_list_.empty())
	{
		table_page_index = free_list_.front();
		free_list_.pop_front();
	}
	else
	{
		uint32_t swapped_buffer_index = lru_list_.back();
		
		if (page_state[swapped_buffer_index] & 1)
		{
			return -1;
		}

		table_page_index = buffer_to_table_[swapped_buffer_index];

		// swap out
		lru_list_.pop_back();
		lru_page_table_.erase(swapped_buffer_index);
		page_state[swapped_buffer_index] &= ~(1 << 1);
		buffer_to_table_[swapped_buffer_index] = -1;
		//LOGI("Swap out buffer page: {}", swapped_buffer_index);
	}

	//LOGI("Swap in buffer page: {}", buffer_page_index);

	lru_list_.push_front(buffer_page_index);
	lru_page_table_[buffer_page_index] = lru_list_.begin();

	return table_page_index;
}

}