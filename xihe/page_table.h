#pragma once

#include <list>

#include "backend/device.h"
#include "backend/buffer.h"

#define PAGE_SIZE (65536 * 2)
//#define MAX_VERTEX_TABLE_SIZE (3ULL * 1024 * 1024 * 1024)
#define MAX_INDEX_TABLE_SIZE (1ULL * 1024 * 1024 * 1024)
#define MAX_BUFFER_SIZE (1ULL * 1024 * 1024 * 1024)
#define MAX_VERTEX_TABLE_SIZE    (24 * 1024 * 1024)
//#define MAX_INDEX_TABLE_SIZE     (8 * 1024 * 1024)
//#define MAX_BUFFER_SIZE          (8 * 1024 * 1024)
#define MAX_VERTEX_TABLE_PAGE uint32_t(MAX_VERTEX_TABLE_SIZE / PAGE_SIZE)
#define MAX_INDEX_TABLE_PAGE uint32_t(MAX_INDEX_TABLE_SIZE / PAGE_SIZE)
#define MAX_BUFFER_PAGE uint32_t(MAX_BUFFER_SIZE / PAGE_SIZE)

namespace xihe
{
template <typename DataType>
class PageTable : public backend::allocated::SparseResources
{
  public:

	enum class ReplacementPolicy
	{
		LRU,
		RANDOM
	};

	ReplacementPolicy replacement_policy_ = ReplacementPolicy::LRU;

	PageTable(backend::Device &device, uint32_t table_page_num, vk::DeviceSize page_size);

	void init(uint32_t buffer_count, uint32_t buffer_page_count);
	void allocate_pages();

	void execute(backend::CommandBuffer &command_buffer, uint16_t *page_state);

	int32_t swap_in(uint16_t *page_state, uint32_t buffer_page_index);
	int32_t swap_in_random(uint16_t *page_state, uint32_t buffer_page_index);
	int32_t swap_in_lru(uint16_t *page_state, uint32_t buffer_page_index);

	std::vector<std::unique_ptr<backend::Buffer>> buffers_;        // all buffers
	std::vector<std::vector<DataType>>            data_;           // every buffer page's data

private:
	backend::Device &device_;

	const backend::Queue *sparse_queue_{nullptr};

	uint32_t buffer_page_count_{};        // number of pages in the all buffers
	uint32_t buffer_count_{};             // number of buffers

	std::vector<std::unique_ptr<backend::Buffer>> staging_buffers_;        // every buffer page's staging buffer, for data transfer
	std::vector<uint32_t>                         page_swapped_in_;        // ervery buffer page's table index, -1 for not in the table

	// random
	std::list<uint32_t> free_list_;

	// LRU
	std::list<uint32_t>                                    lru_list_;
	std::unordered_map<int, std::list<uint32_t>::iterator> lru_page_table_;

	std::vector<uint32_t> table_to_buffer_;        // every table page's vertex page index
	std::vector<uint32_t> buffer_to_table_;        // every vertex page's buffer page index
};
}