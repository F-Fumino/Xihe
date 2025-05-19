#pragma once

#include "stats_provider.h"
#include "common/logging.h"

namespace xihe::stats
{
class FrameTimeProvider : public StatsProvider
{
public:
	FrameTimeProvider(std::set<StatIndex> &requested_stats)
	{
		// Remove from requested set to stop other providers looking for it.
		requested_stats.erase(StatIndex::kFrameTimes);
	
	}

	bool is_available(StatIndex index) const override
	{
		return index == StatIndex::kFrameTimes ||
			index == StatIndex::kFrameTimeAvg || 
			index == StatIndex::kFrameTimeMax;
	}

	Counters sample(float delta_time) override
	{
		Counters res;

		frame_num_ += 1;

		if (ignored_frame_num_ == -1)
		{
			frame_times_sum_ += delta_time;
			max_frame_time_ = std::max(max_frame_time_, delta_time);

			res[StatIndex::kFrameTimeAvg].result = frame_times_sum_ / frame_num_ * 1000.f;
			res[StatIndex::kFrameTimeMax].result = max_frame_time_ * 1000.f; 
		}
		else
		{
			res[StatIndex::kFrameTimeAvg].result = 0.f;
			res[StatIndex::kFrameTimeMax].result = 0.f; 
		}

		//LOGI("Average Frame Time: {}", res[StatIndex::kFrameTimeAvg].result);

		if (frame_num_ == ignored_frame_num_)
		{
			ignored_frame_num_ = -1;
			frame_num_         = 0;
		}

		// frame_times comes directly from delta_time
		res[StatIndex::kFrameTimes].result   = delta_time * 1000.f;
		return res;
	}

private:
	int32_t  ignored_frame_num_{10};
	uint32_t frame_num_{0};
    float    frame_times_sum_{0};
	float    max_frame_time_{0};
};
}
