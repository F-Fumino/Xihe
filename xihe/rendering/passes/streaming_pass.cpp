#include "streaming_pass.h"

#include "common/timer.h"

namespace xihe::rendering
{
StreamingPass::StreamingPass(GpuLoDScene &gpu_lod_scene) :
    gpu_lod_scene_(gpu_lod_scene)
{}

void StreamingPass::execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables)
{
	Timer timer;
	timer.start();

	active_frame.reset_fence();
<<<<<<< HEAD
	
	auto time = timer.stop();
	LOGI("Wait time: {} ms", time * 1000.0f);

	gpu_lod_scene_.streaming(command_buffer);
=======

	auto time = timer.stop();
	LOGI("Wait time: {} ms", time * 1000.0f);
	
	
	 gpu_lod_scene_.streaming(command_buffer);
>>>>>>> parent of 9a05f49 (split cluster culling and cluster draw pre)
}

}        // namespace xihe::rendering
