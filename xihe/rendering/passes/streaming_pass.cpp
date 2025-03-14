#include "streaming_pass.h"

namespace xihe::rendering
{
StreamingPass::StreamingPass(GpuLoDScene &gpu_lod_scene) :
    gpu_lod_scene_(gpu_lod_scene)
{}

void StreamingPass::execute(backend::CommandBuffer &command_buffer, RenderFrame &active_frame, std::vector<ShaderBindable> input_bindables)
{

}

}        // namespace xihe::rendering
