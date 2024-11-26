#include "temp_app.h"

#include "backend/shader_compiler/glsl_compiler.h"
#include "rendering/subpasses/geometry_subpass.h"
#include "rendering/subpasses/lighting_subpass.h"
#include "scene_graph/components/camera.h"

namespace xihe
{
TempApp::TempApp()
{
	// Adding device extensions
	add_device_extension(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
	add_device_extension(VK_EXT_MESH_SHADER_EXTENSION_NAME);
	add_device_extension(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);

	backend::GlslCompiler::set_target_environment(glslang::EShTargetSpv, glslang::EShTargetSpv_1_4);
}

bool TempApp::prepare(Window *window)
{
	if (!XiheApp::prepare(window))
	{
		return false;
	}

	load_scene("scenes/sponza/Sponza01.gltf");
	// load_scene("scenes/cube.gltf");
	assert(scene_ && "Scene not loaded");

	update_bindless_descriptor_sets();

	auto &camera_node = xihe::sg::add_free_camera(*scene_, "main_camera", render_context_->get_surface_extent());
	auto  camera      = &camera_node.get_component<xihe::sg::Camera>();

	// geometry pass
	{
		rendering::PassInfo pass_info{};

		pass_info.outputs = {
		    //{rendering::RdgResourceType::kAttachment, "albedo", vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eInputAttachment},
		    {rendering::RdgResourceType::kSwapchain, "swapchain"},
		    {rendering::RdgResourceType::kAttachment, "depth", common::get_suitable_depth_format(get_device()->get_gpu().get_handle()), vk::ImageUsageFlagBits::eInputAttachment},
		    {rendering::RdgResourceType::kAttachment, "normal", vk::Format::eA2B10G10R10UnormPack32, vk::ImageUsageFlagBits::eInputAttachment},
		};

		auto subpass = std::make_unique<rendering::GeometrySubpass>(*render_context_, backend::ShaderSource{"deferred/geometry.vert"}, backend::ShaderSource{"deferred/geometry.frag"}, *scene_, *camera);

		std::vector<std::unique_ptr<rendering::Subpass>> subpasses;
		subpasses.push_back(std::move(subpass));

		rdg_builder_->add_raster_pass("geometry_pass", std::move(pass_info), std::move(subpasses));
	}

	//// lighting pass
	//{
	//	rendering::PassInfo pass_info{};
	//	pass_info.inputs = {
	//	    {rendering::RdgResourceType::kAttachment, "albedo"},
	//	    {rendering::RdgResourceType::kAttachment, "depth"},
	//	    {rendering::RdgResourceType::kAttachment, "normal"}};
	//	pass_info.outputs = {
	//	    {rendering::RdgResourceType::kSwapchain, "swapchain"}};

	//	auto lighting_vs = backend::ShaderSource{"deferred/lighting.vert"};
	//	auto lighting_fs = backend::ShaderSource{"deferred/lighting.frag"};
	//	auto subpass     = std::make_unique<rendering::LightingSubpass>(*render_context_, std::move(lighting_vs), std::move(lighting_fs), *camera, *scene_,nullptr);

	//	std::vector<std::unique_ptr<rendering::Subpass>> subpasses;
	//	subpasses.push_back(std::move(subpass));

	//	rdg_builder_->add_raster_pass("lighting_pass", std::move(pass_info), std::move(subpasses));
	//}

	return true;
}

void TempApp::update(float delta_time)
{
	XiheApp::update(delta_time);
}

void TempApp::request_gpu_features(backend::PhysicalDevice &gpu)
{
	XiheApp::request_gpu_features(gpu);

	REQUEST_REQUIRED_FEATURE(gpu, vk::PhysicalDeviceDynamicRenderingFeatures, dynamicRendering);
}
}

std::unique_ptr<xihe::Application> create_application()
{
	return std::make_unique<xihe::TempApp>();
}