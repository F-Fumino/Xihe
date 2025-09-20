#include "sample_app.h"

#include "backend/shader_compiler/glsl_compiler.h"
#include "rendering/passes/bloom_pass.h"
#include "rendering/passes/cascade_shadow_pass.h"
#include "rendering/passes/clustered_lighting_pass.h"
#include "rendering/passes/geometry_pass.h"
#include "rendering/passes/mesh_draw_preparation.h"
#include "rendering/passes/mesh_draw_lod_preparation.h"
#include "rendering/passes/mesh_pass.h"
#include "rendering/passes/mesh_lod_pass.h"
#include "rendering/passes/occlusion_pass.h"
#include "rendering/passes/hzb_pass.h"
#include "rendering/passes/copy_pass.h"
#include "rendering/passes/mipmap_pass.h"
#include "rendering/passes/streaming_pass.h"
#include "rendering/passes/pointshadows_pass.h"
#include "rendering/passes/occlusion_draw_preparation.h"
#include "rendering/passes/instance_culling.h"
#include "rendering/passes/cluster_culling.h"
#include "rendering/passes/geometry_mesh_pass.h"
#include "rendering/passes/test_pass.h"
#include "scene_graph/components/camera.h"
#include "scene_graph/components/light.h"
#include "scene_graph/components/mesh.h"
#include "stats/stats.h"

//#define MESH_SHADER
//#define OCCLUSION
#define HAS_TEXTURE
//#define FIXED_CAMERA_TRACK

namespace xihe
{
using namespace rendering;

SampleApp::SampleApp()
{
	add_device_extension(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
	add_device_extension(VK_EXT_MESH_SHADER_EXTENSION_NAME);
	add_device_extension(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
	add_device_extension(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME);

	// for device address
	add_device_extension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);

	// for debug
	//add_device_extension(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
	add_device_extension(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);

	backend::GlslCompiler::set_target_environment(glslang::EShTargetSpv, glslang::EShTargetSpv_1_4);
}

bool SampleApp::prepare(Window *window)
{
	if (!XiheApp::prepare(window))
	{
		return false;
	}

	render_context_->create_sparse_bind_queue();

	asset_loader_ = std::make_unique<AssetLoader>(*device_);

#ifdef HAS_TEXTURE
	load_scene("scenes/sponza/Sponza01.gltf");
#else
	load_scene("scenes/factory/factory.gltf");
	/*load_scene("scenes/factory/lod1.gltf");*/
	/*load_scene("scenes/factory/lod.gltf");*/
	//load_scene("scenes/welded/9.gltf");
	//load_scene("scenes/industry/model30.gltf");
	//load_scene("scenes/factory/factory2.gltf");
	/*load_scene("scenes/factory/mesh280.gltf");*/
	/*load_scene("scenes/factory/mesh13.gltf");*/
	/*load_scene("scenes/factory/mesh143.gltf");*/
	/*load_scene("scenes/factory/mixed.gltf");*/
	/*load_scene("scenes/cold/mixed.gltf");*/
	/*load_scene("scenes/cold/sim.gltf");*/
#endif
	assert(scene_ && "Scene not loaded");
	update_bindless_descriptor_sets();

	gpu_lod_scene_ = std::make_unique<GpuLoDScene>(*device_);
	gpu_lod_scene_->initialize(*scene_);

	auto *skybox_texture = asset_loader_->load_texture_cube(*scene_, "skybox", "textures/uffizi_cube.ktx");

	auto light_pos   = glm::vec3(-150.0f, 188.0f, -225.0f);
	// auto light_pos   = glm::vec3(150.0f, 0.0f, 0.0f);
	auto light_color = glm::vec3(1.0, 1.0, 1.0);

	// Magic numbers used to offset lights in the Sponza scene
	/*for (int i = -4; i < 4; ++i)
	{
		for (int j = 0; j < 5; ++j)
		{
			glm::vec3 pos = light_pos;
			pos.x += i * 400;
			pos.z += j * 150;
			pos.y = 8;
			for (int k = 0; k < 6; ++k)
			{
				pos.y         = pos.y + (k * 50);
				light_color.x = static_cast<float>(rand()) / (RAND_MAX);
				light_color.y = static_cast<float>(rand()) / (RAND_MAX);
				light_color.z = static_cast<float>(rand()) / (RAND_MAX);
				sg::LightProperties props;
				props.color     = light_color;
				props.intensity = 2.0f;
				props.range     = 700.f;
				add_point_light(*scene_, pos, props);
			}
		}
	}*/

	for (int i = -4; i < 4; ++i)
	{
		for (int j = 0; j < 2; ++j)
		{
			glm::vec3 pos = light_pos;
			pos.x += i * 400;
			pos.z += j * (225 + 140);
			pos.y = 8;

			for (int k = 0; k < 3; ++k)
			{
				pos.y = pos.y + (k * 100);

				light_color.x = static_cast<float>(rand()) / (RAND_MAX);
				light_color.y = static_cast<float>(rand()) / (RAND_MAX);
				light_color.z = static_cast<float>(rand()) / (RAND_MAX);

				sg::LightProperties props;
				props.color     = light_color;
				props.intensity = 1.f;
				props.range     = 700.f;
				add_point_light(*scene_, pos, props);
			}
		}
	}

#ifndef FIXED_CAMERA_TRACK
	auto &camera_node = sg::add_free_camera(*scene_, "main_camera", render_context_->get_surface_extent());
#else
	/*auto &camera_node = sg::add_circle_path_camera(*scene_, "main_camera", render_context_->get_surface_extent(), 0.01f, glm::vec3(-7303.0f, -2219.0f, -35.0f), 1000.0f);*/
	/*auto &camera_node = sg::add_circle_path_camera(*scene_, "main_camera", render_context_->get_surface_extent(), 0.6f, glm::vec3(0.0f, 0.0f, 0.0f), 100.0f, glm::vec3(0.0f, 0.0f, 1.0f));*/
	auto &camera_node = sg::add_circle_path_camera(*scene_, "main_camera", render_context_->get_surface_extent(), 7.2f, glm::vec3(0.0f, 0.0f, 0.0f), 40.0f, glm::vec3(0.418212, -0.241846, 0.875000));        // 运动较快的相机
	//auto &camera_node = sg::add_circle_path_camera(*scene_, "main_camera", render_context_->get_surface_extent(), 2.4f, glm::vec3(0.0f, 0.0f, 0.0f), 100.0f, glm::vec3(0.0f, 0.0f, 1.0f));        // 运动较快的相机
	/*auto &camera_node = sg::add_circle_path_camera(*scene_, "main_camera", render_context_->get_surface_extent(), 0.6f, glm::vec3(0.0f, 0.0f, 0.0f), 10.0f, glm::vec3(0.0f, 0.0f, 1.0f));*/
	//auto &camera_node = sg::add_circle_path_camera(*scene_, "main_camera", render_context_->get_surface_extent(), 0.0f, glm::vec3(0.0f, 0.0f, 0.0f), 6.0f, glm::vec3(0.0f, 0.0f, 1.0f));
	//auto &camera_node = sg::add_line_path_camera(*scene_, "main_camera", render_context_->get_surface_extent(), 0.3f, glm::vec3(0.0f, -200.0f, 0.0f), glm::vec3(0.0f, 10.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));       // 剔除测试应该用的是这个
	//auto &camera_node = sg::add_line_path_camera(*scene_, "main_camera", render_context_->get_surface_extent(), 0.0f, glm::vec3(0.0f, -200.0f, 0.0f), glm::vec3(0.0f, 10.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));        // 远景
	//auto &camera_node = sg::add_line_path_camera(*scene_, "main_camera", render_context_->get_surface_extent(), 0.0f, glm::vec3(0.0f, -100.0f, 0.0f), glm::vec3(0.0f, 10.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));  // 中景
	//auto &camera_node = sg::add_line_path_camera(*scene_, "main_camera", render_context_->get_surface_extent(), 0.0f, glm::vec3(0.0f, -10.0f, 0.0f), glm::vec3(0.0f, 10.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));    // 近景
#endif        // FIXED_CAMERA_TRACK

	auto camera = &camera_node.get_component<sg::Camera>();
	camera_     = camera;

#ifdef MESH_SHADER

	// geometry pass
	{
		auto mesh_preparation_pass = std::make_unique<MeshDrawLoDPreparationPass>(*gpu_lod_scene_, *camera);
		graph_builder_->add_pass("Mesh Draw LoD Preparation", std::move(mesh_preparation_pass))
		    .bindables({{.type = BindableType::kStorageBufferWrite, .name = "draw command", .buffer_size = gpu_lod_scene_->get_instance_count() * sizeof(MeshDrawCommand)}})
		    .shader({"mesh_shading/prepare_mesh_draws.comp"})
		    .finalize();

		auto geometry_pass = std::make_unique<MeshLoDPass>(*gpu_lod_scene_, *camera);

		PassBindable hzb_bindable{BindableType::kSampledFromLastFrame, "hzb", vk::Format::eR32Sfloat, ExtentDescriptor::SwapchainRelative(1.0, 1.0)};
		hzb_bindable.image_properties.has_mip_levels = true;

		graph_builder_->add_pass("Geometry", std::move(geometry_pass))
		    .bindables({{.type = BindableType::kStorageBufferRead, .name = "draw command"},
		                {.type = BindableType::kStorageBufferReadWrite, .name = "page state", .buffer_size = static_cast<uint32_t>(gpu_lod_scene_->get_page_state_buffer().get_size())},
		                hzb_bindable,
	#ifdef OCCLUSION
		                {.type = BindableType::kStorageBufferWrite, .name = "recheck list", .buffer_size = static_cast<uint32_t>(gpu_lod_scene_->get_recheck_list_buffer().get_size())}
	#endif        // OCCLUSION
		    })
		    .attachments({{AttachmentType::kDepth, "depth"},
		                  {AttachmentType::kColor, "albedo"},
		                  {AttachmentType::kColor, "normal", vk::Format::eA2B10G10R10UnormPack32}})
		    .shader({"deferred/geometry_lod.task", "deferred/geometry_lod.mesh", "deferred/geometry_lod.frag"})
		    .finalize();
	}

#else
	
	{
		auto instance_culling_pass = std::make_unique<InstanceCullingPass>(*gpu_lod_scene_, *camera);
		graph_builder_->add_pass("Instance Culling", std::move(instance_culling_pass))
		    .bindables({{.type = BindableType::kStorageBufferWrite, .name = "instance visibility", .buffer_size = gpu_lod_scene_->get_instance_count() * sizeof(uint32_t)}})
		    .shader({"mesh_shading/instance_culling.comp"})
		    .finalize();
	}

	{
		PassBindable hzb_bindable{BindableType::kSampledFromLastFrame, "hzb", vk::Format::eR32Sfloat, ExtentDescriptor::SwapchainRelative(1.0, 1.0)};
		hzb_bindable.image_properties.has_mip_levels = true;

		auto cluster_culling_pass = std::make_unique<ClusterCullingPass>(*gpu_lod_scene_, *camera);
		graph_builder_->add_pass("Cluster Culling", std::move(cluster_culling_pass))
		    .bindables({
				hzb_bindable,
		        {.type = BindableType::kStorageBufferRead, .name = "instance visibility", .buffer_size = gpu_lod_scene_->get_instance_count() * sizeof(uint32_t)},
		        {.type = BindableType::kStorageBufferWrite, .name = "indirect command", .buffer_size = static_cast<uint32_t>(gpu_lod_scene_->get_indirect_command_buffer().get_size())},
				{.type = BindableType::kStorageBufferWrite, .name = "counts", .buffer_size = static_cast<uint32_t>(gpu_lod_scene_->get_counts_buffer().get_size())},
		        {.type = BindableType::kStorageBufferWrite, .name = "global index", .buffer_size = static_cast<uint32_t>(gpu_lod_scene_->get_global_index_buffer().get_size())},
		        {.type = BindableType::kStorageBufferReadWrite, .name = "page state", .buffer_size = static_cast<uint32_t>(gpu_lod_scene_->get_page_state_buffer().get_size())}
			})
		    .shader({"mesh_shading/cluster_culling.comp"})
		    .finalize();
	}

	{
		auto geometry_pass = std::make_unique<GeometryMeshPass>(*gpu_lod_scene_, *camera);

		graph_builder_->add_pass("Geometry", std::move(geometry_pass))
		    .bindables({
				{.type = BindableType::kStorageBufferReadAndIndirect, .name = "indirect command"},
		        {.type = BindableType::kHostBufferRead, .name = "counts"},
		        {.type = BindableType::kStorageBufferRead, .name = "global index"}
		    })
		    .attachments({{AttachmentType::kDepth, "depth"},
		                  {AttachmentType::kColor, "albedo"},
		                  {AttachmentType::kColor, "normal", vk::Format::eA2B10G10R10UnormPack32}})
		    .shader({"mesh_shading/geometry_mesh.vert", "mesh_shading/geometry_mesh.frag"})
		    .finalize();
	}
	
#endif        // MESH_SHADER

	{
		auto streaming_pass = std::make_unique<StreamingPass>(*gpu_lod_scene_);
		graph_builder_->add_pass("Streaming", std::move(streaming_pass))
		    .bindables({{.type = BindableType::kHostBufferReadWrite, .name = "page state"}})
		    .shader({""})
		    .finalize();
	}

	// hzb pass
	{
		PassBindable hzb_copy_bindable{BindableType::kStorageWrite, "hzb", vk::Format::eR32Sfloat, ExtentDescriptor::SwapchainRelative(1.0, 1.0)};
		hzb_copy_bindable.image_properties.has_mip_levels = true;
		hzb_copy_bindable.only_read_from_last_pass_       = true;

		auto copy_pass = std::make_unique<CopyPass>();
		graph_builder_->add_pass("copy", std::move(copy_pass))
		    .bindables({{BindableType::kSampled, "depth"},
		                hzb_copy_bindable})
		    .shader({"hzb/copy.comp"})
		    .finalize();

		PassBindable hzb_bindable{BindableType::kStorageReadWrite, "hzb", vk::Format::eR32Sfloat, ExtentDescriptor::SwapchainRelative(1.0, 1.0)};
		hzb_bindable.image_properties.has_mip_levels = true;

		auto mipmap_pass = std::make_unique<MipmapPass>();
		graph_builder_->add_pass("mipmap", std::move(mipmap_pass))
		    .bindables({{hzb_bindable}})
		    .shader({"hzb/mipmap.comp"})
		    .finalize();
	}

#ifdef OCCLUSION

	#ifdef MESH_SHADER

	{
		auto occlusion_preparation_pass = std::make_unique<OcclusionPreparationPass>(*gpu_lod_scene_);
		graph_builder_->add_pass("Occlusion Preparation", std::move(occlusion_preparation_pass))
		    .bindables({{.type = BindableType::kStorageBufferWrite, .name = "occlusion command", .buffer_size = static_cast<uint32_t>(gpu_lod_scene_->get_occlusion_command_buffer().get_size())},
		                {.type = BindableType::kStorageBufferRead, .name = "recheck list", .buffer_size = static_cast<uint32_t>(gpu_lod_scene_->get_recheck_list_buffer().get_size())},
		                {.type = BindableType::kStorageBufferWrite, .name = "recheck cluster", .buffer_size = static_cast<uint32_t>(gpu_lod_scene_->get_recheck_cluster_buffer().get_size())},
		                {.type = BindableType::kStorageBufferWrite, .name = "recheck count", .buffer_size = static_cast<uint32_t>(gpu_lod_scene_->get_recheck_counts_buffer().get_size())}})
		    .shader({"deferred/occlusion_preparation.comp"})
		    .finalize();
	}

	// HZB0 pass
	{
		PassBindable hzb0_copy_bindable{BindableType::kStorageWrite, "hzb0", vk::Format::eR32Sfloat, ExtentDescriptor::SwapchainRelative(1.0, 1.0)};
		hzb0_copy_bindable.image_properties.has_mip_levels = true;
		hzb0_copy_bindable.only_read_from_last_pass_       = true;

		auto copy0_pass = std::make_unique<CopyPass>();
		graph_builder_->add_pass("copy0", std::move(copy0_pass))
		    .bindables({{.type = BindableType::kSampled, .name = "depth", .only_read_from_last_pass_ = true},
		                hzb0_copy_bindable})
		    .shader({"hzb/copy.comp"})
		    .finalize();

		PassBindable hzb0_bindable{BindableType::kStorageReadWrite, "hzb0", vk::Format::eR32Sfloat, ExtentDescriptor::SwapchainRelative(1.0, 1.0)};
		hzb0_bindable.image_properties.has_mip_levels = true;

		auto mipmap0_pass = std::make_unique<MipmapPass>();
		graph_builder_->add_pass("mipmap0", std::move(mipmap0_pass))
		    .bindables({{hzb0_bindable}})
		    .shader({"hzb/mipmap.comp"})
		    .finalize();
	}

	{
		auto occlusion_pass = std::make_unique<OcclusionPass>(*gpu_lod_scene_, *camera);

		PassBindable hzb0_bindable{BindableType::kSampled, "hzb0", vk::Format::eR32Sfloat, ExtentDescriptor::SwapchainRelative(1.0, 1.0)};
		hzb0_bindable.image_properties.has_mip_levels = true;

		graph_builder_->add_pass("Occlusion", std::move(occlusion_pass))
		    .bindables({{.type = BindableType::kStorageBufferRead, .name = "occlusion command"},
		                {.type = BindableType::kStorageBufferRead, .name = "recheck count"},
		                {.type = BindableType::kStorageBufferRead, .name = "recheck cluster"},
		                hzb0_bindable})
		    .attachments({{.type = AttachmentType::kDepth, .name = "depth", .clear_on_load = false},
		                  {.type = AttachmentType::kColor, .name = "albedo", .clear_on_load = false},
		                  {.type = AttachmentType::kColor, .name = "normal", .format = vk::Format::eA2B10G10R10UnormPack32, .clear_on_load = false}})
		    .shader({"deferred/occlusion.task", "deferred/occlusion.mesh", "deferred/occlusion.frag"})
		    .finalize();
	}

	#else

	{

	}

	#endif

#endif        // OCCLUSION

	// lighting pass
	{
		auto lighting_pass = std::make_unique<ClusteredLightingPass>(scene_->get_components<sg::Light>(), *camera, nullptr, skybox_texture);

		graph_builder_->add_pass("Lighting", std::move(lighting_pass))

		    .bindables({{BindableType::kSampled, "depth"},
		                {BindableType::kSampled, "albedo"},
		                {BindableType::kSampled, "normal"}/*,
		                {BindableType::kSampled, "shadowmap"},
		                {BindableType::kSampledCube, "point shadowmaps"}*/})

		    .attachments({{AttachmentType::kColor, "lighting", vk::Format::eR16G16B16A16Sfloat}})

		    .shader({"deferred/lighting.vert", "deferred/lighting_unshadow.frag"})

		    .finalize();
	}

	// bloom pass
	{
		auto extract_pass = std::make_unique<BloomExtractPass>();

		graph_builder_->add_pass("Bloom Extract", std::move(extract_pass))
		    .bindables({{BindableType::kSampled, "lighting"},
		                {BindableType::kStorageWrite, "bloom_extract", vk::Format::eR16G16B16A16Sfloat, ExtentDescriptor::SwapchainRelative(0.5, 0.5)}})
		    .shader({"post_processing/bloom/threshold.comp"})
		    .finalize();

		auto downsample_pass0 = std::make_unique<BloomComputePass>();
		graph_builder_->add_pass("Bloom Downsample 0", std::move(downsample_pass0))
		    .bindables({{BindableType::kSampled, "bloom_extract"},
		                {BindableType::kStorageWrite, "bloom_down_sample_0", vk::Format::eR16G16B16A16Sfloat, ExtentDescriptor::SwapchainRelative(0.25, 0.25)}})
		    .shader({"post_processing/bloom/blur_down_first.comp"})
		    .finalize();

		auto downsample_pass1 = std::make_unique<BloomComputePass>();
		graph_builder_->add_pass("Bloom Downsample 1", std::move(downsample_pass1))
		    .bindables({{BindableType::kSampled, "bloom_down_sample_0"},
		                {BindableType::kStorageWrite, "bloom_down_sample_1", vk::Format::eR16G16B16A16Sfloat, ExtentDescriptor::SwapchainRelative(0.125, 0.125)}})
		    .shader({"post_processing/bloom/blur_down.comp"})
		    .finalize();

		auto downsample_pass2 = std::make_unique<BloomComputePass>();
		graph_builder_->add_pass("Bloom Downsample 2", std::move(downsample_pass2))
		    .bindables({{BindableType::kSampled, "bloom_down_sample_1"},
		                {BindableType::kStorageWrite, "bloom_down_sample_2", vk::Format::eR16G16B16A16Sfloat, ExtentDescriptor::SwapchainRelative(0.0625, 0.0625)}})
		    .shader({"post_processing/bloom/blur_down.comp"})
		    .finalize();

		auto upsample_pass0 = std::make_unique<BloomComputePass>();
		graph_builder_->add_pass("Bloom Upsample 0", std::move(upsample_pass0))
		    .bindables({{BindableType::kSampled, "bloom_down_sample_2"},
		                {BindableType::kStorageWrite, "bloom_up_sample_0", vk::Format::eR16G16B16A16Sfloat, ExtentDescriptor::SwapchainRelative(0.125, 0.125)}})
		    .shader({"post_processing/bloom/blur_up.comp"})
		    .finalize();

		auto upsample_pass1 = std::make_unique<BloomComputePass>();
		graph_builder_->add_pass("Bloom Upsample 1", std::move(upsample_pass1))
		    .bindables({{BindableType::kSampled, "bloom_up_sample_0"},
		                {BindableType::kStorageWrite, "bloom_up_sample_1", vk::Format::eR16G16B16A16Sfloat, ExtentDescriptor::SwapchainRelative(0.25, 0.25)}})
		    .shader({"post_processing/bloom/blur_up.comp"})
		    .finalize();

		auto upsample_pass2 = std::make_unique<BloomComputePass>();
		graph_builder_->add_pass("Bloom Upsample 2", std::move(upsample_pass2))
		    .bindables({{BindableType::kSampled, "bloom_up_sample_1"},
		                {BindableType::kStorageWrite, "bloom_up_sample_2", vk::Format::eR16G16B16A16Sfloat, ExtentDescriptor::SwapchainRelative(0.5, 0.5)}})
		    .shader({"post_processing/bloom/blur_up.comp"})
		    .finalize();
	}

	gui_ = std::make_unique<Gui>(*this, *window, stats_.get());
	// composite pass
	{
		auto composite_pass = std::make_unique<BloomCompositePass>();
		graph_builder_->add_pass("Bloom Composite", std::move(composite_pass))
		    .bindables({{BindableType::kSampled, "lighting"},
		                {BindableType::kSampled, "bloom_up_sample_2"}})
		    .shader({"post_processing/bloom_composite.vert", "post_processing/bloom_composite.frag"})
		    .gui(gui_.get())
		    .present()
		    .finalize();
	}
	{
	}

	graph_builder_->build();
	LOGI("Graph builds successfully!");

#ifdef HAS_TEXTURE
	MeshPass::show_texture();
	MeshLoDPass::show_texture();
	GeometryMeshPass::show_texture();
#endif

	return true;
}

void SampleApp::update(float delta_time)
{
	static int num         = 0;
	auto camera_node = camera_->get_node();
	if (camera_node->has_component<xihe::sg::Script>())
	{
		auto &camera_script = camera_node->get_component<xihe::sg::Script>();
		if (camera_script.is_end())
		{
			num++;
			if (num == 80)
			{
				LOGI("Average Culling Primitive: {}", stats_->get_data(xihe::stats::StatIndex::kClippingPrimsAvg)[0]);
				LOGI("Average GPU Time: {}", stats_->get_data(xihe::stats::StatIndex::kGpuTimeAvg)[0]);
				LOGI("Average Frame Time: {}", stats_->get_data(xihe::stats::StatIndex::kFrameTimeAvg)[0]);
				LOGI("Average Page Table Time: {}", gpu_lod_scene_->get_page_table_time() * 1000);
				LOGI("Average Bind Time: {}", gpu_lod_scene_->get_bind_time() * 1000);
				LOGI("Average Page Table Hit Probability: {}", gpu_lod_scene_->get_page_table_hit_probability());
				LOGI("Average Memory Utilization: {}", gpu_lod_scene_->get_memory_utilization());
			}
		}
	}

	MeshLoDPass::show_meshlet_view(show_meshlet_view_);
	MeshLoDPass::use_lod(use_lod_);
	MeshLoDPass::show_lod_view(show_lod_view_);
	//MeshLoDPass::freeze_frustum(freeze_frustum_, camera_);
	MeshLoDPass::show_line(show_line_);
	MeshLoDPass::use_occlusion(use_occlusion_);

	GeometryMeshPass::show_meshlet_view(show_meshlet_view_);
	ClusterCullingPass::use_lod(use_lod_);

	//LightingPass::show_cascade_view(show_cascade_view_);
	XiheApp::update(delta_time);
}

void SampleApp::request_gpu_features(backend::PhysicalDevice &gpu)
{
	XiheApp::request_gpu_features(gpu);
	
	// for sparse resources
	gpu.get_mutable_requested_features().sparseBinding = VK_TRUE;
	gpu.get_mutable_requested_features().sparseResidencyBuffer = VK_TRUE;

	// for buffer 
	gpu.get_mutable_requested_features().shaderInt16 = VK_TRUE;
	gpu.get_mutable_requested_features().shaderInt64 = VK_TRUE;
	
	// for line
	gpu.get_mutable_requested_features().fillModeNonSolid = VK_TRUE;

	// for debug
	//REQUEST_REQUIRED_FEATURE(gpu, vk::PhysicalDeviceFaultFeaturesEXT, deviceFault);

	REQUEST_REQUIRED_FEATURE(gpu, vk::PhysicalDeviceMeshShaderFeaturesEXT, meshShader);
	REQUEST_REQUIRED_FEATURE(gpu, vk::PhysicalDeviceMeshShaderFeaturesEXT, meshShaderQueries);
	REQUEST_REQUIRED_FEATURE(gpu, vk::PhysicalDeviceMeshShaderFeaturesEXT, taskShader);

	REQUEST_REQUIRED_FEATURE(gpu, vk::PhysicalDeviceVulkan11Features, shaderDrawParameters);
	REQUEST_REQUIRED_FEATURE(gpu, vk::PhysicalDeviceVulkan11Features, storageBuffer16BitAccess);
	
	REQUEST_REQUIRED_FEATURE(gpu, vk::PhysicalDeviceVulkan12Features, storageBuffer8BitAccess);
	REQUEST_REQUIRED_FEATURE(gpu, vk::PhysicalDeviceVulkan12Features, shaderInt8);
	REQUEST_REQUIRED_FEATURE(gpu, vk::PhysicalDeviceVulkan12Features, bufferDeviceAddress);
	
	REQUEST_REQUIRED_FEATURE(gpu, vk::PhysicalDeviceFragmentShadingRateFeaturesKHR, primitiveFragmentShadingRate);
	//REQUEST_REQUIRED_FEATURE(gpu, vk::PhysicalDeviceDescriptorIndexingFeatures, descriptorBindingStorageBufferUpdateAfterBind);
	// REQUEST_REQUIRED_FEATURE(gpu, vk::PhysicalDeviceFragmentShadingRateFeaturesKHR, attachmentFragmentShadingRate);
}

void SampleApp::draw_gui()
{
	gui_->show_stats(*stats_);

	gui_->show_views_window(
	    /* body = */ [this]() {
		    ImGui::Checkbox("Meshlet", &show_meshlet_view_);
		    ImGui::Checkbox("LOD", &use_lod_);
		    ImGui::Checkbox("LOD visual", &show_lod_view_);
		    //ImGui::Checkbox("LOD可视化", &show_lod_view_);
		    ImGui::Checkbox("Wireframe", &show_line_);
		    ImGui::Checkbox("Occlusion", &use_occlusion_);
	    },
	    /* lines = */ 2);
}
}        // namespace xihe

std::unique_ptr<xihe::Application> create_application()
{
	return std::make_unique<xihe::SampleApp>();
}