#define TINYGLTF_IMPLEMENTATION
#include "gltf_loader.h"

#include <future>
#include <limits>
#include <queue>

#include "common/glm_common.h"
#include <ctpl_stl.h>
#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "xatlas.h"

#include "backend/buffer.h"
#include "backend/device.h"
#include "common/helpers.h"
#include "common/logging.h"
#include "common/timer.h"
#include "platform/filesystem.h"

#include "components/light.h"
#include "components/image/astc.h"
#include "asset_loader.h"
#include "geometry_data.h"
#include "components/camera.h"
#include "components/image.h"
#include "components/material.h"
#include "components/mesh.h"
#include "components/sampler.h"
#include "components/sub_mesh.h"
#include "components/texture.h"
#include "components/transform.h"
#include "node.h"
#include "scene.h"

//#define CALCULATE_NORMAL_MAP

namespace xihe
{
namespace
{

inline vk::Filter find_min_filter(int min_filter)
{
	switch (min_filter)
	{
		case TINYGLTF_TEXTURE_FILTER_NEAREST:
		case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST:
		case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR:
			return vk::Filter::eNearest;
		case TINYGLTF_TEXTURE_FILTER_LINEAR:
		case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST:
		case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR:
			return vk::Filter::eLinear;
		default:
			return vk::Filter::eLinear;
	}
};

inline vk::SamplerMipmapMode find_mipmap_mode(int min_filter)
{
	switch (min_filter)
	{
		case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST:
		case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST:
			return vk::SamplerMipmapMode::eNearest;
		case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR:
		case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR:
			return vk::SamplerMipmapMode::eLinear;
		default:
			return vk::SamplerMipmapMode::eLinear;
	}
};

inline vk::Filter find_mag_filter(int mag_filter)
{
	switch (mag_filter)
	{
		case TINYGLTF_TEXTURE_FILTER_NEAREST:
			return vk::Filter::eNearest;
		case TINYGLTF_TEXTURE_FILTER_LINEAR:
			return vk::Filter::eLinear;
		default:
			return vk::Filter::eLinear;
	}
};

inline vk::SamplerAddressMode find_wrap_mode(int wrap)
{
	switch (wrap)
	{
		case TINYGLTF_TEXTURE_WRAP_REPEAT:
			return vk::SamplerAddressMode::eRepeat;
		case TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE:
			return vk::SamplerAddressMode::eClampToEdge;
		case TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT:
			return vk::SamplerAddressMode::eMirroredRepeat;
		default:
			return vk::SamplerAddressMode::eRepeat;
	}
}

bool texture_needs_srgb_colorspace(const std::string &name)
{
	// The gltf spec states that the base and emissive textures MUST be encoded with the sRGB
	// transfer function. All other texture types are linear.
	if (name == "baseColorTexture" || name == "emissiveTexture")
	{
		return true;
	}

	// metallicRoughnessTexture, normalTexture & occlusionTexture must be linear
	assert(name == "metallicRoughnessTexture" || name == "normalTexture" || name == "occlusionTexture");
	return false;
}

std::vector<uint8_t> get_attribute_data(const tinygltf::Model *model, uint32_t accessor_id)
{
	assert(accessor_id < model->accessors.size());
	auto &accessor = model->accessors[accessor_id];
	assert(accessor.bufferView < model->bufferViews.size());
	auto &bufferView = model->bufferViews[accessor.bufferView];
	assert(bufferView.buffer < model->buffers.size());
	auto &buffer = model->buffers[bufferView.buffer];

	size_t stride    = accessor.ByteStride(bufferView);
	size_t startByte = accessor.byteOffset + bufferView.byteOffset;
	size_t endByte   = startByte + accessor.count * stride;

	return {buffer.data.begin() + startByte, buffer.data.begin() + endByte};
};

size_t get_attribute_size(const tinygltf::Model *model, uint32_t accessor_id)
{
	assert(accessor_id < model->accessors.size());
	return model->accessors[accessor_id].count;
};

size_t get_attribute_stride(const tinygltf::Model *model, uint32_t accessor_id)
{
	assert(accessor_id < model->accessors.size());
	auto &accessor = model->accessors[accessor_id];
	assert(accessor.bufferView < model->bufferViews.size());
	auto &bufferView = model->bufferViews[accessor.bufferView];

	return accessor.ByteStride(bufferView);
};

vk::Format get_attribute_format(const tinygltf::Model *model, uint32_t accessor_id)
{
	assert(accessor_id < model->accessors.size());
	auto &accessor = model->accessors[accessor_id];

	vk::Format format;

	switch (accessor.componentType)
	{
		case TINYGLTF_COMPONENT_TYPE_BYTE:
		{
			static const std::map<int, vk::Format> mapped_format = {{TINYGLTF_TYPE_SCALAR, vk::Format::eR8Sint},
			                                                        {TINYGLTF_TYPE_VEC2, vk::Format::eR8G8Sint},
			                                                        {TINYGLTF_TYPE_VEC3, vk::Format::eR8G8B8Sint},
			                                                        {TINYGLTF_TYPE_VEC4, vk::Format::eR8G8B8A8Sint}};

			format = mapped_format.at(accessor.type);

			break;
		}
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
		{
			static const std::map<int, vk::Format> mapped_format = {{TINYGLTF_TYPE_SCALAR, vk::Format::eR8Uint},
			                                                        {TINYGLTF_TYPE_VEC2, vk::Format::eR8G8Uint},
			                                                        {TINYGLTF_TYPE_VEC3, vk::Format::eR8G8B8Uint},
			                                                        {TINYGLTF_TYPE_VEC4, vk::Format::eR8G8B8A8Uint}};

			static const std::map<int, vk::Format> mapped_format_normalize = {{TINYGLTF_TYPE_SCALAR, vk::Format::eR8Unorm},
			                                                                  {TINYGLTF_TYPE_VEC2, vk::Format::eR8G8Unorm},
			                                                                  {TINYGLTF_TYPE_VEC3, vk::Format::eR8G8B8Unorm},
			                                                                  {TINYGLTF_TYPE_VEC4, vk::Format::eR8G8B8A8Unorm}};

			if (accessor.normalized)
			{
				format = mapped_format_normalize.at(accessor.type);
			}
			else
			{
				format = mapped_format.at(accessor.type);
			}

			break;
		}
		case TINYGLTF_COMPONENT_TYPE_SHORT:
		{
			static const std::map<int, vk::Format> mapped_format = {{TINYGLTF_TYPE_SCALAR, vk::Format::eR16Sint},
			                                                        {TINYGLTF_TYPE_VEC2, vk::Format::eR16G16Sint},
			                                                        {TINYGLTF_TYPE_VEC3, vk::Format::eR16G16B16Sint},
			                                                        {TINYGLTF_TYPE_VEC4, vk::Format::eR16G16B16A16Sint}};

			format = mapped_format.at(accessor.type);

			break;
		}
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
		{
			static const std::map<int, vk::Format> mapped_format = {{TINYGLTF_TYPE_SCALAR, vk::Format::eR16Uint},
			                                                        {TINYGLTF_TYPE_VEC2, vk::Format::eR16G16Uint},
			                                                        {TINYGLTF_TYPE_VEC3, vk::Format::eR16G16B16Uint},
			                                                        {TINYGLTF_TYPE_VEC4, vk::Format::eR16G16B16A16Uint}};

			static const std::map<int, vk::Format> mapped_format_normalize = {{TINYGLTF_TYPE_SCALAR, vk::Format::eR16Unorm},
			                                                                  {TINYGLTF_TYPE_VEC2, vk::Format::eR16G16Unorm},
			                                                                  {TINYGLTF_TYPE_VEC3, vk::Format::eR16G16B16Unorm},
			                                                                  {TINYGLTF_TYPE_VEC4, vk::Format::eR16G16B16A16Unorm}};

			if (accessor.normalized)
			{
				format = mapped_format_normalize.at(accessor.type);
			}
			else
			{
				format = mapped_format.at(accessor.type);
			}

			break;
		}
		case TINYGLTF_COMPONENT_TYPE_INT:
		{
			static const std::map<int, vk::Format> mapped_format = {{TINYGLTF_TYPE_SCALAR, vk::Format::eR32Sint},
			                                                        {TINYGLTF_TYPE_VEC2, vk::Format::eR32G32Sint},
			                                                        {TINYGLTF_TYPE_VEC3, vk::Format::eR32G32B32Sint},
			                                                        {TINYGLTF_TYPE_VEC4, vk::Format::eR32G32B32A32Sint}};

			format = mapped_format.at(accessor.type);

			break;
		}
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
		{
			static const std::map<int, vk::Format> mapped_format = {{TINYGLTF_TYPE_SCALAR, vk::Format::eR32Uint},
			                                                        {TINYGLTF_TYPE_VEC2, vk::Format::eR32G32Uint},
			                                                        {TINYGLTF_TYPE_VEC3, vk::Format::eR32G32B32Uint},
			                                                        {TINYGLTF_TYPE_VEC4, vk::Format::eR32G32B32A32Uint}};

			format = mapped_format.at(accessor.type);

			break;
		}
		case TINYGLTF_COMPONENT_TYPE_FLOAT:
		{
			static const std::map<int, vk::Format> mapped_format = {{TINYGLTF_TYPE_SCALAR, vk::Format::eR32Sfloat},
			                                                        {TINYGLTF_TYPE_VEC2, vk::Format::eR32G32Sfloat},
			                                                        {TINYGLTF_TYPE_VEC3, vk::Format::eR32G32B32Sfloat},
			                                                        {TINYGLTF_TYPE_VEC4, vk::Format::eR32G32B32A32Sfloat}};

			format = mapped_format.at(accessor.type);

			break;
		}
		default:
		{
			format = vk::Format::eUndefined;
			break;
		}
	}

	return format;
};

vk::IndexType get_index_type(const tinygltf::Model *model, int accessor_index)
{
	const auto &accessor = model->accessors[accessor_index];
	switch (accessor.componentType)
	{
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
			return vk::IndexType::eUint8EXT;        // Note: Requires enabling extension VK_EXT_index_type_uint8
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
			return vk::IndexType::eUint16;
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
			return vk::IndexType::eUint32;
		default:
			throw std::runtime_error("Unsupported index component type");
	}
}

std::vector<uint8_t> convert_indices_to_uint16(const std::vector<uint8_t> &indices)
{
	size_t               count = indices.size();
	std::vector<uint8_t> converted_indices(count * 2);        // uint16_t is 2 bytes

	const uint8_t *src = indices.data();
	uint16_t      *dst = reinterpret_cast<uint16_t *>(converted_indices.data());

	for (size_t i = 0; i < count; ++i)
	{
		dst[i] = static_cast<uint16_t>(src[i]);
	}

	return converted_indices;
}

std::vector<uint8_t> convert_underlying_data_stride(const std::vector<uint8_t> &src_data, uint32_t src_stride, uint32_t dst_stride)
{
	auto elem_count = to_u32(src_data.size()) / src_stride;

	std::vector<uint8_t> result(elem_count * dst_stride);

	for (uint32_t idxSrc = 0, idxDst = 0;
	     idxSrc < src_data.size() && idxDst < result.size();
	     idxSrc += src_stride, idxDst += dst_stride)
	{
		std::copy_n(src_data.begin() + idxSrc, src_stride, result.begin() + idxDst);
	}

	return result;
}

std::string to_snake_case(const std::string &text)
{
	std::stringstream result;

	for (const auto ch : text)
	{
		if (std::isalpha(ch))
		{
			if (std::isspace(ch))
			{
				result << "_";
			}
			else
			{
				if (std::isupper(ch))
				{
					result << "_";
				}

				result << static_cast<char>(std::tolower(ch));
			}
		}
		else
		{
			result << ch;
		}
	}

	return result.str();
}

std::vector<bool> parse_srgb_requirements(const tinygltf::Model &model)
{
	std::vector<bool> needs_srgb(model.images.size(), false);

	for (const auto &material : model.materials)
	{
		for (const auto &value : material.values)
		{
			if (value.first.find("Texture") != std::string::npos && texture_needs_srgb_colorspace(value.first))
			{
				int texture_index = value.second.TextureIndex();
				if (texture_index >= 0 && texture_index < model.textures.size())
				{
					int image_index = model.textures[texture_index].source;
					if (image_index >= 0 && image_index < needs_srgb.size())
					{
						needs_srgb[image_index] = true;
					}
				}
			}
		}

		for (const auto &value : material.additionalValues)
		{
			if (value.first.find("Texture") != std::string::npos && texture_needs_srgb_colorspace(value.first))
			{
				int texture_index = value.second.TextureIndex();
				if (texture_index >= 0 && texture_index < model.textures.size())
				{
					int image_index = model.textures[texture_index].source;
					if (image_index >= 0 && image_index < needs_srgb.size())
					{
						needs_srgb[image_index] = true;
					}
				}
			}
		}
	}

	return needs_srgb;
}

}        // namespace

std::unordered_map<std::string, bool> GltfLoader::supported_extensions_ = {
    {"KHR_lights_punctual", false}};

GltfLoader::GltfLoader(backend::Device &device) :
    device_{device}
{}

std::unique_ptr<sg::Scene> GltfLoader::read_scene_from_file(const std::string &file_name, int scene_index)
{
	std::string err;
	std::string warn;

	tinygltf::TinyGLTF loader;

	fs::Path gltf_file_path = fs::path::get(fs::path::Type::kAssets) / file_name;

	fs::get_extension(file_name);
	bool import_result = false;
	if (fs::get_extension(file_name) == "gltf")
	{
		import_result = loader.LoadASCIIFromFile(&model_, &err, &warn, gltf_file_path.string());
	}
	else if(fs::get_extension(file_name) == "glb")
	{
		import_result = loader.LoadBinaryFromFile(&model_, &err, &warn, gltf_file_path.string());
	}

	if (!err.empty())
	{
		LOGE("Error loading gltf model: {}.", err.c_str());

		return nullptr;
	}

	if (!warn.empty())
	{
		LOGW("{}", warn.c_str());
	}

	if (!import_result)
	{
		LOGE("Failed to load gltf file {}.", gltf_file_path.string().c_str());

		return nullptr;
	}

	const size_t pos1 = file_name.find_last_of('/');
	const size_t pos2 = file_name.find_last_of('.');

	model_path_ = file_name.substr(0, pos1);
	std::string name = file_name.substr(pos1 + 1, pos2 - pos1 - 1);

	if (pos1 == std::string::npos)
	{
		model_path_.clear();
	}

	return std::make_unique<sg::Scene>(load_scene(name, scene_index));
}

std::unique_ptr<sg::SubMesh> GltfLoader::minimal_read_model(const std::string &file_name)
{
	tinygltf::Model    model;
	tinygltf::TinyGLTF loader;
	std::string        err, warn;
	fs::Path           gltf_file_path = fs::path::get(fs::path::Type::kAssets) / file_name;

	if (!loader.LoadASCIIFromFile(&model, &err, &warn, gltf_file_path.string()))
	{
		throw std::runtime_error("Failed to load GLTF file: " + err);
	}

	if (model.meshes.empty())
	{
		throw std::runtime_error("No mesh found in GLTF file");
	}

	const auto &gltf_mesh = model.meshes[0];

	for (const auto &gltf_primitive : gltf_mesh.primitives)
	{
		MeshPrimitiveData primitive_data;
		primitive_data.name = gltf_mesh.name;

		// auto submesh_name = fmt::format("'{}' mesh, primitive #{}", gltf_mesh.name, i_primitive);
		// auto submesh      = std::make_unique<sg::SubMesh>(std::move(submesh_name));

		for (auto &attribute : gltf_primitive.attributes)
		{
			VertexAttributeData attrib_data;
			std::string         attrib_name = attribute.first;
			std::ranges::transform(attrib_name, attrib_name.begin(), ::tolower);

			int accessor_index = attribute.second;
			attrib_data.format = get_attribute_format(&model, accessor_index);
			attrib_data.stride = to_u32(get_attribute_stride(&model, accessor_index));
			attrib_data.data   = get_attribute_data(&model, accessor_index);

			primitive_data.attributes[attrib_name] = std::move(attrib_data);

			if (attrib_name == "position")
			{
				primitive_data.vertex_count = to_u32(model.accessors[accessor_index].count);
			}
		}

		if (gltf_primitive.indices >= 0)
		{
			int accessor_index         = gltf_primitive.indices;
			primitive_data.index_count = to_u32(get_attribute_size(&model, accessor_index));
			primitive_data.index_type  = get_index_type(&model, accessor_index);
			primitive_data.indices     = get_attribute_data(&model, accessor_index);

			// Handle index format conversion if necessary
			if (primitive_data.index_type == vk::IndexType::eUint8EXT)
			{
				primitive_data.indices    = convert_indices_to_uint16(primitive_data.indices);
				primitive_data.index_type = vk::IndexType::eUint16;
			}
		}

		auto submesh = std::make_unique<sg::SubMesh>(primitive_data, device_);

		return submesh;
	}
	return nullptr;
}

sg::Scene GltfLoader::load_scene(std::string name, int scene_index)
{
	sg::Scene scene;

	scene.set_name(name);

	// Check extensions
	for (auto &used_extension : model_.extensionsUsed)
	{
		auto it = supported_extensions_.find(used_extension);

		// Check if extension isn't supported by the GLTFLoader
		if (it == supported_extensions_.end())
		{
			// If extension is required then we shouldn't allow the scene to be loaded
			if (std::ranges::find(model_.extensionsRequired, used_extension) != model_.extensionsRequired.end())
			{
				throw std::runtime_error("Cannot load glTF file. Contains a required unsupported extension: " + used_extension);
			}
			else
			{
				// Otherwise, if extension isn't required (but is in the file) then print a warning to the user
				LOGW("glTF file contains an unsupported extension, unexpected results may occur: {}", used_extension);
			}
		}
		else
		{
			// Extension is supported, so enable it
			LOGI("glTF file contains extension: {}", used_extension);
			it->second = true;
		}
	}

	// Load lights
	std::vector<std::unique_ptr<sg::Light>> light_components = parse_khr_lights_punctual();

	scene.set_components(std::move(light_components));

	// Load samplers
	std::vector<std::unique_ptr<sg::Sampler>>
	    sampler_components(model_.samplers.size());

	for (size_t sampler_index = 0; sampler_index < model_.samplers.size(); sampler_index++)
	{
		auto sampler                      = parse_sampler(model_.samplers[sampler_index]);
		sampler_components[sampler_index] = std::move(sampler);
	}

	scene.set_components(std::move(sampler_components));

	Timer timer;
	timer.start();

	// Load images
	auto thread_count = std::thread::hardware_concurrency();
	thread_count      = thread_count == 0 ? 1 : thread_count;
	ctpl::thread_pool thread_pool(thread_count);

	auto image_count = to_u32(model_.images.size());

	std::vector<std::future<std::unique_ptr<sg::Image>>> image_component_futures;

	auto srgb_flags = parse_srgb_requirements(model_);

	for (size_t image_index = 0; image_index < image_count; image_index++)
	{
		auto fut = thread_pool.push(
		    [this, image_index, &srgb_flags](size_t) {
			    auto image = parse_image(model_.images[image_index], srgb_flags[image_index]);

			    LOGI("Loaded gltf image #{} ({})", image_index, model_.images[image_index].uri.c_str());

			    return image;
		    });

		image_component_futures.push_back(std::move(fut));
	}

	std::vector<std::unique_ptr<sg::Image>> image_components;

	// Upload images to GPU. We do this in batches of 64MB of data to avoid needing
	// double the amount of memory (all the images and all the corresponding buffers).
	// This helps keep memory footprint lower which is helpful on smaller devices.
	size_t image_index = 0;
	while (image_index < image_count)
	{
		std::vector<backend::Buffer> transient_buffers;

		auto &command_buffer = device_.request_command_buffer();

		command_buffer.begin(vk::CommandBufferUsageFlagBits::eOneTimeSubmit, nullptr);

		size_t batch_size = 0;

		// Deal with 64MB of image data at a time to keep memory footprint low
		while (image_index < image_count && batch_size < 64 * 1024 * 1024)
		{
			// Wait for this image to complete loading, then stage for upload
			image_components.push_back(image_component_futures[image_index].get());

			auto &image = image_components[image_index];

			backend::Buffer stage_buffer = backend::Buffer::create_staging_buffer(device_, image->get_data());

			batch_size += image->get_data().size();

			upload_image_to_gpu(command_buffer, stage_buffer, *image);

			transient_buffers.push_back(std::move(stage_buffer));

			image_index++;
		}

		command_buffer.end();

		auto &queue = device_.get_queue_by_flags(vk::QueueFlagBits::eGraphics, 0);

		queue.submit(command_buffer, device_.request_fence());

		device_.get_fence_pool().wait();
		device_.get_fence_pool().reset();
		device_.get_command_pool().reset_pool();
		device_.wait_idle();

		// Remove the staging buffers for the batch we just processed
		transient_buffers.clear();
	}

	scene.set_components(std::move(image_components));

	auto elapsed_time = timer.stop();

	LOGI("Time spent loading images: {} seconds across {} threads.", xihe::to_string(elapsed_time), thread_count);

	// Load textures
	std::unique_ptr<sg::BindlessTextures> bindless_textures = std::make_unique<sg::BindlessTextures>("bindless_textures");

	auto images                  = scene.get_components<sg::Image>();
	auto samplers                = scene.get_components<sg::Sampler>();
	auto default_sampler_linear  = create_default_sampler(TINYGLTF_TEXTURE_FILTER_LINEAR);
	auto default_sampler_nearest = create_default_sampler(TINYGLTF_TEXTURE_FILTER_NEAREST);
	bool used_nearest_sampler    = false;

	for (auto &gltf_texture : model_.textures)
	{
		auto texture = parse_texture(gltf_texture);

		assert(gltf_texture.source < images.size());
		texture->set_image(*images[gltf_texture.source]);

		if (gltf_texture.sampler >= 0 && gltf_texture.sampler < static_cast<int>(samplers.size()))
		{
			texture->set_sampler(*samplers[gltf_texture.sampler]);
		}
		else
		{
			if (gltf_texture.name.empty())
			{
				gltf_texture.name = images[gltf_texture.source]->get_name();
			}

			// Get the properties for the image format. We'll need to check whether a linear sampler is valid.
			const vk::FormatProperties fmtProps = device_.get_gpu().get_format_properties(images[gltf_texture.source]->get_format());

			if (fmtProps.optimalTilingFeatures &
			    vk::FormatFeatureFlagBits::eSampledImageFilterLinear)
			{
				texture->set_sampler(*default_sampler_linear);
			}
			else
			{
				texture->set_sampler(*default_sampler_nearest);
				used_nearest_sampler = true;
			}
		}
		bindless_textures->add_texture(std::move(texture));
		//scene.add_component(std::move(texture));
	}

	/*scene.add_component(std::move(default_sampler_linear));
	if (used_nearest_sampler)
		scene.add_component(std::move(default_sampler_nearest));*/

	// Load materials
	/*bool                       has_textures = scene.has_component<sg::Texture>();
	std::vector<sg::Texture *> textures;
	if (has_textures)
	{
		textures = scene.get_components<sg::Texture>();
	}*/

	auto textures = bindless_textures->get_textures();

	for (auto &gltf_material : model_.materials)
	{
		auto material = parse_material(gltf_material);

		for (auto &gltf_value : gltf_material.values)
		{
			if (gltf_value.first.find("Texture") != std::string::npos)
			{
				std::string tex_name = to_snake_case(gltf_value.first);

				assert(gltf_value.second.TextureIndex() < textures.size());
				sg::Texture *tex = textures[gltf_value.second.TextureIndex()];

				material->textures[tex_name] = tex;

				material->set_texture_index(tex_name, gltf_value.second.TextureIndex());
			}
		}

		for (auto &gltf_value : gltf_material.additionalValues)
		{
			if (gltf_value.first.find("Texture") != std::string::npos)
			{
				std::string tex_name = to_snake_case(gltf_value.first);

				assert(gltf_value.second.TextureIndex() < textures.size());
				sg::Texture *tex = textures[gltf_value.second.TextureIndex()];

				material->textures[tex_name] = tex;

				material->set_texture_index(tex_name, gltf_value.second.TextureIndex());
			}
		}

		scene.add_component(std::move(material));
	}

	scene.add_component(std::move(bindless_textures));

	auto default_material = create_default_material();

	// Load meshes
	auto materials = scene.get_components<sg::PbrMaterial>();

#ifdef CALCULATE_NORMAL_MAP
	auto new_bindless_textures = scene.get_components<sg::BindlessTextures>();
	assert(new_bindless_textures.size() == 1);
	sg::BindlessTextures *bindless = new_bindless_textures[0];
#endif        // CALCULATE_NORMAL_MAP

	for (auto &gltf_mesh : model_.meshes)
	{
		auto mesh = parse_mesh(gltf_mesh);

		// Used to generate meshlets
		std::vector<float> vertex_positions;

		for (size_t i_primitive = 0; i_primitive < gltf_mesh.primitives.size(); i_primitive++)
		{
			const auto &gltf_primitive = gltf_mesh.primitives[i_primitive];
			MeshPrimitiveData primitive_data;
			primitive_data.name = fmt::format("'{}' mesh, primitive #{}", gltf_mesh.name, i_primitive);

			//auto submesh_name = fmt::format("'{}' mesh, primitive #{}", gltf_mesh.name, i_primitive);
			//auto submesh      = std::make_unique<sg::SubMesh>(std::move(submesh_name));

			for (auto &attribute : gltf_primitive.attributes)
			{
				VertexAttributeData attrib_data;
				std::string attrib_name = attribute.first;
				std::ranges::transform(attrib_name, attrib_name.begin(), ::tolower);

				int accessor_index = attribute.second;
				attrib_data.format = get_attribute_format(&model_, accessor_index);
				attrib_data.stride = to_u32(get_attribute_stride(&model_, accessor_index));
				attrib_data.data   = get_attribute_data(&model_, accessor_index);

				primitive_data.attributes[attrib_name] = std::move(attrib_data);

				if (attrib_name == "position")
				{
					primitive_data.vertex_count = to_u32(model_.accessors[accessor_index].count);
				}
			}

			if (gltf_primitive.indices >= 0)
			{
				int accessor_index         = gltf_primitive.indices;
				primitive_data.index_count = to_u32(get_attribute_size(&model_, accessor_index));
				primitive_data.index_type  = get_index_type(&model_, accessor_index);
				primitive_data.indices     = get_attribute_data(&model_, accessor_index);

				// Handle index format conversion if necessary
				if (primitive_data.index_type == vk::IndexType::eUint8EXT)
				{
					primitive_data.indices    = convert_indices_to_uint16(primitive_data.indices);
					primitive_data.index_type = vk::IndexType::eUint16;
				}

			}

			//auto submesh = std::make_unique<sg::SubMesh>(primitive_data, device_);
			/*auto mshader_mesh = std::make_unique<sg::MshaderMesh>(primitive_data, device_);*/

			sg::PbrMaterial *material = nullptr;
			if (gltf_primitive.material < 0)
			{
				material = default_material.get();
			}
			else
			{
				assert(gltf_primitive.material < materials.size());
				material = materials[gltf_primitive.material];
			}
			//submesh->set_material(*material);
			/*mshader_mesh->set_material(*material);*/

			//mesh->add_submesh(*submesh);

			//scene.add_component(std::move(submesh));

			/*mesh->add_mshader_mesh(*mshader_mesh);*/

		#ifdef CALCULATE_NORMAL_MAP
			std::vector<uint8_t> normal_map;
			uint32_t             width, height;

			fs::Path             scene_path = fs::path::get(fs::path::Type::kStorage) / scene.get_name();
			
			generate_normal_map(primitive_data, normal_map, width, height, scene_path.string());
			auto image = parse_image("normal_texture", normal_map, width, height);

			// upload image
			auto &command_buffer = device_.request_command_buffer();
			command_buffer.begin(vk::CommandBufferUsageFlagBits::eOneTimeSubmit, nullptr);
			backend::Buffer stage_buffer = backend::Buffer::create_staging_buffer(device_, image->get_data());
			upload_image_to_gpu(command_buffer, stage_buffer, *image);
			command_buffer.end();
			auto &queue = device_.get_queue_by_flags(vk::QueueFlagBits::eGraphics, 0);
			queue.submit(command_buffer, device_.request_fence());
			device_.get_fence_pool().wait();
			device_.get_fence_pool().reset();
			device_.get_command_pool().reset_pool();
			device_.wait_idle();

			auto texture = std::make_unique<sg::Texture>("normal_texture");
			texture->set_image(*image);
			texture->set_sampler(*default_sampler_linear);
			bindless->add_texture(std::move(texture));

			material->textures["normal_texture"] = bindless->get_textures().back();
			material->set_texture_index("normal_texture", textures.size() + i_primitive);

			scene.add_component(std::move(image));

		#endif

			mesh->add_submesh_data(*material, std::move(primitive_data));

			/*scene.add_component(std::move(mshader_mesh));*/
		}

		scene.add_component(std::move(mesh));
	}

	device_.get_fence_pool().wait();
	device_.get_fence_pool().reset();
	device_.get_command_pool().reset_pool();

#ifdef CALCULATE_NORMAL_MAP
	/*scene.clear_components<sg::BindlessTextures>();
	scene.add_component(std::unique_ptr<sg::BindlessTextures>(bindless));*/
#endif

	scene.add_component(std::move(default_sampler_linear));
	if (used_nearest_sampler)
		scene.add_component(std::move(default_sampler_nearest));

	scene.add_component(std::move(default_material));

	// Load cameras
	for (auto &gltf_camera : model_.cameras)
	{
		auto camera = parse_camera(gltf_camera);
		scene.add_component(std::move(camera));
	}

	// Load nodes
	auto meshes = scene.get_components<sg::Mesh>();

	std::vector<std::unique_ptr<sg::Node>> nodes;

	for (size_t node_index = 0; node_index < model_.nodes.size(); ++node_index)
	{
		auto gltf_node = model_.nodes[node_index];
		auto node      = parse_node(gltf_node, node_index);

		if (gltf_node.mesh >= 0)
		{
			assert(gltf_node.mesh < meshes.size());
			auto mesh = meshes[gltf_node.mesh];

			node->set_component(*mesh);

			mesh->add_node(*node);
		}

		if (gltf_node.camera >= 0)
		{
			auto cameras = scene.get_components<sg::Camera>();
			assert(gltf_node.camera < cameras.size());
			auto camera = cameras[gltf_node.camera];

			node->set_component(*camera);

			camera->set_node(*node);
		}

		if (auto extension = get_extension(gltf_node.extensions, KHR_LIGHTS_PUNCTUAL_EXTENSION))
		{
			auto lights      = scene.get_components<sg::Light>();
			int  light_index = extension->Get("light").Get<int>();
			assert(light_index < lights.size());
			auto light = lights[light_index];

			node->set_component(*light);

			light->set_node(*node);
		}

		nodes.push_back(std::move(node));
	}

	// Load scenes
	std::queue<std::pair<sg::Node &, int>> traverse_nodes;

	tinygltf::Scene *gltf_scene{nullptr};

	if (scene_index >= 0 && scene_index < static_cast<int>(model_.scenes.size()))
	{
		gltf_scene = &model_.scenes[scene_index];
	}
	else if (model_.defaultScene >= 0 && model_.defaultScene < static_cast<int>(model_.scenes.size()))
	{
		gltf_scene = &model_.scenes[model_.defaultScene];
	}
	else if (model_.scenes.size() > 0)
	{
		gltf_scene = &model_.scenes[0];
	}

	if (!gltf_scene)
	{
		throw std::runtime_error("Couldn't determine which scene to load!");
	}

	auto root_node = std::make_unique<sg::Node>(0, gltf_scene->name);

	for (auto node_index : gltf_scene->nodes)
	{
		traverse_nodes.push(std::make_pair(std::ref(*root_node), node_index));
	}

	while (!traverse_nodes.empty())
	{
		auto node_it = traverse_nodes.front();
		traverse_nodes.pop();

		assert(node_it.second < nodes.size());
		auto &current_node       = *nodes[node_it.second];
		auto &traverse_root_node = node_it.first;

		current_node.set_parent(traverse_root_node);
		traverse_root_node.add_child(current_node);

		for (auto child_node_index : model_.nodes[node_it.second].children)
		{
			traverse_nodes.push(std::make_pair(std::ref(current_node), child_node_index));
		}
	}

	scene.set_root_node(*root_node);
	nodes.push_back(std::move(root_node));

	// Store nodes into the scene
	scene.set_nodes(std::move(nodes));

	// Create node for the default camera
	auto camera_node = std::make_unique<sg::Node>(-1, "default_camera");

	auto default_camera = create_default_camera();
	default_camera->set_node(*camera_node);
	camera_node->set_component(*default_camera);
	scene.add_component(std::move(default_camera));

	scene.get_root_node().add_child(*camera_node);
	scene.add_node(std::move(camera_node));

	if (!scene.has_component<sg::Light>())
	{
		// Add a default light if none are present
		xihe::sg::add_directional_light(scene, glm::quat({glm::radians(-90.0f), 0.0f, glm::radians(30.0f)}));
	}

	return scene;
}

static void RandomColor(uint8_t *color)
{
	for (int i = 0; i < 3; i++)
		color[i] = uint8_t((rand() % 255 + 192) * 0.5f);
}

static void SetPixel(uint8_t *dest, int destWidth, int x, int y, const uint8_t *color)
{
	uint8_t *pixel = &dest[x * 4 + y * (destWidth * 4)];
	pixel[0]       = color[0];
	pixel[1]       = color[1];
	pixel[2]       = color[2];
	pixel[3]       = color[3];
}

static void RasterizeLine(uint8_t *dest, int destWidth, const int *p1, const int *p2, const uint8_t *color)
{
	const int dx = abs(p2[0] - p1[0]), sx = p1[0] < p2[0] ? 1 : -1;
	const int dy = abs(p2[1] - p1[1]), sy = p1[1] < p2[1] ? 1 : -1;
	int       err = (dx > dy ? dx : -dy) / 2;
	int       current[2];
	current[0] = p1[0];
	current[1] = p1[1];
	while (SetPixel(dest, destWidth, current[0], current[1], color), current[0] != p2[0] || current[1] != p2[1])
	{
		const int e2 = err;
		if (e2 > -dx)
		{
			err -= dy;
			current[0] += sx;
		}
		if (e2 < dy)
		{
			err += dx;
			current[1] += sy;
		}
	}
}

static void RasterizeTriangle(uint8_t *dest, int destWidth, const int *t0, const int *t1, const int *t2, const uint8_t *color)
{
	if (t0[1] > t1[1])
		std::swap(t0, t1);
	if (t0[1] > t2[1])
		std::swap(t0, t2);
	if (t1[1] > t2[1])
		std::swap(t1, t2);
	int total_height = t2[1] - t0[1];
	for (int i = 0; i < total_height; i++)
	{
		bool  second_half    = i > t1[1] - t0[1] || t1[1] == t0[1];
		int   segment_height = second_half ? t2[1] - t1[1] : t1[1] - t0[1];
		float alpha          = (float) i / total_height;
		float beta           = (float) (i - (second_half ? t1[1] - t0[1] : 0)) / segment_height;
		int   A[2], B[2];
		for (int j = 0; j < 2; j++)
		{
			A[j] = int(t0[j] + (t2[j] - t0[j]) * alpha);
			B[j] = int(second_half ? t1[j] + (t2[j] - t1[j]) * beta : t0[j] + (t1[j] - t0[j]) * beta);
		}
		if (A[0] > B[0])
			std::swap(A, B);
		for (int j = A[0]; j <= B[0]; j++)
			SetPixel(dest, destWidth, j, t0[1] + i, color);
	}
}

void RasterizeTriangle(
    uint8_t *imageData,
    uint32_t imageWidth,
    int v0[2], int v1[2], int v2[2],
    const glm::vec3 &n0,
    const glm::vec3 &n1,
    const glm::vec3 &n2)
{
	// Compute triangle bounding box
	int minX = std::min({v0[0], v1[0], v2[0]});
	int minY = std::min({v0[1], v1[1], v2[1]});
	int maxX = std::max({v0[0], v1[0], v2[0]});
	int maxY = std::max({v0[1], v1[1], v2[1]});

	// Clamp to image bounds
	minX = std::max(minX, 0);
	minY = std::max(minY, 0);
	maxX = std::min(maxX, static_cast<int>(imageWidth) - 1);
	maxY = std::min(maxY, static_cast<int>(imageWidth) - 1);

	// Triangle setup
	glm::vec2 p0(v0[0], v0[1]);
	glm::vec2 p1(v1[0], v1[1]);
	glm::vec2 p2(v2[0], v2[1]);

	float area = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
	if (area == 0.0f)
		return;

	float inv_area = 1.0f / area;

	for (int y = minY; y <= maxY; ++y)
	{
		for (int x = minX; x <= maxX; ++x)
		{
			glm::vec2 p(x + 0.5f, y + 0.5f);

			// Compute barycentric coordinates
			float w0 = (p1.x - p.x) * (p2.y - p.y) - (p2.x - p.x) * (p1.y - p.y);
			float w1 = (p2.x - p.x) * (p0.y - p.y) - (p0.x - p.x) * (p2.y - p.y);
			float w2 = (p0.x - p.x) * (p1.y - p.y) - (p1.x - p.x) * (p0.y - p.y);

			w0 *= inv_area;
			w1 *= inv_area;
			w2 *= inv_area;

			if (w0 >= 0 && w1 >= 0 && w2 >= 0)
			{
				// Interpolate normal
				glm::vec3 n = glm::normalize(w0 * n0 + w1 * n1 + w2 * n2);

				// Encode to 8-bit RGB: world-space normal in [-1,1] ¡ú [0,255]
				uint8_t r = static_cast<uint8_t>((n.x * 0.5f + 0.5f) * 255.0f);
				uint8_t g = static_cast<uint8_t>((n.y * 0.5f + 0.5f) * 255.0f);
				uint8_t b = static_cast<uint8_t>((n.z * 0.5f + 0.5f) * 255.0f);
				uint8_t a = 255;

				uint32_t index       = (y * imageWidth + x) * 4;
				imageData[index + 0] = r;
				imageData[index + 1] = g;
				imageData[index + 2] = b;
				imageData[index + 3] = a;
			}
		}
	}
}

void GltfLoader::generate_normal_map(MeshPrimitiveData &primitive_data, std::vector<uint8_t> &normal_map, uint32_t &width, uint32_t &height, std::string path)
{
	xatlas::Atlas   *atlas = xatlas::Create();
	
	xatlas::MeshDecl meshDecl;

	auto vertex_positions = reinterpret_cast<const float *>(primitive_data.attributes.at("position").data.data());
	auto vertex_normals   = reinterpret_cast<const float *>(primitive_data.attributes.at("normal").data.data());
	const float *vertex_uvs = nullptr;
	if (primitive_data.attributes.find("texcoord_0") != primitive_data.attributes.end())
	{
		vertex_uvs = reinterpret_cast<const float *>(primitive_data.attributes.at("texcoord_0").data.data());
	}

	meshDecl.vertexCount          = primitive_data.vertex_count;
	meshDecl.vertexPositionData   = vertex_positions;
	meshDecl.vertexPositionStride = sizeof(float) * 3;
	meshDecl.vertexNormalData     = vertex_normals;
	meshDecl.vertexNormalStride   = sizeof(float) * 3;

	std::vector<uint8_t> uv_data(primitive_data.vertex_count * sizeof(float) * 2);

	if (vertex_uvs)
	{
		meshDecl.vertexUvData   = vertex_uvs;
		meshDecl.vertexUvStride = sizeof(float) * 2;
	}

	meshDecl.indexCount  = primitive_data.index_count;
	meshDecl.indexData   = primitive_data.indices.data();
	if (primitive_data.index_type == vk::IndexType::eUint16)
	{
		meshDecl.indexFormat = xatlas::IndexFormat::UInt16;
	}
	else if (primitive_data.index_type == vk::IndexType::eUint32)
	{
		meshDecl.indexFormat = xatlas::IndexFormat::UInt32;
	}

	xatlas::AddMeshError error = xatlas::AddMesh(atlas, meshDecl, 1);
	if (error != xatlas::AddMeshError::Success)
	{
		xatlas::Destroy(atlas);
		LOGE("error");
		return;
	}

	xatlas::AddMeshJoin(atlas);
	xatlas::Generate(atlas);

	if (atlas->width == 0 || atlas->height == 0)
		return;

	const uint32_t       image_data_size = atlas->width * atlas->height * 4;
	normal_map.resize(atlas->atlasCount * image_data_size, 128);        // default normal: (0,0,1)

	for (uint32_t m = 0; m < atlas->meshCount; m++)
	{
		const xatlas::Mesh &mesh           = atlas->meshes[m];

		for (uint32_t i = 0; i < mesh.vertexCount; i++)
		{
			const xatlas::Vertex &v  = mesh.vertexArray[i];
			float                 u_ = v.uv[0] / float(atlas->width);
			float                 v_ = v.uv[1] / float(atlas->height);

			// Write to uv_data
			std::memcpy(&uv_data[v.xref * sizeof(float) * 2], &u_, sizeof(float));
			std::memcpy(&uv_data[v.xref * sizeof(float) * 2 + sizeof(float)], &v_, sizeof(float));
		}

		const uint32_t      faceCount      = mesh.indexCount / 3;
		uint32_t            faceFirstIndex = 0;

		for (uint32_t f = 0; f < faceCount; ++f)
		{
			int       verts_uv[3][2];
			glm::vec2 uv[3];
			glm::vec3 pos[3];
			glm::vec3 normal[3];

			for (int j = 0; j < 3; ++j)
			{
				const uint32_t        index = mesh.indexArray[faceFirstIndex + j];
				const xatlas::Vertex &v     = mesh.vertexArray[index];

				verts_uv[j][0] = int(v.uv[0]);
				verts_uv[j][1] = int(v.uv[1]);
				uv[j]          = glm::vec2(v.uv[0] / float(atlas->width), v.uv[1] / float(atlas->height));

				const float *pos_ptr = vertex_positions + v.xref * 3;
				const float *nrm_ptr = vertex_normals + v.xref * 3;
				pos[j]               = glm::make_vec3(pos_ptr);
				normal[j]            = glm::normalize(glm::make_vec3(nrm_ptr));
			}

			uint8_t *imageData = &normal_map[0];

			RasterizeTriangle(
			    imageData,
			    atlas->width,
			    verts_uv[0], verts_uv[1], verts_uv[2],
			    normal[0], normal[1], normal[2]);

			faceFirstIndex += 3;
		}
	}

	VertexAttributeData attrib_data;
	attrib_data.format                      = vk::Format::eR32G32Sfloat;
	attrib_data.stride                      = sizeof(float) * 2;
	attrib_data.data                        = uv_data;
	primitive_data.attributes["texcoord_0"] = std::move(attrib_data);

	// Save normal map
	fs::Path output_dir(path);
	for (uint32_t i = 0; i < atlas->atlasCount; i++)
	{
		std::string name      = std::format("{}_normal{:02}.tga", primitive_data.name, i);
		fs::Path    full_path = output_dir / name;
		stbi_write_tga(full_path.string().c_str(), atlas->width, atlas->height, 4, &normal_map[i * image_data_size]);
	}

	width = atlas->width;
	height = atlas->height;

	xatlas::Destroy(atlas);
}

std::unique_ptr<sg::Node> GltfLoader::parse_node(const tinygltf::Node &gltf_node, size_t index) const
{
	auto node = std::make_unique<sg::Node>(index, gltf_node.name);

	auto &transform = node->get_component<sg::Transform>();

	if (!gltf_node.translation.empty())
	{
		glm::vec3 translation;

		std::ranges::transform(gltf_node.translation, glm::value_ptr(translation), TypeCast<double, float>{});

		transform.set_translation(translation);
	}

	if (!gltf_node.rotation.empty())
	{
		glm::quat rotation;

		std::ranges::transform(gltf_node.rotation, glm::value_ptr(rotation), TypeCast<double, float>{});

		transform.set_rotation(rotation);
	}

	if (!gltf_node.scale.empty())
	{
		glm::vec3 scale;

		std::ranges::transform(gltf_node.scale, glm::value_ptr(scale), TypeCast<double, float>{});

		transform.set_scale(scale);
	}

	if (!gltf_node.matrix.empty())
	{
		glm::mat4 matrix;

		std::ranges::transform(gltf_node.matrix, glm::value_ptr(matrix), TypeCast<double, float>{});

		transform.set_matrix(matrix);
	}

	return node;
}

std::unique_ptr<sg::Camera> GltfLoader::parse_camera(const tinygltf::Camera &gltf_camera) const
{
	std::unique_ptr<sg::Camera> camera;

	if (gltf_camera.type == "perspective")
	{
		auto perspective_camera = std::make_unique<sg::PerspectiveCamera>(gltf_camera.name);

		perspective_camera->set_aspect_ratio(static_cast<float>(gltf_camera.perspective.aspectRatio));
		perspective_camera->set_field_of_view(static_cast<float>(gltf_camera.perspective.yfov));
		perspective_camera->set_near_plane(static_cast<float>(gltf_camera.perspective.znear));
		perspective_camera->set_far_plane(static_cast<float>(gltf_camera.perspective.zfar));

		camera = std::move(perspective_camera);
	}
	else
	{
		LOGW("Camera type not supported");
	}

	return camera;
}

std::unique_ptr<sg::Mesh> GltfLoader::parse_mesh(const tinygltf::Mesh &gltf_mesh) const
{
	return std::make_unique<sg::Mesh>(gltf_mesh.name);
}

std::unique_ptr<sg::PbrMaterial> GltfLoader::parse_material(const tinygltf::Material &gltf_material) const
{
	auto material = std::make_unique<sg::PbrMaterial>(gltf_material.name);

	material->base_color_factor = glm::vec4(1.0f);

	for (auto &gltf_value : gltf_material.values)
	{
		if (gltf_value.first == "baseColorFactor")
		{
			const auto &color_factor    = gltf_value.second.ColorFactor();
			material->base_color_factor = glm::vec4(color_factor[0], color_factor[1], color_factor[2], color_factor[3]);
		}
		else if (gltf_value.first == "metallicFactor")
		{
			material->metallic_factor = static_cast<float>(gltf_value.second.Factor());
		}
		else if (gltf_value.first == "roughnessFactor")
		{
			material->roughness_factor = static_cast<float>(gltf_value.second.Factor());
		}
	}

	for (auto &gltf_value : gltf_material.additionalValues)
	{
		if (gltf_value.first == "emissiveFactor")
		{
			const auto &emissive_factor = gltf_value.second.number_array;

			material->emissive = glm::vec3(emissive_factor[0], emissive_factor[1], emissive_factor[2]);
		}
		else if (gltf_value.first == "alphaMode")
		{
			if (gltf_value.second.string_value == "BLEND")
			{
				material->alpha_mode = sg::AlphaMode::kBlend;
			}
			else if (gltf_value.second.string_value == "OPAQUE")
			{
				material->alpha_mode = sg::AlphaMode::kOpaque;
			}
			else if (gltf_value.second.string_value == "MASK")
			{
				material->alpha_mode = sg::AlphaMode::kMask;
			}
		}
		else if (gltf_value.first == "alphaCutoff")
		{
			material->alpha_cutoff = static_cast<float>(gltf_value.second.number_value);
		}
		else if (gltf_value.first == "doubleSided")
		{
			material->double_sided = gltf_value.second.bool_value;
		}
	}

	return material;
}

std::unique_ptr<sg::Image> GltfLoader::parse_image(tinygltf::Image &gltf_image, bool is_srgb) const
{
	std::unique_ptr<sg::Image> image{nullptr};

	if (!gltf_image.image.empty())
	{
		// Image embedded in gltf file
		auto mipmap = sg::Mipmap{
		    /* .level = */ 0,
		    /* .offset = */ 0,
		    /* .extent = */ {/* .width = */ static_cast<uint32_t>(gltf_image.width),
		                     /* .height = */ static_cast<uint32_t>(gltf_image.height),
		                     /* .depth = */ 1u}};
		std::vector<sg::Mipmap> mipmaps{mipmap};
		image = std::make_unique<sg::Image>(gltf_image.name, std::move(gltf_image.image), std::move(mipmaps));
	}
	else
	{
		// Load image from uri
		auto image_uri = model_path_ + "/" + gltf_image.uri;
		image          = sg::Image::load(gltf_image.name, image_uri, sg::Image::kUnknown);
	}


	// Check whether the format is supported by the GPU
	 if (sg::is_astc(image->get_format()))
	{
		if (!device_.is_image_format_supported(image->get_format()))
		{
			LOGW("ASTC not supported: decoding {}", image->get_name());
			image = std::make_unique<sg::Astc>(*image);
			image->generate_mipmaps();
		}
	 }

	if (is_srgb)
	{
		image->coerce_format_to_srgb();
	}

	image->create_vk_image(device_);

	return image;
}

std::unique_ptr<sg::Image> GltfLoader::parse_image(std::string name, std::vector<uint8_t> &data, uint32_t width, uint32_t height) const
{
	std::unique_ptr<sg::Image> image{nullptr};

	auto mipmap = sg::Mipmap{
	    /* .level = */ 0,
	    /* .offset = */ 0,
	    /* .extent = */ {/* .width = */ width,
	                     /* .height = */ height,
	                     /* .depth = */ 1u}};
	std::vector<sg::Mipmap> mipmaps{mipmap};
	image = std::make_unique<sg::Image>(name, std::move(data), std::move(mipmaps));

	image->create_vk_image(device_);

	return image;
}

std::unique_ptr<sg::Sampler> GltfLoader::parse_sampler(const tinygltf::Sampler &gltf_sampler) const
{
	auto name = gltf_sampler.name;

	vk::Filter             min_filter     = find_min_filter(gltf_sampler.minFilter);
	vk::Filter             mag_filter     = find_mag_filter(gltf_sampler.magFilter);
	vk::SamplerMipmapMode  mipmap_mode    = find_mipmap_mode(gltf_sampler.minFilter);
	vk::SamplerAddressMode address_mode_u = find_wrap_mode(gltf_sampler.wrapS);
	vk::SamplerAddressMode address_mode_v = find_wrap_mode(gltf_sampler.wrapT);
	// vk::SamplerAddressMode address_mode_w = find_wrap_mode(gltf_sampler.wrapR);
	vk::SamplerCreateInfo  sampler_info{};

	sampler_info.magFilter    = mag_filter;
	sampler_info.minFilter    = min_filter;
	sampler_info.mipmapMode   = mipmap_mode;
	sampler_info.addressModeU = address_mode_u;
	sampler_info.addressModeV = address_mode_v;
	// sampler_info.addressModeW = address_mode_w;
	sampler_info.borderColor  = vk::BorderColor::eFloatOpaqueWhite;
	sampler_info.maxLod       = std::numeric_limits<float>::max();

	backend::Sampler vk_sampler{device_, sampler_info};
	vk_sampler.set_debug_name(gltf_sampler.name);

	return std::make_unique<sg::Sampler>(name, std::move(vk_sampler));
}

std::unique_ptr<sg::Texture> GltfLoader::parse_texture(const tinygltf::Texture &gltf_texture) const
{
	return std::make_unique<sg::Texture>(gltf_texture.name);
}

std::unique_ptr<sg::PbrMaterial> GltfLoader::create_default_material()
{
	tinygltf::Material gltf_material;
	return parse_material(gltf_material);
}

std::unique_ptr<sg::Sampler> GltfLoader::create_default_sampler(int filter)
{
	tinygltf::Sampler gltf_sampler;

	gltf_sampler.minFilter = filter;
	gltf_sampler.magFilter = filter;

	gltf_sampler.wrapS = TINYGLTF_TEXTURE_WRAP_REPEAT;
	gltf_sampler.wrapT = TINYGLTF_TEXTURE_WRAP_REPEAT;
	//gltf_sampler.wrapR = TINYGLTF_TEXTURE_WRAP_REPEAT;

	return parse_sampler(gltf_sampler);
}

std::unique_ptr<sg::Camera> GltfLoader::create_default_camera()
{
	tinygltf::Camera gltf_camera;

	gltf_camera.name = "default_camera";
	gltf_camera.type = "perspective";

	gltf_camera.perspective.aspectRatio = 1.77f;
	gltf_camera.perspective.yfov        = 1.0f;
	gltf_camera.perspective.znear       = 0.1f;
	gltf_camera.perspective.zfar        = 1000.0f;

	return parse_camera(gltf_camera);
}

tinygltf::Value *GltfLoader::get_extension(tinygltf::ExtensionMap &tinygltf_extensions, const std::string &extension)
{
	auto it = tinygltf_extensions.find(extension);
	if (it != tinygltf_extensions.end())
	{
		return &it->second;
	}
	else
	{
		return nullptr;
	}
}

std::vector<std::unique_ptr<sg::Light>> GltfLoader::parse_khr_lights_punctual()
{
	if (is_extension_enabled(KHR_LIGHTS_PUNCTUAL_EXTENSION))
	{
		if (!model_.extensions.contains(KHR_LIGHTS_PUNCTUAL_EXTENSION) || !model_.extensions.at(KHR_LIGHTS_PUNCTUAL_EXTENSION).Has("lights"))
		{
			return {};
		}
		auto &khr_lights = model_.extensions.at(KHR_LIGHTS_PUNCTUAL_EXTENSION).Get("lights");

		std::vector<std::unique_ptr<sg::Light>> light_components(khr_lights.ArrayLen());

		for (size_t light_index = 0; light_index < khr_lights.ArrayLen(); ++light_index)
		{
			auto &khr_light = khr_lights.Get(static_cast<int>(light_index));

			// Spec states a light has to have a type to be valid
			if (!khr_light.Has("type"))
			{
				LOGE("KHR_lights_punctual extension: light {} doesn't have a type!", light_index);
				throw std::runtime_error("Couldn't load glTF file, KHR_lights_punctual extension is invalid");
			}

			auto light = std::make_unique<sg::Light>(khr_light.Get("name").Get<std::string>());

			sg::LightType       type;
			sg::LightProperties properties;

			// Get type
			auto &gltf_light_type = khr_light.Get("type").Get<std::string>();
			if (gltf_light_type == "point")
			{
				type = sg::LightType::kPoint;
			}
			else if (gltf_light_type == "spot")
			{
				type = sg::LightType::kSpot;
			}
			else if (gltf_light_type == "directional")
			{
				type = sg::LightType::kDirectional;
			}
			else
			{
				LOGE("KHR_lights_punctual extension: light type '{}' is invalid", gltf_light_type);
				throw std::runtime_error("Couldn't load glTF file, KHR_lights_punctual extension is invalid");
			}

			// Get properties
			if (khr_light.Has("color"))
			{
				properties.color = glm::vec3(
				    static_cast<float>(khr_light.Get("color").Get(0).Get<double>()),
				    static_cast<float>(khr_light.Get("color").Get(1).Get<double>()),
				    static_cast<float>(khr_light.Get("color").Get(2).Get<double>()));
			}

			if (khr_light.Has("intensity"))
			{
				properties.intensity = static_cast<float>(khr_light.Get("intensity").Get<double>());
			}

			if (type != sg::LightType::kDirectional)
			{
				properties.range = static_cast<float>(khr_light.Get("range").Get<double>());
				if (type != sg::LightType::kPoint)
				{
					if (!khr_light.Has("spot"))
					{
						LOGE("KHR_lights_punctual extension: spot light doesn't have a 'spot' property set", gltf_light_type);
						throw std::runtime_error("Couldn't load glTF file, KHR_lights_punctual extension is invalid");
					}

					properties.inner_cone_angle = static_cast<float>(khr_light.Get("spot").Get("innerConeAngle").Get<double>());

					if (khr_light.Get("spot").Has("outerConeAngle"))
					{
						properties.outer_cone_angle = static_cast<float>(khr_light.Get("spot").Get("outerConeAngle").Get<double>());
					}
					else
					{
						// Spec states default value is PI/4
						properties.outer_cone_angle = glm::pi<float>() / 4.0f;
					}
				}
			}
			else if (type == sg::LightType::kDirectional || type == sg::LightType::kSpot)
			{
				// The spec states that the light will inherit the transform of the node.
				// The light's direction is defined as the 3-vector (0.0, 0.0, -1.0) and
				// the rotation of the node orients the light accordingly.
				properties.direction = glm::vec3(0.0f, 0.0f, -1.0f);
			}

			light->set_light_type(type);
			light->set_properties(properties);

			light_components[light_index] = std::move(light);
		}

		return light_components;
	}
	else
	{
		return {};
	}
}

bool GltfLoader::is_extension_enabled(const std::string &requested_extension)
{
	auto it = supported_extensions_.find(requested_extension);
	if (it != supported_extensions_.end())
	{
		return it->second;
	}
	else
	{
		return false;
	}
}
}        // namespace xihe