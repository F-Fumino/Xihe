#version 460

// #define HAS_BASE_COLOR_TEXTURE
// #define HAS_NORMAL_TEXTURE

precision highp float;

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_debug_printf : require

layout (location = 0) in PerVertexData
{
  vec4 pos;
  vec3 normal;
  vec2 uv;
  flat uint mesh_draw_index;
  flat uint instance_index;
} v_in;

layout (location = 0) out vec4 o_albedo;
layout (location = 1) out vec4 o_normal;

#include "mesh_shading/mesh.h"

layout (std430, binding = 3) readonly buffer MeshDrawBuffer
{
    MeshLoDDraw mesh_draws[];
};

layout(std430, binding = 4) readonly buffer MeshInstanceDrawBuffer
{
    MeshInstanceDraw instances[];
};

layout (set = 1, binding = 10 ) uniform sampler2D global_textures[];

void main(void)
{

#ifdef HAS_NORMAL_TEXTURE
    vec4 normal_map = texture(global_textures[nonuniformEXT(mesh_draws[v_in.mesh_draw_index].texture_indices.z)], v_in.uv);
    vec3 normal = 2 * normal_map.xyz - 1;
    mat4 model = instances[v_in.instance_index].model;
    normal = normalize(mat3(model) * normal);
    o_normal = vec4(0.5 * normal + 0.5, 1.0);
#else 
    vec3 normal = normalize(v_in.normal);
    // Transform normals from [-1, 1] to [0, 1]
    o_normal = vec4(0.5 * normal + 0.5, 1.0);
#endif

	vec4 base_color = vec4(1.0, 0.0, 0.0, 1.0);
#ifdef HAS_BASE_COLOR_TEXTURE
    base_color = texture(global_textures[nonuniformEXT(mesh_draws[v_in.mesh_draw_index].texture_indices.x)], v_in.uv);
#else
    base_color = mesh_draws[v_in.mesh_draw_index].base_color_factor;
#endif
    o_albedo = base_color;
#if defined(SHOW_MESHLET_VIEW) || defined(SHOW_LOD_VIEW)
    o_albedo = v_in.pos;
#endif
}
