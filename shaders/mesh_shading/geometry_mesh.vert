#version 450
#extension GL_ARB_shader_draw_parameters : require
#extension GL_EXT_debug_printf : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

#define PAGE_SIZE (1 * 1024 * 1024)
#define MAX_LOD 10.0

layout(set = 0, binding = 1) uniform GlobalUniform {
    mat4 view;
    mat4 view_proj;
    vec4 frustum_planes[6];
    vec3 camera_position;
} global_uniform;

#include "mesh_shading/mesh.h"

layout(std430, binding = 2) readonly buffer IndirectCommandBuffer {
    IndirectCommand indirect_commands[]; 
};

layout(std430, binding = 3) readonly buffer MeshInstanceDrawBuffer {
    MeshInstanceDraw instances[];
};

layout (buffer_reference, std430) buffer _scene_data
{
    uint scene_data[];
};

layout (std430, binding = 4) readonly buffer SceneDataBufferAddressList
{
    _scene_data scene_data_buffer_addresses[];
};

layout(buffer_reference, std430) buffer _cluster_data {
    Cluster clusters[];
};

layout (std430, binding = 5) readonly buffer ClusterBufferAddress
{
    _cluster_data cluster_buffer_address;
};

layout(std430, binding = 6) readonly buffer ClusterGroupBuffer {
    ClusterGroup cluster_groups[];
};

layout(location = 0) out vec4 vPos;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec2 vUV;
layout(location = 3) flat out uint vMeshDrawIndex;
layout(location = 4) flat out uint vInstanceIndex;

void main()
{
    uint command_index = gl_InstanceIndex;

    // debugPrintfEXT("command_index: %d\n", command_index);

    IndirectCommand command = indirect_commands[command_index];

    uint instance_index = command.instance_index;
    uint cluster_index  = command.cluster_index;

    MeshInstanceDraw instance = instances[instance_index];
    mat4 model = instance.model;

    _cluster_data cd = cluster_buffer_address;
    Cluster cluster  = cd.clusters[cluster_index];
    ClusterGroup cluster_group = cluster_groups[cluster.cluster_group_index];

    vec4 cluster_color;

#ifdef SHOW_MESHLET_VIEW
    float min_value = 0.1;
    cluster_color = vec4(
        max(float((cluster_index * 37) % 255) / 255.0, min_value),
        max(float((cluster_index * 73) % 255) / 255.0, min_value),
        max(float((cluster_index * 151) % 255) / 255.0, min_value),
        1.0
    );
#elif defined(SHOW_LOD_VIEW)
    float lodFactor = clamp(cluster_group.lod / MAX_LOD, 0.0, 1.0);
    cluster_color = vec4(
        lodFactor,   
        0.0,            
        1.0 - lodFactor, 
        1.0              
    );
#endif

    uint buffer_index = cluster_group.page_index;
    uint page_offset  = cluster_group.page_offset;

    _scene_data sdb = scene_data_buffer_addresses[buffer_index];

    // if (sdb.scene_data == 0)
    // {
    //     return;
    // }

    uint vertex_offset = sdb.scene_data[page_offset + cluster_group.meshlets_offset + cluster.cluster_index * 4 + 0];
    uint vertex_count  = sdb.scene_data[page_offset + cluster_group.meshlets_offset + cluster.cluster_index * 4 + 1];

    uint vertices_offset = page_offset + cluster_group.vertices_offset;
    uint vertex_indices_offset = page_offset + cluster_group.vertex_indices_offset;

    // debugPrintfEXT("command.vertex_offset: %d\n", command.vertex_offset);
    // debugPrintfEXT("gl_VertexIndex: %d\n", gl_VertexIndex);
    // debugPrintfEXT("command.global_vertex_offset: %d\n", command.global_vertex_offset);
    uint local_vertex_idx = uint(gl_VertexIndex) - uint(command.vertex_offset);
    // debugPrintfEXT("local_vertex_idx: %d\n", local_vertex_idx);
    // uint vertex_index = sdb.scene_data[vertex_indices_offset + vertex_offset + gl_VertexIndex - command.vertex_offset];
    // int raw = int(gl_VertexIndex) - command.vertex_offset;

    if (local_vertex_idx < 0 || local_vertex_idx >= vertex_count)
    {
        debugPrintfEXT("gl_VertexIndex: %d, command: %d, vertex index out of range: %d, vertex count: %d\n", gl_VertexIndex, command.vertex_offset, local_vertex_idx, vertex_count);
	    return;
	}

    // uint vertex_index = sdb.scene_data[vertex_indices_offset + vertex_offset];
    uint vertex_index = sdb.scene_data[vertex_indices_offset + vertex_offset + local_vertex_idx];

    vec4 pos = vec4(
        uintBitsToFloat(sdb.scene_data[vertices_offset + vertex_index * 8 + 0]),
        uintBitsToFloat(sdb.scene_data[vertices_offset + vertex_index * 8 + 1]),
        uintBitsToFloat(sdb.scene_data[vertices_offset + vertex_index * 8 + 2]),
        1.0
    );

    vec3 normal = vec3(
        uintBitsToFloat(sdb.scene_data[vertices_offset + vertex_index * 8 + 4]),
        uintBitsToFloat(sdb.scene_data[vertices_offset + vertex_index * 8 + 5]),
        uintBitsToFloat(sdb.scene_data[vertices_offset + vertex_index * 8 + 6])
    );

    vec2 uv = vec2(
        uintBitsToFloat(sdb.scene_data[vertices_offset + vertex_index * 8 + 3]),
        uintBitsToFloat(sdb.scene_data[vertices_offset + vertex_index * 8 + 7])
    );

    vPos    = model * pos;
    vNormal = mat3(model) * normal;
    vUV     = uv;

    vMeshDrawIndex = instance.mesh_draw_index;
    vInstanceIndex = instance_index;

    gl_Position = global_uniform.view_proj * vPos;

#if defined(SHOW_LOD_VIEW) || defined(SHOW_MESHLET_VIEW)
    vPos = cluster_color;
#endif
}