struct Meshlet
{
	uint  vertex_offset;
	uint  triangle_offset;
	uint  vertex_count;
	uint  triangle_count;
	vec3  center;
	float radius;
	vec3  cone_axis;
	float cone_cutoff;

	uint  mesh_draw_index;
	uint  padding[3];

	uint vertex_page_index1;
	uint vertex_page_index2;
	uint triangle_page_index1;
	uint triangle_page_index2;

	// lod
	vec3 cone_apex;
	uint lod;

	float parent_error;
	float cluster_error;
	float pdd1;
	float pdd2;

	vec4 parent_bounding_sphere;
};

struct Cluster
{
	uint cluster_group_index;
	uint cluster_index;
	uint mesh_draw_index;
	uint padding;

	// for culling

	vec4  bounding_sphere;
	vec3  cone_axis;
	float cone_cutoff;
};

struct ClusterGroup
{
	uint page_index;
	uint page_offset;
	uint size;
	uint offset;

	uint vertices_offset;
	uint vertex_indices_offset;
	uint triangles_offset;
	uint meshlets_offset;

	// for lod

	uint  lod;
	float cluster_error;
	float parent_error;
	uint  padding2;

	vec4 bounding_sphere;
	vec4 parent_bounding_sphere;
};

struct MeshDraw
{
	// x = diffuse index, y = roughness index, z = normal index, w = occlusion index.
	// Occlusion and roughness are encoded in the same texture
	uvec4 texture_indices;
	vec4  base_color_factor;
	vec4  metallic_roughness_occlusion_factor;

	uint meshlet_offset;
	uint meshlet_count;
	uint mesh_vertex_offset;
	uint mesh_triangle_offset;
};

struct MeshLoDDraw
{
	// x = diffuse index, y = roughness index, z = normal index, w = occlusion index.
	// Occlusion and roughness are encoded in the same texture
	uvec4 texture_indices;
	vec4  base_color_factor;
	vec4  metallic_roughness_occlusion_factor;

	uint cluster_offset;
	uint cluster_count;
};


struct MeshInstanceDraw
{
	mat4 model;
	mat4 model_inverse;

	uint mesh_draw_index;
	uint padding[3];
};

struct MeshDrawCommand
{
	//// VkDrawIndexedIndirectCommand
	// uint index_count;
	// uint instance_count;
	// uint first_index;
	// uint vertex_offset;
	// uint first_instance;

	// VkDrawMeshTasksIndirectCommandEXT
	uint group_count_x;
	uint group_count_y;
	uint group_count_z;

	uint instance_index;
};