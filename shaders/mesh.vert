#version 450

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec3 vColor;

layout (location = 0) out vec3 fNormal;
layout (location = 1) out vec3 fColor;

/*layout(set = 0, binding = 0) uniform  CameraBuffer {
	mat4 view;
	mat4 proj;
	mat4 viewproj;
} cameraData;*/

layout( push_constant ) uniform constants
{
	vec4 data;
	mat4 render_matrix;
} PushConstants;


void main()
{
	mat4 transformMatrix = (PushConstants.render_matrix);
	gl_Position = transformMatrix * vec4(vPosition, 1.0f);
	fColor = vColor;
	fNormal = vec3(1.0f, 0.0f, 0.0f);
}
