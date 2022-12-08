#version 460

layout (location = 0) in vec3 fNormal;
layout (location = 1) in vec3 fColor;

//output write
layout (location = 0) out vec4 outFragColor;

layout(set = 0, binding = 1) uniform  SceneData{
	vec4 ambientColor;
	vec4 sunlightDirection; //w for sun power
	vec4 sunlightColor;
} sceneData;

void main()
{
	//return red
	outFragColor = vec4(fColor + sceneData.ambientColor.xyz, 1.0f);
}