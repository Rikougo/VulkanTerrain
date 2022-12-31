#version 460

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec2 inTexCoord;
layout (location = 2) in float inHeight;

//output write
layout (location = 0) out vec4 outFragColor;

layout(set = 0, binding = 1) uniform  SceneData{
	vec4 ambientColor;
	vec4 sunlightDirection; //w for sun power
	vec4 sunlightColor;
    float terrainSubdivision;
} sceneData;

void main()
{
	float diff = max(dot(inNormal, sceneData.sunlightDirection.xyz), 0.0);
    vec3 diffuse = diff * sceneData.sunlightColor.rgb;

    vec3 ambient = 0.1f * sceneData.sunlightColor.rgb;

    vec4 color = vec4(inHeight, inHeight, inHeight, 1.0f);
    // outFragColor = vec4((diffuse + ambient) * color.rgb, 1.0);
    outFragColor = color;
}