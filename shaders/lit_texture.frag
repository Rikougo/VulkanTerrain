#version 460

layout (location = 0) in vec3 fNormal;
layout (location = 1) in vec3 fColor;
layout (location = 2) in vec2 fTexCoord;

//output write
layout (location = 0) out vec4 outFragColor;

layout(set = 0, binding = 1) uniform  SceneData{
	vec4 ambientColor;
	vec4 sunlightDirection; //w for sun power
	vec4 sunlightColor;
    float terrainSubdivision;
    float minDistance;
    float maxDistance;
	vec3 clickedPoint;
    uint viewMode;
} sceneData;
layout(set = 2, binding = 0) uniform sampler2D texture0;

void main()
{
	float diff = max(dot(fNormal, sceneData.sunlightDirection.xyz), 0.0);
    vec3 diffuse = diff * sceneData.sunlightColor.rgb;

    vec3 ambient = 0.1f * sceneData.sunlightColor.rgb;

    vec4 color = vec4(texture(texture0, fTexCoord).xyz, 1.0f);
    outFragColor = vec4((diffuse + ambient) * color.rgb, 1.0);
}