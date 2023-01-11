#version 460

#define SELECTION_SIZE 0.5f

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec2 inTexCoord;
layout (location = 2) in float inHeight;
layout (location = 3) in vec4 inPosition;

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
    bool useLightning;
} sceneData;

float fog(float density)
{
	const float LOG2 = -1.442695;
	float dist = gl_FragCoord.z / gl_FragCoord.w * 0.1;
	float d = density * dist;
	return 1.0 - clamp(exp2(d * d * LOG2), 0.0, 1.0);
}

void main()
{
	vec4 color = !sceneData.useLightning ? vec4(inNormal.xyz, 1.0f) : vec4(0.75f, 0.75f, 0.75f, 1.0f);

	bool selected = length(inPosition.xyz - sceneData.clickedPoint) < 0.25f;
	if (selected) {
		color = mix(vec4(0.2f, 0.0f, 0.8f, 1.0f), color, 1.0f - length(inPosition.xyz - sceneData.clickedPoint) * 5.0f);
	}

    const vec4 fogColor = vec4(0.0f, 0.0f, 0.0f, 0.0);

	if (sceneData.useLightning) {
		float diff = max(dot(inNormal, sceneData.sunlightDirection.xyz), 0.0);
		vec3 diffuse = diff * sceneData.sunlightColor.rgb;
		vec4 finalColor = mix(vec4((diffuse) * color.rgb, 1.0), fogColor, fog(0.25));
		if (selected) {
			outFragColor = mix(vec4(0.2f, 0.0f, 0.8f, 1.0f), finalColor, 1.0f - length(inPosition.xyz - sceneData.clickedPoint) * 2.0f);
		} else {
			outFragColor = finalColor;
		}
	} else {
		outFragColor = mix(color, fogColor, fog(0.25));
	}
}