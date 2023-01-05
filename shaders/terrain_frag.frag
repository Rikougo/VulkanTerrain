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

float fog(float density)
{
	const float LOG2 = -1.442695;
	float dist = gl_FragCoord.z / gl_FragCoord.w * 0.1;
	float d = density * dist;
	return 1.0 - clamp(exp2(d * d * LOG2), 0.0, 1.0);
}

void main()
{
	float diff = max(dot(inNormal, sceneData.sunlightDirection.xyz), 0.0);
    vec3 diffuse = diff * sceneData.sunlightColor.rgb;

    vec3 ambient = 0.1f * sceneData.sunlightColor.rgb;

    vec4 color = vec4(inHeight, inHeight, inHeight, 1.0f);
    const vec4 fogColor = vec4(0.0f, 0.0f, 0.0f, 0.0);
	// outFragColor = mix(vec4((diffuse + ambient) * color.rgb, 1.0), fogColor, fog(0.25));
	outFragColor = mix(color, fogColor, fog(0.25));
    // outFragColor = mix(color, fogColor, fog(0.25));
}