#version 460

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec2 inTexCoord;
layout (location = 2) in float inHeight;

//output write
layout (location = 0) out vec4 outFragColor;

void main()
{
	outFragColor = vec4(inTexCoord.xy, 0.0f, 1.0f);
}