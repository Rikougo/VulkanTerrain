#version 460

layout (location = 0) in vec3 fNormal;
layout (location = 1) in vec3 fColor;

//output write
layout (location = 0) out vec4 outFragColor;

void main()
{
	//return red
	outFragColor = vec4(fColor, 1.0f);
}