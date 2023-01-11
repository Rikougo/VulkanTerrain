// tessellation control shader
#version 460

layout (vertices=4) out; // patch size

// varying input from vertex shader
layout (location = 0) in vec3 inNormal[];
layout (location = 1) in vec2 inTexCoord[];

// varying output to evaluation shader
layout (location = 0) out vec3 outNormal[];
layout (location = 1) out vec2 outTexCoord[];

layout(set = 0, binding = 0) uniform  CameraBuffer {
	mat4 view;
	mat4 proj;
	mat4 viewproj;
    vec3 cameraPosition;
} cameraData;

layout(set = 0, binding = 1) uniform  SceneData {
	vec4 ambientColor;
	vec4 sunlightDirection;
	vec4 sunlightColor;
    float terrainSubdivision;
    float displacementFactor;
    float minDistance;
    float maxDistance;
	vec3 clickedPoint;
    uint viewMode;
} sceneData;

void main()
{
    // ----------------------------------------------------------------------
    // pass attributes through
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
	outNormal[gl_InvocationID] = inNormal[gl_InvocationID];
    outTexCoord[gl_InvocationID] = inTexCoord[gl_InvocationID];

    // ----------------------------------------------------------------------
    // invocation zero controls tessellation levels for the entire patch
    if (gl_InvocationID == 0)
    {
        vec4 eyeSpacePos00 = cameraData.view * gl_in[0].gl_Position;
        vec4 eyeSpacePos01 = cameraData.view * gl_in[1].gl_Position;
        vec4 eyeSpacePos10 = cameraData.view * gl_in[2].gl_Position;
        vec4 eyeSpacePos11 = cameraData.view * gl_in[3].gl_Position;

        float distance00 = clamp((abs(eyeSpacePos00.z)-sceneData.minDistance) / (sceneData.maxDistance-sceneData.minDistance), 0.0, 1.0);
        float distance01 = clamp((abs(eyeSpacePos01.z)-sceneData.minDistance) / (sceneData.maxDistance-sceneData.minDistance), 0.0, 1.0);
        float distance10 = clamp((abs(eyeSpacePos10.z)-sceneData.minDistance) / (sceneData.maxDistance-sceneData.minDistance), 0.0, 1.0);
        float distance11 = clamp((abs(eyeSpacePos11.z)-sceneData.minDistance) / (sceneData.maxDistance-sceneData.minDistance), 0.0, 1.0);

        float tessLevel0 = mix( sceneData.terrainSubdivision, 1.0f, min(distance10, distance00));
        float tessLevel1 = mix( sceneData.terrainSubdivision, 1.0f, min(distance00, distance01));
        float tessLevel2 = mix( sceneData.terrainSubdivision, 1.0f, min(distance01, distance11));
        float tessLevel3 = mix( sceneData.terrainSubdivision, 1.0f, min(distance11, distance10));

        gl_TessLevelOuter[0] = tessLevel0;
        gl_TessLevelOuter[1] = tessLevel1;
        gl_TessLevelOuter[2] = tessLevel2;
        gl_TessLevelOuter[3] = tessLevel3;

        gl_TessLevelInner[0] = max(tessLevel1, tessLevel3);
        gl_TessLevelInner[1] = max(tessLevel0, tessLevel2);
    }
}
	