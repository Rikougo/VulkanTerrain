#version 460

layout (set = 0, binding = 1) uniform sampler2D displacementMap;

layout(quads, equal_spacing, cw) in;

layout (location = 0) in vec3 inNormal[];
layout (location = 1) in vec2 inUV[];

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec2 outUV;
layout (location = 2) out float outHeight;
layout (location = 3) out vec4 outPosition;

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
    bool useLightning;
} sceneData;

layout(set = 2, binding = 0) uniform sampler2D heightMap;

#define M_PI 3.14159265358979323846

float rand(vec2 co){return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);}
float rand (vec2 co, float l) {return rand(vec2(rand(co), l));}
float rand (vec2 co, float l, float t) {return rand(vec2(rand(co, l), t));}

float perlin(vec2 p, float dim, float time) {
	vec2 pos = floor(p * dim);
	vec2 posx = pos + vec2(1.0, 0.0);
	vec2 posy = pos + vec2(0.0, 1.0);
	vec2 posxy = pos + vec2(1.0);

	float c = rand(pos, dim, time);
	float cx = rand(posx, dim, time);
	float cy = rand(posy, dim, time);
	float cxy = rand(posxy, dim, time);

	vec2 d = fract(p * dim);
	d = -0.5 * cos(d * M_PI) + 0.5;

	float ccx = mix(c, cx, d.x);
	float cycxy = mix(cy, cxy, d.x);
	float center = mix(ccx, cycxy, d.y);

	return center * 2.0 - 1.0;
}

// p must be normalized!
float perlin(vec2 p, float dim) {
	return perlin(p, dim, 0.0);
}

void main()
{
	// get patch coordinate
    float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;

    // ----------------------------------------------------------------------
    // retrieve control point texture coordinates
    vec2 t00 = inUV[0];
    vec2 t01 = inUV[1];
    vec2 t10 = inUV[2];
    vec2 t11 = inUV[3];

    // bilinearly interpolate texture coordinate across patch
    vec2 t0 = (t01 - t00) * u + t00;
    vec2 t1 = (t11 - t10) * u + t10;
    vec2 texCoord = (t1 - t0) * v + t0;
    outUV = texCoord;

    // lookup texel at patch coordinate for height and scale + shift as desired
    float texHeight = texture(heightMap, outUV).r;//perlin(texCoord, 64) * 0.5f + 0.5f;
    outHeight = texHeight;

    // ----------------------------------------------------------------------
    // retrieve control point position coordinates
    vec4 p00 = gl_in[0].gl_Position;
    vec4 p01 = gl_in[1].gl_Position;
    vec4 p10 = gl_in[2].gl_Position;
    vec4 p11 = gl_in[3].gl_Position;

    float d = texture(heightMap, outUV - vec2( 0.0f, 0.01f)).r;
    float t = texture(heightMap, outUV + vec2( 0.0f, 0.01f)).r;
    float r = texture(heightMap, outUV - vec2(0.01f,  0.0f)).r;
    float l = texture(heightMap, outUV + vec2(0.01f,  0.0f)).r;

    // compute patch surface normal
    vec4 uVec = (p01) - (p00);
    vec4 vVec = (p10) - (p00);
    vec4 normal = normalize(vec4(cross(vVec.xyz, uVec.xyz), 0));
    outNormal = normalize(vec3(2 * r - l, 4, 2 * d - t));

    // bilinearly interpolate position coordinate across patch
    vec4 p0 = (p01 - p00) * u + p00;
    vec4 p1 = (p11 - p10) * u + p10;
    vec4 p = (p1 - p0) * v + p0;

    // displace point along normal
    p += normal *  (texHeight * sceneData.displacementFactor);
    outPosition = p;
	gl_Position = cameraData.viewproj * outPosition;
}