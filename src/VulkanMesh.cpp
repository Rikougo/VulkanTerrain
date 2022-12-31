//
// Created by Profanateur on 12/6/2022.
//

#include "VulkanMesh.hpp"


VertexInputDescription Vertex::GetVertexDescription() {
    VertexInputDescription description;

    //we will have just 1 vertex buffer binding, with a per-vertex rate
    VkVertexInputBindingDescription mainBinding = {};
    mainBinding.binding = 0;
    mainBinding.stride = sizeof(Vertex);
    mainBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    description.bindings.push_back(mainBinding);

    //Position will be stored at Location 0
    VkVertexInputAttributeDescription positionAttribute = {};
    positionAttribute.binding = 0;
    positionAttribute.location = 0;
    positionAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
    positionAttribute.offset = offsetof(Vertex, position);

    //Normal will be stored at Location 1
    VkVertexInputAttributeDescription normalAttribute = {};
    normalAttribute.binding = 0;
    normalAttribute.location = 1;
    normalAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
    normalAttribute.offset = offsetof(Vertex, normal);

    //Color will be stored at Location 2
    VkVertexInputAttributeDescription colorAttribute = {};
    colorAttribute.binding = 0;
    colorAttribute.location = 2;
    colorAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
    colorAttribute.offset = offsetof(Vertex, color);

    VkVertexInputAttributeDescription uvAttribute = {};
	uvAttribute.binding = 0;
	uvAttribute.location = 3;
	uvAttribute.format = VK_FORMAT_R32G32_SFLOAT;
	uvAttribute.offset = offsetof(Vertex, uv);

    description.attributes.push_back(positionAttribute);
    description.attributes.push_back(normalAttribute);
    description.attributes.push_back(colorAttribute);
	description.attributes.push_back(uvAttribute);
    return description;
}

bool Mesh::LoadFromObj(const std::filesystem::path &p_path) {
    //l_attribute will contain the vertex arrays of the file
    tinyobj::attrib_t l_attribute;
    //l_shapes contains the info for each separate object in the file
    std::vector<tinyobj::shape_t> l_shapes;
    //l_materials contains the information about the material of each shape, but we won't use it.
    std::vector<tinyobj::material_t> l_materials;

    //error and warning output from the load function
    std::string l_warning;
    std::string l_error;

    //load the OBJ file
    tinyobj::LoadObj(
            &l_attribute,
            &l_shapes,
            &l_materials,
            &l_warning,
            &l_error,
            p_path.string().c_str(), nullptr);
    //make sure to output the warnings to the console, in case there are issues with the file
    if (!l_warning.empty()) {
        std::cout << "WARN: " << l_warning << std::endl;
    }
    //if we have any error, print it to the console, and break the mesh loading.
    //This happens if the file can't be found or is malformed
    if (!l_error.empty()) {
        std::cerr << l_error << std::endl;
        return false;
    }

    // Loop over shapes
    for (size_t s = 0; s < l_shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < l_shapes[s].mesh.num_face_vertices.size(); f++) {

            //hardcode loading to triangles
            int fv = 3;

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = l_shapes[s].mesh.indices[index_offset + v];

                //vertex position
                tinyobj::real_t vx = l_attribute.vertices[3 * idx.vertex_index + 0];
                tinyobj::real_t vy = l_attribute.vertices[3 * idx.vertex_index + 1];
                tinyobj::real_t vz = l_attribute.vertices[3 * idx.vertex_index + 2];
                //vertex normal
                tinyobj::real_t nx = l_attribute.normals[3 * idx.normal_index + 0];
                tinyobj::real_t ny = l_attribute.normals[3 * idx.normal_index + 1];
                tinyobj::real_t nz = l_attribute.normals[3 * idx.normal_index + 2];

                //copy it into our vertex
                Vertex new_vert;
                new_vert.position.x = vx;
                new_vert.position.y = vy;
                new_vert.position.z = vz;

                new_vert.normal.x = nx;
                new_vert.normal.y = ny;
                new_vert.normal.z = nz;

                //we are setting the vertex color as the vertex normal. This is just for display purposes
                new_vert.color = new_vert.normal;

                tinyobj::real_t ux = l_attribute.texcoords[2 * idx.texcoord_index + 0];
			    tinyobj::real_t uy = l_attribute.texcoords[2 * idx.texcoord_index + 1];

                new_vert.uv.x = ux;
                new_vert.uv.y = 1-uy;


                m_vertices.push_back(new_vert);
            }
            index_offset += fv;
        }
    }

    return true;
}

Mesh VulkanUtil::CreateQuad(float p_size /*= 1*/, uint8_t p_resolution /*= 1*/) {
    Mesh l_result{};

    if (p_resolution <= 1) {
        l_result.m_vertices = {
                {{-p_size, 0.0f, -p_size}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
                {{ p_size, 0.0f, -p_size}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
                {{-p_size, 0.0f,  p_size}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
                {{ p_size, 0.0f,  p_size}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}
        };
    }

    for(unsigned i = 0; i <= p_resolution - 1; i++)
    {
        for(unsigned j = 0; j <= p_resolution - 1; j++)
        {
            l_result.m_vertices.push_back(Vertex{
                {-p_size/2.0f + p_size* i / (float)p_resolution, 0.0f, -p_size/2.0f + p_size* j / (float)p_resolution},
                {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {i / (float)p_resolution, j / (float)p_resolution}});
            l_result.m_vertices.push_back(Vertex{
                {-p_size/2.0f + p_size* (i+1) / (float)p_resolution, 0.0f, -p_size/2.0f + p_size* j / (float)p_resolution},
                {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {(i+1) / (float)p_resolution, j / (float)p_resolution}});
            l_result.m_vertices.push_back(Vertex{
                {-p_size/2.0f + p_size* i / (float)p_resolution, 0.0f, -p_size/2.0f + p_size* (j+1) / (float)p_resolution},
                {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {i / (float)p_resolution, (j+1) / (float)p_resolution}});
            l_result.m_vertices.push_back(Vertex{
                {-p_size/2.0f + p_size* (i+1) / (float)p_resolution, 0.0f, -p_size/2.0f + p_size* (j+1) / (float)p_resolution},
                {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {(i+1) / (float)p_resolution, (j+1) / (float)p_resolution}});
        }
    }

    return l_result;
}
