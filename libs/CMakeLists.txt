set(GLFW_DIR ${CMAKE_CURRENT_SOURCE_DIR}/glfw)
add_subdirectory(${GLFW_DIR})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/VMA)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vkbootstrap)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tinyobj)

set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/imgui)
add_library(IMGUI STATIC)

target_sources( IMGUI
                PRIVATE
                    ${IMGUI_DIR}/imgui_demo.cpp
                    ${IMGUI_DIR}/imgui_draw.cpp
                    ${IMGUI_DIR}/imgui_tables.cpp
                    ${IMGUI_DIR}/imgui_widgets.cpp
                    ${IMGUI_DIR}/imgui.cpp
                    ${IMGUI_DIR}/misc/cpp/imgui_stdlib.cpp

                PRIVATE
                    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
                    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
                )

target_include_directories( IMGUI
                            PUBLIC ${IMGUI_DIR}
                            PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                            PUBLIC ${IMGUI_DIR}/backends
                            PUBLIC ${SDL2_DIR}/include
                            )

target_link_libraries(IMGUI PUBLIC glfw ${CMAKE_DL_LIBS})