#include <VulkanEngine.hpp>

int main(int argc, const char* argv[]) {
    VulkanEngine l_engine{};

    l_engine.Init();
    l_engine.Run();
    l_engine.Cleanup();
}