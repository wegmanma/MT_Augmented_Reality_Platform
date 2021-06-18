#include <iostream>
#include <stdexcept>
#include "VulkanFramework.hpp"


VulkanFramework *globalFrameworkPtr;

int main() {
    VulkanFramework app;
    globalFrameworkPtr = &app;
    setenv("DISPLAY", ":0", 0);
    try {
        app.run();
    }
    catch (const std::exception & e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

