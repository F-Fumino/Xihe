#include "platform/linux/linux_platform.h"
#include "platform/glfw_window.h"

namespace xihe
{
    void LinuxPlatform::create_window(const Window::Properties &properties)
    {
        window_ = std::make_unique<GlfwWindow>(this, properties);
    }
}
