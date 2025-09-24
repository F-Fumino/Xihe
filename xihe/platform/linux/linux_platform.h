#pragma once

#include "platform/platform.h"

namespace xihe
{
    class LinuxPlatform : public Platform
    {
    protected:
        void create_window(const Window::Properties &properties) override;
    };
}
