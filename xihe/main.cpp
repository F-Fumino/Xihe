#include "platform/window.h"
#include <fstream>
#include <iostream>
#include <memory>

#ifdef _WIN32
#	include "platform/windows/windows_platform.h"
#else
#	include "platform/linux/linux_platform.h"
#endif

extern std::unique_ptr<xihe::Application> create_application();

#ifdef _WIN32
#	include <Windows.h>

int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR lpCmdLine, INT nCmdShow)
{
	AllocConsole();
	FILE *fDummy;
	freopen_s(&fDummy, "CONIN$", "r", stdin);
	freopen_s(&fDummy, "CONOUT$", "w", stderr);
	freopen_s(&fDummy, "CONOUT$", "w", stdout);

	xihe::WindowsPlatform platform{};
#else

int main(int argc, char **argv)
{
	xihe::LinuxPlatform platform{};
#endif

	xihe::Window::OptionalProperties properties{};
	properties.title = "Xi He";
	properties.vsync = xihe::Window::Vsync::OFF;
	platform.set_window_properties(properties);

	const auto code = platform.initialize();
	platform.start_app("xihe", create_application);

	if (code == xihe::ExitCode::kSuccess)
	{
		platform.main_loop();
	}

	platform.terminate(code);

#ifdef _WIN32
	FreeConsole();
#endif

	return 0;
}
