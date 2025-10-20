// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "util_runfile.h"
#include <iostream>
#include "logger.h"
#include <filesystem>

#ifndef _WIN32
#include "tools/cpp/runfiles/runfiles.h"
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <mach-o/dyld.h>  // For _NSGetExecutablePath
#endif

#ifdef _WIN32
#include <Windows.h>
#endif

namespace p11 { namespace runfile {

std::string getExecutablePath()
{
#if defined(__APPLE__)
  char path[1024];
  uint32_t size = sizeof(path);
  XCHECK_EQ(_NSGetExecutablePath(path, &size), 0) << "Buffer size too small, needed " << size;
  return path;
#elif defined(__linux__)
  constexpr int kBufferSize = 1024;
  char result[kBufferSize];
  ssize_t count = readlink("/proc/self/exe", result, kBufferSize);
  return std::string(result, (count > 0) ? count : 0);
#elif defined(_WIN32)
  constexpr int kMaxPath = 1024;
  WCHAR path[kMaxPath];
  GetModuleFileNameW(nullptr, path, kMaxPath);
  std::wstring wpath(path);
  return std::string(wpath.begin(), wpath.end());
#else
#error "unsupported platform"
#endif
}

std::string getRunfileResourcePath(const std::string& res_path)
{
#if defined(LOOK_FOR_RUNFILES_IN_MAC_APP)
  std::filesystem::path p(getExecutablePath());
  const std::string path = p.parent_path().parent_path();
  return path + "/" + res_path;
#elif defined(_WIN32)
  std::filesystem::path p(getExecutablePath());
  const std::wstring wpath = p.parent_path();
  const std::string path(wpath.begin(), wpath.end());
  return path + "\\" + res_path;
#else
  using bazel::tools::cpp::runfiles::Runfiles;
  std::string error;
  const std::string exe_path = getExecutablePath();
  std::unique_ptr<Runfiles> runfiles(Runfiles::Create(exe_path, &error));
  XCHECK(runfiles.get()) << error;

  std::string bazel_main_path = runfiles->Rlocation("_main/");
  if (!std::filesystem::is_directory(std::filesystem::path(bazel_main_path))) {
    // HACK: _main/ for some versions of bazel __main__ for others
    return runfiles->Rlocation("__main__/" + res_path);
  }
  return runfiles->Rlocation("_main/" + res_path);
#endif
}

}}  // namespace p11::runfile
