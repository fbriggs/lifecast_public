// MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#pragma once

#if defined(_WIN32)
#include "third_party/dirent.h"
#else
#include <dirent.h>
#endif

#include <chrono>
#include <iostream>
#include <string>
#include <algorithm>
#include <fstream>
#include <regex>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <ctime>
#include "util_string.h"

namespace p11 { namespace file {
static std::vector<std::string> getFilesInDir(const std::string& dir_name)
{
  std::vector<std::string> filenames;
  DIR* dtemp;
  dirent* dent;
  dtemp = opendir(dir_name.c_str());
  if (!dtemp) return std::vector<std::string>();
  while (true) {
    dent = readdir(dtemp);
    if (!dent) break;
    if (std::string(dent->d_name)[0] == '.') continue;
    filenames.push_back(std::string(dent->d_name));
  }
  closedir(dtemp);

  std::sort(filenames.begin(), filenames.end());
  return filenames;
}

static std::string filenamePrefix(const std::string& filename)
{
  const std::vector<std::string> tokens = string::split(filename, '.');
  XCHECK_EQ(tokens.size(), 2);
  return tokens[0];
}

static std::string filenameExtension(const std::string& filename)
{
  const std::vector<std::string> tokens = string::split(filename, '.');
  XCHECK_GE(tokens.size(), 1);
  return tokens[tokens.size() - 1];
}

static std::string lastDirInPath(std::string path)
{
  XCHECK_GE(path.size(), 1);
  if (path[path.size() - 1] == '/') path.pop_back();
  const std::vector<std::string> tokens = string::split(path, '/');
  XCHECK_GE(tokens.size(), 1);
  return tokens[tokens.size() - 1];
}

// Returns the subset of filenames that contain the substring 'pattern'
static std::vector<std::string> filterFilesContaining(
    const std::vector<std::string>& filenames, const std::string& pattern)
{
  std::vector<std::string> matches;
  std::copy_if(
      filenames.begin(), filenames.end(), std::back_inserter(matches), [&](const std::string& f) {
        return f.find(pattern) != std::string::npos;
      });
  return matches;
}

static bool fileExists(const std::string& file_path)
{
  std::ifstream file(file_path);
  return file.good();
}

static bool directoryExists(const std::string& dir_path)
{
  return std::filesystem::is_directory(std::filesystem::path(dir_path));
}

static bool createDirectoryIfNotExists(const std::filesystem::path& dir) {
  if (!std::filesystem::is_directory(dir)) {
    std::filesystem::create_directories(dir);
    return true;
  }
  return false;
}

static int countMatchingFiles(const std::string& dir, const std::string& regex)
{
  std::vector<std::string> filenames = file::getFilesInDir(dir);
  std::regex pattern(regex);
  int count = 0;
  for (const std::string& f : filenames) {
    if (std::regex_match(f, pattern)) ++count;
  }
  return count;
}


// Generate a unique filename, useful to avoid overwriting existing files
static std::string createTimestampFilename(const std::string& directory, 
                                           const std::string& baseName, 
                                           const std::string& extension) {
    // Current date/time based on current system
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm = *std::localtime(&now_c);
    
    // Use stringstream to format the time appropriately
    std::ostringstream timestamp;
    timestamp << std::put_time(&now_tm, "%Y_%m_%d-%H%M");

    // Create the output filename string
    std::filesystem::path outputPath(directory);
    outputPath /= baseName + "-" + timestamp.str() + "." + extension;

    return outputPath.string();
}

static void clearDirectoryContents(const std::string& directory) {
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        std::filesystem::remove_all(entry);
    }
}

static void openFileExplorer(const std::string& directory) {
#if defined(_WIN32)
    std::string command = "explorer \"" + directory + "\"";
#elif defined(__APPLE__)
    std::string command = "open \"" + directory + "\"";
#else
    std::string command = "xdg-open \"" + directory + "\"";
#endif
    std::system(command.c_str());
}

// NOTE: path can have wildcards
static void crossPlatformDelete(const std::string path) {
#if defined(_WIN32)
  std::string command = "del \"" + path + "\"";
  std::replace(command.begin(), command.end(), '/', '\\');
#else
  std::string command = "bash -c 'rm " + path + "'";
#endif
  XPLINFO << command;
  std::system(command.c_str());
}

}}  // namespace p11::file
