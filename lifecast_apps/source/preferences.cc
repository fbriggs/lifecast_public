// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "preferences.h"
#include "logger.h"
#include "util_string.h"
#include <fstream>
#include <filesystem>

#ifndef _WIN32
#include <wordexp.h>
#else
#include <array>
#include <codecvt>
#include <shlobj.h>
#include <stdexcept>
#endif

namespace p11 { namespace preferences {

std::string getPrefFilePath()
{
#ifndef _WIN32
  wordexp_t exp_result;
  wordexp("~/lifecast_preferences.txt", &exp_result, 0);
  const std::string path(exp_result.we_wordv[0]);
  wordfree(&exp_result);
  return path;
#else
  namespace fs = std::filesystem;
  std::array<wchar_t, 256> path;
  HRESULT result = SHGetFolderPathW(NULL, CSIDL_APPDATA, NULL, 0, path.data());
  if (SUCCEEDED(result)) {
    std::wstring widePathStr(path.data());
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t> > cvt;
    std::string narrow = cvt.to_bytes(widePathStr);
    fs::path prefPath = fs::path(narrow) / "lifecast.pref";
    return prefPath.string();
  } else {
    throw std::runtime_error("Can't retrieve AppData path");
  }
#endif
}

void setPrefs(const std::map<std::string, std::string>& prefs)
{
  std::ofstream f;
  f.open(getPrefFilePath());
  for (auto const& [k, v] : prefs) f << k << "," << v << std::endl;
  f.close();
}

std::map<std::string, std::string> getPrefs()
{
  std::map<std::string, std::string> prefs;
  std::ifstream file(getPrefFilePath());
  if (!file.is_open()) return prefs;

  for (std::string line; std::getline(file, line);) {
    const std::vector<std::string> parts = string::split(line, ',');
    XCHECK_EQ(parts.size(), 2) << "Corrupted preferences file " << getPrefFilePath()
                               << " line: " << line;
    prefs[parts[0]] = parts[1];
  }
  file.close();
  return prefs;
}

}}  // namespace p11::preferences
