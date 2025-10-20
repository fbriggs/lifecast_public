; MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

[Setup]
SourceDir=..\bazel-bin\source
AppName=Lifecast Volumetric Video Editor
AppVersion=2.7
WizardStyle=modern
DefaultDirName={autopf}\Lifecast
DefaultGroupName=Lifecast
UninstallDisplayIcon={app}\vve.exe
Compression=lzma2
;Compression=none
;DiskSpanning=yes
SolidCompression=yes
OutputDir=userdocs:VolumetricVideoEditor
; Output will go in the Documents folder.. not great but ok
; "ArchitecturesAllowed=x64" specifies that Setup cannot run on
; anything but x64.
ArchitecturesAllowed=x64
; "ArchitecturesInstallIn64BitMode=x64" requests that the install be
; done in "64-bit mode" on x64, meaning it should use the native
; 64-bit Program Files directory and the 64-bit view of the registry.
ArchitecturesInstallIn64BitMode=x64
LicenseFile=EULA.rtf
OutputBaseFilename=Lifecast_VolumetricVideoEditor_Install_2.7

[Files]
Source: "vve.exe"; DestDir: "{app}"; DestName: "Lifecast_VolumetricVideoEditor.exe"
Source: "*.dll"; DestDir: "{app}"
Source: "*.pt"; DestDir: "{app}"
Source: "*.ttf"; DestDir: "{app}"
Source: "*.ico"; DestDir: "{app}"
;Source: "*.zip"; DestDir: "{app}"

[Icons]
Name: "{group}\VolumetricVideoEditor"; Filename: "{app}\Lifecast_VolumetricVideoEditor.exe"; IconFilename: "{app}\icon_256.ico"
Name: "{userdesktop}\Lifecast_VolumetricVideoEditor"; Filename: "{app}\Lifecast_VolumetricVideoEditor.exe"; IconFilename: "{app}\icon_256.ico"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}";
