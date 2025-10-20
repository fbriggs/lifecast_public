== How to build the windows installer ===

1) Download Inno Setup here:
    https://jrsoftware.org/isinfo.php

2) Build applications

    bazel build -c opt --copt="-Dwindows_hide_console=true" --linkopt="/SUBSYSTEM:WINDOWS" //source:vve

    or

    We compile tinycudann for specific GPU types, see https://developer.nvidia.com/cuda-gpus

    bazel build --cuda_archs=compute_75,sm_75,sm_86,sm_89 -c opt --copt="-Dwindows_hide_console=true" --linkopt="/SUBSYSTEM:WINDOWS" //source:volurama

    or

    bazel build -c opt --copt="-Dwindows_hide_console=true" --linkopt="/SUBSYSTEM:WINDOWS" //source:upscale_video

    or

    (note the cuda_arches here support RTX 2080, 3090, 4090. We don't have it setup for explicit support of 5090, but JIT might work). 

    bazel build --local_cpu_resources=10 --jobs=1 --cuda_archs="compute_75:sm_75;compute_86:sm_86;compute_89:sm_89" -c opt --copt=/Dwindows_hide_console --linkopt="/SUBSYSTEM:WINDOWS" //source:4dgstudio

Note the extra flags here disable the command window for the release build.

3) Run Ino Setup on install_vve.iss, install_volurama.iss, etc.
