# ComfyUI-flash-attention-rdna3-win-zluda
A comfyui extension of flash attention v2 optimization for AMD RDNA3 GPU in Windows ZLUDA Envirionment.

# Build dependant libraries
Fork https://github.com/Repeerc/flash-attn-composable-kernel-gfx110x-windows-port
### Step 1: Build CK library and flash attention kernel
 
need: HIP SDK 6.2.4, cmake, ninja

in `ck_fattn_ker`

```bash
mkdir build
cd build
cmake .. -G Ninja -DHIP_PLATFORM=amd -DCMAKE_CXX_COMPILER_ID=Clang -D_CMAKE_HIP_DEVICE_RUNTIME_TARGET=ON -DCMAKE_CXX_COMPILER_FORCED=true -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DPYTHON_EXECUTABLE=C:/path/to/miniconda/envs/<your-environment-name>/bin/python.exe
ninja
```

generate `ck_fttn_lib.dll`

### Step 2: Build python bind module

need: MSVC, cmake, cuda11.8 toolchain, ninja

in `bridge`

```
mkdir build
cd build
cmake .. -G Ninja -DPYTHON_EXECUTABLE=C:/path/to/miniconda/envs/<your-environment-name>/bin/python.exe
ninja
```

generate `ck_fttn_pyb.pyd`

### Step 3: Build flash_attn_wmma.pyd
Fork https://github.com/Repeerc/flash-attention-v2-RDNA3-minimal

Patch C:/path/to/miniconda/envs/<your-environment-name>/Lib/site-packages/torch/utils/cpp_extension.py, remove -fPIC, use Visual Studio linking syntax etc:

```
diff --git a/cpp_extension.py.orig b/cpp_extension.py
index b4a70dc..bccb663 100644
--- a/cpp_extension.py.orig
+++ b/cpp_extension.py
@@ -262,7 +262,6 @@ COMMON_NVCC_FLAGS = [
 ]

 COMMON_HIP_FLAGS = [
-    '-fPIC',
     '-D__HIP_PLATFORM_AMD__=1',
     '-DUSE_ROCM=1',
     '-DHIPBLAS_V2',
@@ -1185,7 +1184,7 @@ def CUDAExtension(name, sources, *args, **kwargs):

         extra_compile_args_dlink = extra_compile_args.get('nvcc_dlink', [])
         extra_compile_args_dlink += ['-dlink']
-        extra_compile_args_dlink += [f'-L{x}' for x in library_dirs]
+        extra_compile_args_dlink += [f'/LIBPATH:{x}' for x in library_dirs]
         extra_compile_args_dlink += [f'-l{x}' for x in dlink_libraries]

         if (torch.version.cuda is not None) and TorchVersion(torch.version.cuda) >= '11.2':
@@ -1999,8 +1998,8 @@ def _prepare_ldflags(extra_ldflags, with_cuda, verbose, is_standalone):
             if CUDNN_HOME is not None:
                 extra_ldflags.append(f'-L{os.path.join(CUDNN_HOME, "lib64")}')
         elif IS_HIP_EXTENSION:
-            extra_ldflags.append(f'-L{_join_rocm_home("lib")}')
-            extra_ldflags.append('-lamdhip64')
+            extra_ldflags.append(f'/LIBPATH:{_join_rocm_home("lib")}')
+            extra_ldflags.append('amdhip64.lib')
     return extra_ldflags
```
Need MSVC Compiler, AMD HIP SDK and rocWMMA Library.

Rocwmma library: https://github.com/ROCm/rocWMMA, check out 6.2.4 branch

clone it and copy ```library/include/rocwmma``` to HIP SDK installation path of ```include``` folder

In cmd.exe, run ```vcvars64.bat``` to active MSVC Environment, then run ```zluda -- python bench_with_sdpa.py```

Look for flash_attn_wmma.pyd in build directory

# Workflow
![Workflow](https://github.com/jiangfeng79/ComfyUI-flash-attention-rdna3-win-zluda/blob/main/workflow/200413_ComfyUI_00001_.png?raw=true)
