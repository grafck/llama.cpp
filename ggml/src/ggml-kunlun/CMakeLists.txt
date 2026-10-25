ggml_add_backend_library(ggml-kunlun
                        ggml-kunlun.cpp)

set(XPU_KUNLUN_ROOT "/usr/local/xpu-4.33.0" CACHE PATH "ggml: path to the Kunlun XPU 4.33 runtime installation")

target_include_directories(ggml-kunlun PRIVATE ${XPU_KUNLUN_ROOT}/include)
target_link_directories  (ggml-kunlun PRIVATE ${XPU_KUNLUN_ROOT}/lib64)
target_link_libraries    (ggml-kunlun PRIVATE xpuapi xpurt)
