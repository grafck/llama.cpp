# Kunlun K200 (KL1) backend — links against the vendored XRE 4.33 runtime
# installed at /usr/local/xpu-4.33.0 (see skill kunlun-xpu-driver).

set(KUNLUN_XPU_PREFIX "xpu-4.33.0")
set(KUNLUN_XPU_ROOT "/usr/local/${KUNLUN_XPU_PREFIX}" CACHE PATH "XRE 4.33 install prefix")

if (NOT EXISTS "${KUNLUN_XPU_ROOT}/include/xpu/runtime.h")
    message(FATAL_ERROR "GGML_KUNLUN: ${KUNLUN_XPU_ROOT}/include/xpu/runtime.h not found (install the vendored xdnn/xpurt tree)")
endif()

file(GLOB GGML_SOURCES_KUNLUN "*.cpp")
file(GLOB GGML_HEADERS_KUNLUN "*.h")

ggml_add_backend_library(ggml-kunlun
    ${GGML_SOURCES_KUNLUN}
    ${GGML_HEADERS_KUNLUN}
)

target_include_directories(ggml-kunlun PRIVATE "${KUNLUN_XPU_ROOT}/include")
target_link_directories(ggml-kunlun PRIVATE "${KUNLUN_XPU_ROOT}/lib64")
target_link_libraries(ggml-kunlun PRIVATE xpuapi xpurt)
target_link_options(ggml-kunlun PRIVATE "-Wl,-rpath,${KUNLUN_XPU_ROOT}/lib64")

target_compile_definitions(ggml PUBLIC GGML_USE_KUNLUN)
