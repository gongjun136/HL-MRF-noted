cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(HL_MRF)

if(WIN32)
  message(STATUS "Test in Windows")
  # Windows 下优先使用本地 vcpkg（你的路径）
  set(VCPKG_ROOT_DIR "F:\\vcpkg_test\\vcpkg\\")
  set(CMAKE_PREFIX_PATH "${VCPKG_ROOT_DIR}installed\\x64-windows\\")
  include("${VCPKG_ROOT_DIR}scripts\\buildsystems\\vcpkg.cmake")
#   add_compile_options(/EHsc)        # 启用 C++ 异常处理（/EHsc）
#   add_definitions(-DNOMINMAX)       # 禁用 <Windows.h> 中的 min/max 宏污染
#   add_compile_options("/bigobj")    # 解决模板/大目标导致的段数量限制
#   add_compile_definitions(_HAS_TR1_NAMESPACE)      # 兼容旧 TR1 命名空间（C++11 相关）
#   add_compile_definitions(_HAS_STD_BYTE=0)         # 避免 std::byte 与 Windows.h 中 byte 冲突
#   set(LIB_PREFIX "lib")             # Windows 下额外前缀（供内部脚本使用）
endif()

# 添加宏定义，控制LOG日志和性能分析
add_definitions(-DMRF_DEBUG)
# 添加宏定义，设置项目根目录路径
add_definitions(-DROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/")
# 1) 显式找 lz4（比 PCL/FLANN 更早暴露问题）
find_package(lz4 CONFIG REQUIRED)

# 2) 兼容补丁：把大写命名空间的目标映射到小写的 lz4::lz4
#    注意：不能用 ALIAS 到 IMPORTED；这里用 INTERFACE IMPORTED + 转发
if(TARGET lz4::lz4 AND NOT TARGET LZ4::lz4_shared)
  add_library(LZ4::lz4_shared INTERFACE IMPORTED)
  set_target_properties(LZ4::lz4_shared PROPERTIES
    INTERFACE_LINK_LIBRARIES lz4::lz4)
endif()

if(TARGET lz4::lz4 AND NOT TARGET LZ4::lz4_static)
  add_library(LZ4::lz4_static INTERFACE IMPORTED)
  set_target_properties(LZ4::lz4_static PROPERTIES
    INTERFACE_LINK_LIBRARIES lz4::lz4)
endif()
find_package (PCL REQUIRED COMPONENTS)
include_directories (${PCL_INCLUDE_DIRS})
link_directories (${PCL_LIBRARY_DIRS})
add_definitions (${PCL_DEFINITIONS})

find_package(Boost COMPONENTS graph filesystem)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


include_directories(./include)
include_directories(./include/Graph)
include_directories(./include/GROR)

set(Graph ./include/Graph/scan_graph_inference.h
                   ./include/Graph/transformation_graph.h
                   ./include/Graph/transformation_graph.hpp
                   ./src/scan_graph_inference.cpp
                   )
set(GROR ./include/GROR/include/ia_gror.h
    ./include/GROR/include/ia_gror.hpp
    ./include/GROR/include/gror_pre.h
    ./include/GROR/include/gror_common.h
    ./include/GROR/src/gror_pre.cpp
)
set(HL_MRF ./include/HL_MRF.h 
                    ./src/HL_MRF.cpp
                   )

set(SourceFile ./src/example_HL_MRF.cpp)

add_executable(HL_MRF ${Graph} ${GROR} ${HL_MRF} ${SourceFile} )
target_link_libraries(HL_MRF ${PCL_LIBRARIES} ${Boost_LIBRARIES})
