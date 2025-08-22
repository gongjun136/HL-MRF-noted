#include "HL_MRF.h"

int main(int argc, char* argv[])
{
    //==================== 参数解析 ====================
    // ss: 输入点云目录；so: 结果输出目录
    // block_size: 分块时每块包含的扫描数量（默认 5）
    // downsample_size: 粗配准阶段下采样体素（m）（默认 0.1）
    // downsample_size_icp: ICP 搜索对应点的距离阈值（默认与 downsample_size 相同）
    // lum_iter: LUM 全局优化迭代次数（默认 3）
    // t_MCS: MCS 阈值，用于一致性推断（默认 10）
    // number_of_threads: OpenMP 并行线程数（默认 0 由PCL/OMP决定）
    // visualize_each_block: 是否在每个分块结束后可视化（0/1）
    std::string ss = argv[1];
    std::string so = argv[2];
    int   block_size = std::atoi(argv[3]);
    float downsample_size = std::atof(argv[4]);
    float downsample_size_icp = std::atof(argv[5]);
    float lum_iter = std::atof(argv[6]);
    int   t_MCS = std::atoi(argv[7]);
    int   number_of_threads = std::atoi(argv[8]);
    bool  visualize_each_block = std::atoi(argv[9]) != 0;

    //==================== 框架配置 ====================
    WHU::HL_MRF mf;
    mf.setBlockSize(block_size);
    mf.setPlyPath(ss);
    mf.setCoarseRgistration(CoarseRegistration::GROR); // 使用 GROR 进行粗配准
    mf.setDownsampleSize(downsample_size);
    mf.setDownsampleSizeForICP(downsample_size_icp);
    mf.setLumIterations(lum_iter);
    mf.visualizeEachBlock(visualize_each_block);
    mf.setMaximumConsensusSet(t_MCS);
    mf.setNumberOfThreads(number_of_threads);
    mf.setOutDir(so);

    //==================== 执行多视图配准 ====================
    mf.performMultiviewRegistration();
    // 如需区域生长配准，可改用：mf.performShapeGrowingRegistration();

    return 0;
}
