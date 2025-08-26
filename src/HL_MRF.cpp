/*
 * =========================== 说明（中文注释版） ===========================
 * 本文件实现了 WHU::HL_MRF 多视角层次化配准流程（TLS 点云）:
 *  - 预处理：下采样（voxel_size / voxel_size_icp）、ISS 关键点、FPFH 特征；
 *  - 粗配准：GROR 估计每对扫描的初始变换与评分（best_count / MSAC）；
 *  - 一致性推断：ScanGraphInference 基于 MCS/评分剔除不一致的边；
 *  - 细配准：对 MST 边执行 ICP 微调；
 *  - 位姿拼接：DFS 合成相对根的绝对位姿；
 *  - 全局优化：PCL::registration::LUM 做图优化并导出结果。
 *
 * 关键参数：
 *  - voxel_size：特征与粗配准分辨率；voxel_size_icp：ICP 搜索半径；
 *  - translation_accuracy / rotation_accuracy：一致性推断阈值；
 *  - part_num：一个块内的扫描数量；
 *  - max_consensus_set：MCS 阈值（小于该值的边容易被拒绝）；
 *  - num_threads：OpenMP 并行线程数。
 *
 * 仅添加注释，不改动任何业务逻辑与接口。
 * =======================================================================
 */

#include "../include/HL_MRF.h"

WHU::HL_MRF::HL_MRF() : voxel_size(0.2),
                        voxel_size_icp(0.1),
                        use_cyclic_constraints(1),
                        use_LUM(1),
                        translation_accuracy(0.5),
                        rotation_accuracy(0.087),
                        approx_overlap(0.3),
                        part_num(5),
                        check_blcok(false),
                        bigloop(0),
                        all_count(0),
                        use_pairs(false),
                        max_consensus_set(10),
                        num_threads(0)
{
}
WHU::HL_MRF::~HL_MRF()
{
}

/**
 * @brief 初始化分块与计数器等内部状态。
 *
 * 功能：
 * 1. 根据输入点云文件数量与 `part_num`（每块点云数量）计算分块次数 `part_loop` 与最后一块余数 `part_num_rest`；
 * 2. 当 `bigloop==1` 表示递归对上一次的分块结果再次分块，此时仅处理 `part_clouds`；
 * 3. 当 `use_pairs==true` 时，表示由外部提供配对（pairs），不再进行常规分块。
 */
void WHU::HL_MRF::init()
{
    // ====== 第一层循环 (bigloop == 0)：对原始扫描文件进行分块 ======
    if (bigloop == 0)
    {
        // 计算分块总数 part_loop：
        // 先用整除得到完整块的数量，再加 1 作为最后可能的余数块
        part_loop = static_cast<int>(files.size() / part_num) + 1;

        // 计算最后一块的余数大小（不足 part_num 的数量）
        part_num_rest = static_cast<int>(files.size() % part_num);

        // 重置计数器，表示当前层从第 0 个块开始
        all_count = 0;
    }
    else
    {
        // ====== 第二层及以后循环 (bigloop > 0)：对上一次合并结果再分块 ======
        // （默认只做一块，不再继续递归细分；如需继续分块，可解开注释部分代码）

        /*part_loop = static_cast<int>(part_clouds.size() / part_num) + 1;
        part_num_rest = static_cast<int>(part_clouds.size() % part_num);
        all_count = 0;*/

        // 默认设置为仅 1 块（即把上一次的 part_clouds 全部作为一块继续处理）
        part_loop = 1;

        // 将所有 part_clouds 的数量作为余数
        part_num_rest = part_clouds.size();

        // 重置计数器
        all_count = 0;
    }

    // 如果最后一块刚好能整除，没有余数，则总块数减 1
    if (part_num_rest == 0)
    {
        part_loop--;
    }

    // 如果使用预定义的 pairs（use_pairs==true），
    // 说明不按分块方式生成 pairs，而是直接从磁盘读取 pairs，
    // 因此强制设置只有 1 块
    if (use_pairs)
    {
        part_loop = 1;
    }
}

/**
 * @brief 针对给定块号 block_id 生成该块内部的扫描对（pairs）以及块内扫描数量 nr_scans。
 *
 * 规则：
 * - 常规情况下，块内按照两两组合生成所有边（i<j）；
 * - 处理最后一块时考虑余数 `part_num_rest`；
 * - 当 `use_pairs==true` 时，从磁盘读入 pairs（readPairs）。
 * 同时更新块名（filename），后续用于输出结果文件命名。
 *
 *  @param block_id 当前块id
 *
 */
void WHU::HL_MRF::blockPartition(int block_id)
{
    // 清空上一轮生成的扫描对
    pairs.clear();

    // 清空块名（字符串流），用于后续输出文件命名
    filename.str("");

    // 当前块内的扫描数量计数器
    nr_scans = 0;

    // ====== 设置块名 ======
    if (bigloop == 0) // 第一次大循环：原始文件块
    {
        filename << "/block" << block_id; // 输出命名为 "blockX"
    }
    else // 后续大循环：对已配准的子块继续处理
    {
        filename << "/Final_aligned_scan" << block_id; // 命名为 "Final_aligned_scanX"
    }

    // ====== 情况 1: 非最后一块（且存在多个块） ======
    // - 每块按 part_num 个扫描组成
    // - 生成块内所有的两两组合 (i<j)，保存到 pairs 中
    if (part_loop > 1 && block_id < part_loop - 1)
    {
        for (int i = 0; i < part_num; i++)
        {
            for (int j = i + 1; j < part_num; j++)
            {
                pairs.push_back(std::make_pair(i, j)); // 添加扫描对
                // overlap_score.push_back(1 - all_dists[i][j]); // （可选）保存重叠度
            }
            nr_scans++; // 块内扫描数递增
        }
    }

    // ====== 情况 2: 最后一块且余数存在 ======
    // - 当总文件数不能整除 part_num 时，最后一块只含余数 part_num_rest 个扫描
    // - 仍然生成块内所有两两组合
    if (block_id == part_loop - 1 && part_num_rest != 0)
    {
        for (int i = 0; i < part_num_rest; i++)
        {
            for (int j = i + 1; j < part_num_rest; j++)
            {
                pairs.push_back(std::make_pair(i, j));
            }
            nr_scans++;
        }
    }

    // ====== 情况 3: 最后一块且刚好整除 ======
    // - 说明最后一块依然包含 part_num 个扫描
    // - 同样生成完全图的扫描对
    if (block_id == part_loop - 1 && part_num_rest == 0)
    {
        for (int i = 0; i < part_num; i++)
        {
            for (int j = i + 1; j < part_num; j++)
            {
                pairs.push_back(std::make_pair(i, j));
            }
            nr_scans++;
        }
    }

    // ====== 情况 4: 使用预定义 pairs 文件 ======
    // - 忽略分块逻辑，直接从磁盘读取 pairs
    // - 设置 nr_scans 为文件总数（全部参与配准）
    if (use_pairs)
    {
        // readPairs(pair_path); // 如果有特定路径
        readPairs();             // 读取外部定义的配对关系
        nr_scans = files.size(); // 块内扫描数即为全部扫描
    }
}

/**
 * @brief 将上一轮分块好的点云/关键点/特征转移（或就地移动）到当前块的容器中。
 *
 * 该函数根据当前注册阶段（`bigloop` 值）以及分块情况（`part_loop`、`part_num_rest`），
 * 将成员变量中的点云数据 (`part_clouds`、`part_keypoint_clouds`、`part_keypoint_feature_clouds`)
 * 转移到临时容器 (`clouds`、`keypoint_clouds`、`keypoint_clouds_feature`) 中，
 * 用于后续的粗配准与精配准。
 *
 * @param[out] keypoint_clouds          输出的关键点点云集合，每个元素对应一幅扫描的关键点。
 * @param[out] keypoint_clouds_feature  输出的关键点特征集合（FPFH特征），与关键点点云对应。
 * @param[out] clouds                   输出的稀疏点云集合（经过体素滤波或降采样）。
 * @param[in]  block_id                 当前处理的分块 ID。
 *
 * 说明：当 bigloop==1 且继续分块时，从 part_* 向局部 vectors 移动（std::move），避免拷贝。
 */
void WHU::HL_MRF::transferVaribles(
    std::vector<pcl::PointCloud<PointT>::Ptr> &keypoint_clouds,
    std::vector<pcl::PointCloud<pcl::FPFHSignature33>::Ptr> &keypoint_clouds_feature,
    std::vector<pcl::PointCloud<PointT>::Ptr> &clouds,
    int block_id)
{
    if (bigloop == 0) // 初始阶段（对原始扫描进行分块）
    {
        // 分配内存，大小为当前块内的扫描数 nr_scans
        keypoint_clouds.resize(nr_scans);
        keypoint_clouds_feature.resize(nr_scans);
        clouds.resize(nr_scans);
    }
    else // 继续分块时（已进入 block-to-block 注册阶段）
    {
        keypoint_clouds.resize(nr_scans);
        keypoint_clouds_feature.resize(nr_scans);
        clouds.resize(nr_scans);

        // 如果当前块不是最后一个，直接拷贝 part_num 个点云
        if (part_loop > 1 && block_id < part_loop - 1)
        {
            for (int i = 0; i < part_num; i++)
            {
                // 使用 std::move 将对应的子块数据转移到临时容器，避免拷贝
                clouds[i] = std::move(part_clouds[block_id * part_num + i]);
                keypoint_clouds[i] = std::move(part_keypoint_clouds[block_id * part_num + i]);
                keypoint_clouds_feature[i] = std::move(part_keypoint_feature_clouds[block_id * part_num + i]);
            }
        }

        // 如果当前是最后一个块，且没有剩余（正好整除）
        if (block_id == part_loop - 1 && part_num_rest == 0)
        {
            for (int i = 0; i < part_num; i++)
            {
                clouds[i] = std::move(part_clouds[block_id * part_num + i]);
                keypoint_clouds[i] = std::move(part_keypoint_clouds[block_id * part_num + i]);
                keypoint_clouds_feature[i] = std::move(part_keypoint_feature_clouds[block_id * part_num + i]);
            }
        }
        // 如果当前是最后一个块，且存在剩余（最后一块不足 part_num）
        else if (block_id == part_loop - 1 && part_num_rest != 0)
        {
            for (int i = 0; i < part_num_rest; i++)
            {
                clouds[i] = std::move(part_clouds[block_id * part_num + i]);
                keypoint_clouds[i] = std::move(part_keypoint_clouds[block_id * part_num + i]);
                keypoint_clouds_feature[i] = std::move(part_keypoint_feature_clouds[block_id * part_num + i]);
            }
        }
    }
}

/**
 * @brief 从 `PLYpath` 目录读取所有点云文件（.ply 或 .pcd 不在此处过滤），并按文件名排序。
 * @return 1 表示成功；0 表示失败（路径不存在或非目录）。
 */
bool WHU::HL_MRF::readPLYfiles()
{
    if (!boost::filesystem::exists(PLYpath))
    {
        std::cerr << "...path does not exists!\n";
        return false;
    }
    if (!boost::filesystem::is_directory(PLYpath))
    {
        std::cerr << "...path is not a directory!\n";
        return false;
    }
    for (boost::filesystem::directory_entry &x : boost::filesystem::directory_iterator(PLYpath))
    {
        files.push_back(x.path());
    }
    std::sort(files.begin(), files.end());
    return !files.empty();
}

/**
 * @brief 从 `pairs_dir` 目录解析配对文件名，生成 scans 对应的配对关系列表 pairs。
 *
 * 假设文件名中包含形如 sXX_tYY 的两位编号，解析后转为从 0 开始的索引（-1）。
 */

int WHU::HL_MRF::readPairs()
{
    pairs.clear();
    // pairs.resize(files.size());
    for (boost::filesystem::directory_entry &x : boost::filesystem::directory_iterator(pairs_dir))
    {
        pairs_path.push_back(x.path());
    }
    for (int i = 0; i < pairs_path.size(); i++)
    {
        char s1 = pairs_path[i].filename().string()[1];
        char s1_ = pairs_path[i].filename().string()[2];
        char s2 = pairs_path[i].filename().string()[5];
        char s2_ = pairs_path[i].filename().string()[6];
        int s1_i = s1 - '0';
        s1_i = s1_i * 10;
        int s1_i_ = s1_ - '0';

        s1_i = s1_i + s1_i_ - 1;
        int s2_i = s2 - '0';
        s2_i = s2_i * 10;
        int s2_i_ = s2_ - '0';
        s2_i = s2_i + s2_i_ - 1;
        pairs.push_back(std::make_pair(s1_i, s2_i));
    }
    return 1;
}

/**
 * @brief 基于八叉树叶节点 + VoxelGrid 的混合体素下采样。
 *
 * 该函数实现了一种两阶段的降采样方法：
 * 1. 使用八叉树 (Octree) 将点云划分为空间叶子节点；
 * 2. 对每个叶子节点内的点再应用体素滤波 (VoxelGrid)，得到代表点；
 * 3. 最终输出为所有叶子节点的降采样点集合。
 *
 * @param[in]  cloud_in        输入点云（原始点云，类型 PointT）。
 * @param[out] cloud_out       输出点云（经过降采样后的结果）。
 * @param[in]  downsample_size 体素大小（单位：米），控制降采样精度。
 *
 * @return 返回降采样后点云的点数。
 */
int WHU::HL_MRF::sampleLeafsized(pcl::PointCloud<PointT>::Ptr &cloud_in,
                                 pcl::PointCloud<PointT> &cloud_out,
                                 float downsample_size)
{
    pcl::PointCloud<PointT> cloud_sub; // 临时存放某个叶子节点经过体素滤波后的点
    cloud_out.clear();                 // 清空输出点云，避免累加旧数据

    // ========================== 1. 计算八叉树叶子大小 ==========================
    // leafsize 用于构建八叉树的空间分辨率。
    // 这里采用了一个缩放公式：downsample_size * (max_int^(1/3) - 1)
    // 作用是将用户指定的 downsample_size 映射到八叉树内部索引范围。
    float leafsize = downsample_size *
                     (std::pow(static_cast<int64_t>(std::numeric_limits<int32_t>::max()) - 1, 1.0 / 3.0) - 1);

    // ========================== 2. 构建八叉树 (Octree) ==========================
    pcl::octree::OctreePointCloud<PointT> oct(leafsize); // 基于 leafsize 的 octree
    oct.setInputCloud(cloud_in);                         // 设置输入点云
    oct.defineBoundingBox();                             // 定义八叉树的包围盒（自动计算范围）
    oct.addPointsFromInputCloud();                       // 将输入点云插入到八叉树节点中

    // ========================== 3. 配置体素滤波器 (VoxelGrid) ==========================
    pcl::VoxelGrid<PointT> vg;                                         // 体素滤波器
    vg.setLeafSize(downsample_size, downsample_size, downsample_size); // 设置体素尺寸
    vg.setInputCloud(cloud_in);                                        // 设置输入点云（整云）

    // 获取八叉树叶子节点的数量（仅用于信息统计）
    size_t num_leaf = oct.getLeafCount();

    // ========================== 4. 遍历八叉树的所有叶子节点 ==========================
    // 遍历八叉树中每个叶子节点，提取点索引并在局部执行体素滤波。
    for (auto it = oct.leaf_depth_begin(), it_end = oct.leaf_depth_end(); it != it_end; ++it)
    {
        // 存储当前叶子节点中包含的点索引
        pcl::IndicesPtr ids(new std::vector<int>);

        // 将迭代器转换为叶子节点类型，并获取点索引集合
        pcl::octree::OctreePointCloud<PointT>::LeafNode *node =
            (pcl::octree::OctreePointCloud<PointT>::LeafNode *)*it;
        node->getContainerPtr()->getPointIndices(*ids);

        // 将体素滤波器作用于当前叶子节点的点
        vg.setIndices(ids);   // 设置滤波点的索引
        vg.filter(cloud_sub); // 滤波结果存入 cloud_sub（可能仅包含一个点）

        // 将当前叶子节点的滤波结果加入到最终输出点云
        cloud_out += cloud_sub;
    }

    // ========================== 5. 返回降采样结果 ==========================
    return static_cast<int>(cloud_out.size()); // 输出点数作为返回值
}

/**
 * @brief 根据扩展名加载 .ply 或 .pcd 点云到 cloud。
 * @param[in]  filename 点云路径
 * @param[out] cloud    读取后的点云（Ptr）
 */

void WHU::HL_MRF::readPointCloud(const boost::filesystem::path &filename,
                                 pcl::PointCloud<PointT>::Ptr cloud)
{
    if (!filename.extension().string().compare(".ply"))
    {
        pcl::io::loadPLYFile(filename.string(), *cloud);
        return;
    }
    if (!filename.extension().string().compare(".pcd"))
    {
        pcl::io::loadPCDFile(filename.string(), *cloud);
        return;
    }
}

/**
 * @brief 预处理（并行）：加载原始点云 -> 体素下采样 -> 提取 ISS 关键点 -> 计算 FPFH 特征。
 * ISS（Intrinsic Shape Signatures）通过 特征值分解 判断某点局部曲率差异是否足够显著，得到内部形状的稀疏点
 *
 * 该函数对输入点云集合进行预处理，流程包括：
 * 1. 读取并加载点云文件；
 * 2. 进行体素下采样（用于粗配准和ICP）；
 * 3. 提取 ISS (Intrinsic Shape Signatures) 关键点；
 * 4. 基于关键点计算 FPFH (Fast Point Feature Histogram) 特征。
 *
 * 处理结果存储到传入的 `clouds`、`keypoint_clouds` 和 `keypoint_clouds_feature` 中，
 * 用于后续的粗配准与全局配准。
 *
 * @param[out] keypoint_clouds          提取到的当前块内的所有扫描的 ISS 关键点集合。
 * @param[out] keypoint_clouds_feature  提取到的当前块内的所有扫描的关键点特征集合（FPFH 特征）。
 * @param[out] clouds                   下采样后的当前块内的所有扫描的点云集合（ICP 使用版本）。
 * @param[in]  block_id                 当前处理的分块 ID。
 */
void WHU::HL_MRF::preprocessing(std::vector<pcl::PointCloud<PointT>::Ptr> &keypoint_clouds,
                                std::vector<pcl::PointCloud<pcl::FPFHSignature33>::Ptr> &keypoint_clouds_feature,
                                std::vector<pcl::PointCloud<PointT>::Ptr> &clouds,
                                int block_id)
{
    // 使用 OpenMP 并行加速处理当前块内在每个点云扫描
#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < nr_scans; i++)
    {
        // ========================== 1. 加载点云 ==========================
        cout << "Processing " << files[block_id * part_num + i].stem().string() << ":" << endl;
        cout << "..loading point cloud..";

        pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
        // 从磁盘文件读取点云（支持 PLY/PCD）
        readPointCloud(files[block_id * part_num + i], cloud);
        // 保存原始点云到 clouds 容器
        clouds[i] = cloud;

        // ========================== 2. 体素下采样 ==========================
        cout << "..apply voxel grid filter..\n";
        pcl::PointCloud<PointT>::Ptr voxel_cloud(new pcl::PointCloud<PointT>);
        pcl::PointCloud<PointT>::Ptr voxel_cloud_icp(new pcl::PointCloud<PointT>);

        // 用较大的 voxel_size 下采样，便于关键点提取和特征计算
        sampleLeafsized(clouds[i], *voxel_cloud, voxel_size);
        // 用较小的 voxel_size_icp 下采样，用于后续 ICP 精细配准
        sampleLeafsized(clouds[i], *voxel_cloud_icp, voxel_size_icp);
        // 将 ICP 版本的下采样结果存回 clouds
        clouds[i] = voxel_cloud_icp;

        // ========================== 3. ISS 关键点提取 ==========================
        pcl::PointCloud<PointT>::Ptr issS(new pcl::PointCloud<PointT>);
        pcl::PointIndicesPtr issIdxS(new pcl::PointIndices);
        std::cout << "extracting ISS keypoints..." << voxel_size << std::endl;

        // 调用 GROR 库中的 ISS 提取函数
        GrorPre::issKeyPointExtration(voxel_cloud, issS, issIdxS, voxel_size);

#ifdef MRF_DEBUG
        pcl::io::savePCDFileASCII(std::string(ROOT_DIR) + "/data/" + std::to_string(block_id * part_num + i + 1) + "_ds.pcd",
                                  *voxel_cloud);
        pcl::io::savePCDFileASCII(std::string(ROOT_DIR) + "/data/" + std::to_string(block_id * part_num + i + 1) + "_iss.pcd",
                                  *issS);
#endif

        std::cout << "size of issS = " << issS->size() << "/ " << voxel_cloud->size() << std::endl;

        // 标记点云为稀疏（is_dense = false），表示可能含有 NaN/无效点
        issS->is_dense = false;
        keypoint_clouds[i] = issS;

        // ========================== 4. FPFH 特征计算 ==========================
        std::cout << "computing fpfh..." << std::endl
                  << std::endl;
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhS(new pcl::PointCloud<pcl::FPFHSignature33>());

        // 基于 ISS 提取的关键点索引，在体素下采样点云上计算 FPFH 特征
        GrorPre::fpfhComputation(voxel_cloud, voxel_size, issIdxS, fpfhS);
        keypoint_clouds_feature[i] = fpfhS;

        cout << "ok!\n";
    }
}

/**
 * @brief 粗配准（并行）：基于 GROR（全局鲁棒配准）在每对扫描之间计算初始位姿。
 *
 * 本函数执行点云的粗配准步骤，主要通过以下流程：
 * 1. 遍历所有点云对 (src, tgt)，基于关键点特征 (FPFH) 搜索候选对应点。
 * 2. 使用 GRORInitialAlignment 算法进行初始配准，估计出最佳刚体变换。
 * 3. 保存每个点云对的匹配候选结果 (MatchingCandidate)，包括估计的变换矩阵和匹配评分。
 *
 * @param[in] keypoint_clouds             输入的关键点点云集合（每个点云是原始点云的关键点子集）。
 * @param[in] keypoint_clouds_feature     每个关键点点云对应的 FPFH 特征集合。
 * @param[in] clouds                      原始点云集合（可用于后续 RMSE 计算）。
 * @param[out] pairs_best_count            输出，每个点云对的最佳匹配点数量。
 * @param[out] candidate_matches           输出，每个点云对的配准候选结果（包含变换矩阵和适应度分数）。
 *
 * @note 使用 OpenMP 并行加速，每个点云对的配准任务独立。
 * @note 这里采用 GROR 方法（而非传统 SDOFGR 或 RANSAC），能提升配准鲁棒性。
 */
void WHU::HL_MRF::coarseRgistration(
    std::vector<pcl::PointCloud<PointT>::Ptr> &keypoint_clouds,
    std::vector<pcl::PointCloud<pcl::FPFHSignature33>::Ptr> &keypoint_clouds_feature,
    std::vector<pcl::PointCloud<PointT>::Ptr> &clouds,
    std::vector<int> &pairs_best_count,
    std::vector<pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>> &candidate_matches)
{
    // 输出一个进度条（用“-”表示），长度等于待匹配的点云对数量 nr_pairs
    for (int i = 0; i < nr_pairs; i++)
    {
        cout << "-";
    }
    cout << "\n";

    // 判断当前粗配准方法是否为 GROR
    if (method == CoarseRegistration::GROR)
    {
        // 使用 OpenMP 并行化所有点云对的处理
#pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < nr_pairs; i++)
        {
            int n_optimal = 800;              // 最优采样点数量，用于 GROR 算法
            const int &src = pairs[i].first;  // 源点云 ID
            const int &tgt = pairs[i].second; // 目标点云 ID

            int maxCorr = 5;                                        // 每个点最多找 5 个候选对应点
            pcl::CorrespondencesPtr corr(new pcl::Correspondences); // 存储匹配对应点对
            std::vector<int> corrNOS, corrNOT;                      // 对应点在 src/tgt 中的索引（可调试用）

            // 使用 FPFH 特征进行对应点搜索
            GrorPre::correspondenceSearching(
                keypoint_clouds_feature[src],
                keypoint_clouds_feature[tgt],
                *corr, maxCorr, corrNOS, corrNOT);

            // GROR 初始配准器
            pcl::registration::GRORInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, float> obor;
            pcl::PointCloud<pcl::PointXYZ>::Ptr pcs(new pcl::PointCloud<pcl::PointXYZ>);

            // 将模板类型 PointT 转换为 pcl::PointXYZ（GROR 要求输入格式）
            pcl::PointCloud<pcl::PointXYZ>::Ptr temp_src(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr temp_tgt(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::copyPointCloud<PointT, pcl::PointXYZ>(*keypoint_clouds[src], *temp_src);
            pcl::copyPointCloud<PointT, pcl::PointXYZ>(*keypoint_clouds[tgt], *temp_tgt);

            // 设置 GROR 参数
            obor.setInputSource(temp_src);             // 源关键点
            obor.setInputTarget(temp_tgt);             // 目标关键点
            obor.setResolution(voxel_size);            // 分辨率，用于采样
            obor.setOptimalSelectionNumber(n_optimal); // 设置最优采样数量
            obor.setNumberOfThreads(1);                // 单线程执行（避免内部竞争）
            obor.setInputCorrespondences(corr);        // 输入候选对应点
            obor.setDelta(voxel_size);                 // Delta 参数（搜索步长）
            obor.align(*pcs);                          // 执行初始配准

            // 保存结果
            pcl::registration::MatchingCandidate result;
            result.transformation = obor.getFinalTransformation(); // 估计的刚体变换矩阵
            result.fitness_score = obor.getMSAC();                 // MSAC 评分（匹配优劣）
            pairs_best_count[i] = obor.getBestCount();             // 匹配到的最佳对应点数量

            // 使用 critical 区域避免并发写冲突
#pragma omp critical
            {
                candidate_matches[i] = result; // 保存结果到全局数组
                cout << "*";                   // 输出进度
                cout.clear();
            }
        }
        cout << "\n";
    }
}

/**
 * @brief 执行基于环约束的全局粗配准（global coarse registration）。
 *
 * 本函数通过 **环一致性约束 (loop constraint inference)** 对多扫描数据的配准对进行一致性推理，
 * 剔除不符合全局约束的错误配准对。
 * 在多视角 TLS 点云配准任务中，环一致性是提高整体鲁棒性的关键步骤。
 *
 * @param[out] rejected_pairs_   输出被拒绝的扫描对索引集合（即被判定为不一致或不可靠的配准对）。
 * @param[in]  candidate_matches 每个扫描对的候选匹配结果（包含可能的刚体变换候选解）。
 * @param[in]  pairs_best_count  每个扫描对的“最佳一致支持数”，用于衡量该配准对的稳定性与支持度。
 * @param[in]  nr_scans          总的扫描数量，只有当扫描数 ≥ 3 时才可能形成环路从而执行全局一致性推理。
 *
 * @note
 * - 仅当扫描数大于等于 3，且不使用单纯的 pairwise 配准模式 (`use_pairs == false`) 时，本函数才会执行。
 * - 内部调用 `ScanGraphInference` 类进行推理，该类通过图模型推断出不满足一致性的扫描对。
 * - 当前实现只执行基本推理逻辑，代码中提供的 **循环一致性 (cyclic constraints)** 和 **阈值判别** 逻辑已被注释，
 *   如需更严格的约束可重新启用。
 */
void WHU::HL_MRF::globalCoarseRegistration(
    std::set<int> &rejected_pairs_,
    std::vector<pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>> &candidate_matches,
    std::vector<int> &pairs_best_count,
    int nr_scans)
{
    // 只有在扫描数 ≥ 3 且没有强制使用 pairwise 模式时，才执行基于环的全局粗配准
    if (nr_scans >= 3 && !use_pairs)
    {
        rejected_pairs_.clear(); // 清空上一次运行的结果
        std::cout << "--> loop-based coarse regisration. \n";

        // 创建环约束推理器
        ScanGraphInference sgi;

        // 向推理器传入必要数据：
        sgi.setScanPairs(pairs);                          // 设置扫描对关系（即配准的候选边）
        sgi.setMatchingCandidates(candidate_matches);     // 设置每个扫描对的候选匹配变换
        sgi.setMCS(pairs_best_count);                     // 设置每个扫描对的一致性支持度
        sgi.setRotationAccuracy(rotation_accuracy);       // 设置允许的旋转误差阈值
        sgi.setTranslationAccuracy(translation_accuracy); // 设置允许的平移误差阈值

        // 执行推理：
        rejected_pairs_ = sgi.inference(3, 10);

        std::cout << "IES size: " << rejected_pairs_.size() << "\n";

        // ==============================
        // 下面是额外的一致性检查逻辑，但已被注释掉：
        // - ho_Constraints_true：有效环约束
        // - ho_Constraints：无效环约束
        // - use_cyclic_constraints：是否强制使用循环一致性
        // - max_consensus_set：支持度阈值
        //
        // 若启用，可进一步基于循环一致性或支持度阈值来剔除更多错误配准对。
        // ==============================
    }
}

/**
 * @brief 全局细配准流水线（MST 选边 → 边上 ICP 精配准 → 位姿拼接 → LUM 图优化）。
 *
 * @details
 * 该函数在“全局粗配准 + 一致性推断”之后运行，假设已经得到可靠的配准边集合（去除了明显的不一致边）。
 * 其内部包含四个步骤：
 *  1) mstCalculation：在去除不一致边后，按连通子图计算最小生成树（Prim），并建立 LUM 的局部/全局索引映射；
 *  2) pairwiseICP：仅在 MST 边上执行点到点 ICP 微调，更新匹配的相对变换；
 *  3) concatenateFineRegistration：沿 DFS 路径累计边变换，得到每个子图相对于根的绝对位姿；
 *  4) LUMoptimization：以上一步的绝对位姿为初值，执行 LUM 全局图优化，进一步整体细化。
 *
 * @param[in]  rejected_pairs_  从一致性推断得到的“拒绝边”索引集合（这些边会被忽略）。
 * @param[in,out] matches       （输入）粗配准得到的边变换和打分；（输出）ICP 可能会更新其中的 transformation。
 * @param[in]  clouds           参与 ICP/LUM 的（已下采样）点云集合，序号与扫描索引一致。
 * @param[out] poses            每个连通子图的绝对位姿解（按全局索引存放，未出现的索引为 Zero()）。
 * @param[out] num_of_subtrees  连通子图数量（= connected components 数目）。
 *
 * @pre 已调用 globalCoarseRegistration() 并完成不一致边剔除；
 * @pre clouds 已按 voxel_size_icp 下采样；
 * @post poses[f][i] 给出第 f 个子图中第 i 号扫描的绝对位姿（若 i 不在该子图中则为 Zero()）。
 *
 * @note 仅当 nr_scans ≥ 3 时才执行（小于 3 无法形成环也没有必要做 MST/LUM）。
 * @see mstCalculation(), pairwiseICP(), concatenateFineRegistration(), LUMoptimization()
 */
void WHU::HL_MRF::globalFineRegistration(std::set<int> &rejected_pairs_,
                                         std::vector<pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>> &matches,
                                         std::vector<pcl::PointCloud<PointT>::Ptr> &clouds,
                                         std::vector<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>> &poses,
                                         int &num_of_subtrees)
{
    if (nr_scans >= 3) // 仅在扫描数≥3时才有必要做 MST/ICP/LUM
    {
        // ===== 临时变量区（仅在本函数内使用）====================================
        std::vector<std::vector<int>> LUM_indices_map_;                                                                                        // LUM：全局索引 -> 子图局部索引 的映射表（逐子图）
        std::vector<std::vector<int>> LUM_indices_map_inv_;                                                                                    // LUM：子图局部索引 -> 全局索引 的反向映射表（逐子图）
        std::vector<std::vector<size_t>> edges;                                                                                                // 每个子图的 MST 边集合（边用 after_check_graph_pairs 的下标标识）
        std::vector<int> root;                                                                                                                 // 每个子图选择的根节点（全局索引）
        std::vector<std::pair<int, int>> after_check_graph_pairs;                                                                              // 去除不一致边后的“边列表”（以全局节点索引存放）
        std::vector<pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>> after_check_matches; // 与上面边一一对应的匹配结果
        std::vector<std::vector<pcl::Indices>> nodes_seqs;                                                                                     // 每个子图内的 DFS 路径序列（用于位姿拼接）

        // ===== Step 1: 构图 + MST + LUM 索引映射 ==================================
        // 作用：在去除不一致边后，按连通分量划分子图，对每个子图计算 MST；
        //       同时建立 LUM 的局部/全局索引映射，方便后续图优化使用局部编号。
        mstCalculation(rejected_pairs_, matches,
                       LUM_indices_map_, LUM_indices_map_inv_,
                       edges, root, after_check_graph_pairs, after_check_matches,
                       num_of_subtrees);

        // ===== Step 2: 边上 ICP 微调 ===============================================
        // 作用：仅对 MST 边做点到点 ICP，利用粗配准结果作为初值，对 transformation 进行细化。
        pairwiseICP(edges, after_check_graph_pairs, after_check_matches, num_of_subtrees, clouds);

        // ===== Step 3: 位姿拼接（从根出发，沿 DFS 路径连乘边变换）===================
        // 作用：将“相对边变换”累乘为“相对于根的绝对位姿”，为 LUM 提供良好的初值。
        concatenateFineRegistration(edges, root, after_check_graph_pairs, after_check_matches,
                                    num_of_subtrees, poses, nodes_seqs);

        // ===== Step 4: LUM 全局优化 =================================================
        // 作用：以 Step 3 的绝对位姿为初值，在每个子图上进行 LUM 图优化，进一步整体收敛。
        LUMoptimization(poses, edges, root, nodes_seqs,
                        LUM_indices_map_, LUM_indices_map_inv_,
                        num_of_subtrees, clouds);
    }
}

/**
 * @brief 依据已过滤的一致性边构建子图并在每个连通子图上计算最小生成树（MST），
 *        同时建立 LUM 优化所需的“全局索引 ↔ 子图局部索引”映射与 MST 边列表。
 *
 * @details
 * 流程分为四步：
 *  1) 使用 rejected_pairs_ 过滤掉不一致的图边，将剩余边加入全局图 G，并记录其匹配权重（fitness_score）。
 *  2) 计算 G 的连通分量，将顶点划分到各个子图（connected components），统计子图个数 num_of_subtrees。
 *  3) 为每个子图创建 Boost::subgraph，并将属于该子图的顶点（以“全局索引”）插入到子图（“局部索引”空间）。
 *     将 after_check_graph_pairs 中位于该子图的边按“局部索引”加入子图，并写入边权重。
 *  4) 在每个子图运行 Prim MST，抽取 MST 边，将“局部边索引”映射回“全局边索引”（对应 after_check_graph_pairs 的次序）。
 *     同时构建 LUM_indices_map_ 与 LUM_indices_map_inv_，并为每个子图选取一个根（Prim 产生的根）。
 *
 * @param[in]  rejected_pairs_         需要忽略（已判定为不一致）的边在 pairs/matches 中的下标集合。
 * @param[in,out] matches              所有候选匹配边（与全局 pairs 一一对应）；本函数会将被保留的边附加到 after_check_matches。
 * @param[out] LUM_indices_map_        逐子图的“全局顶点索引 -> 子图局部顶点索引”映射（size=num_of_subtrees）。
 * @param[out] LUM_indices_map_inv_    逐子图的“子图局部顶点索引 -> 全局顶点索引”反向映射（size=num_of_subtrees）。
 * @param[out] edges                   逐子图的 MST 边索引列表；每个元素为 size_t，指向 after_check_graph_pairs/after_check_matches 的条目。
 * @param[out] root                    逐子图的根节点（以全局顶点索引表示）。
 * @param[out] after_check_graph_pairs 过滤后保留的“全局边”（顶点对），与 after_check_matches 一一对应。
 * @param[out] after_check_matches     过滤后保留的匹配记录（与 after_check_graph_pairs 同步追加）。
 * @param[out] num_of_subtrees         连通子图数量（connected components）。
 *
 * @pre  成员变量 nr_scans、nr_pairs、pairs 已正确初始化；matches 与 pairs 等长且一一对应。
 * @post edges[i] 内的每个 size_t 均为 after_check_graph_pairs 的合法下标；LUM 索引映射完成初始化。
 *
 * @note
 *  - 本函数不做 ICP 与 LUM 优化，仅构建 MST 与各种映射，为后续步骤提供结构与初值。
 *  - 边权重使用 matches[i].fitness_score；若其语义是“越小越好”，Prim MST 将倾向选择拟合度更优的边。
 *  - 使用 boost::subgraph 以便处理“全局/局部索引”问题：MST 在局部图上求解，再映射回全局。
 */
void WHU::HL_MRF::mstCalculation(std::set<int> &rejected_pairs_,
                                 std::vector<pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>> &matches,
                                 std::vector<std::vector<int>> &LUM_indices_map_,
                                 std::vector<std::vector<int>> &LUM_indices_map_inv_,
                                 std::vector<std::vector<size_t>> &edges,
                                 std::vector<int> &root,
                                 std::vector<std::pair<int, int>> &after_check_graph_pairs,
                                 std::vector<pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>> &after_check_matches,
                                 int &num_of_subtrees)
{
    Graph G(nr_scans); // 全局图：包含 nr_scans 个顶点（顶点索引约定与“全局扫描索引”一致）

    cout << "compute MST\n";
    std::vector<float> edge_weights; // 与 after_check_graph_pairs/after_check_matches 的顺序严格一致的边权重数组
    edge_weights.reserve(static_cast<size_t>(nr_pairs));
    after_check_graph_pairs.reserve(static_cast<size_t>(nr_pairs));
    after_check_matches.reserve(static_cast<size_t>(nr_pairs));
    // === 1) 过滤不一致边并构建全局图 G ===
    for (std::size_t i = 0; i < nr_pairs; i++)
    {
        if (!rejected_pairs_.empty())
        {
            // 如果边 i 在拒绝集合中，则跳过
            if (find(rejected_pairs_.begin(), rejected_pairs_.end(), static_cast<int>(i)) != rejected_pairs_.end())
            {
                cout << "ignore " << i << ", it's rejected" << endl;
                continue;
            }
        }

        // 从 pairs 取全局端点索引
        const int &src = pairs[i].first;
        const int &tgt = pairs[i].second;

        add_edge(src, tgt, G);                                       // 在全局图中加入边（不设置权重；权重只在子图上用）
        edge_weights.push_back(matches[i].fitness_score);            // 记录该边的权重（与 after_check_* 一致）
        after_check_matches.push_back(matches[i]);                   // 记录过滤后保留的匹配
        after_check_graph_pairs.push_back(std::make_pair(src, tgt)); // 同步记录过滤后保留的边（全局端点）
    }
    const size_t kept_edges = after_check_graph_pairs.size();
    cout << "[MST] build global graph | scans=" << nr_scans
         << " | candidate_edges=" << nr_pairs
         << " | kept=" << kept_edges << endl;

    // === 2) 连通分量划分 ===
    std::cout << "component calculation...";
    std::vector<int> component(num_vertices(G));              // component[v] = 子图编号
    num_of_subtrees = connected_components(G, &component[0]); // 统计子图个数
    edges.resize(num_of_subtrees);                            // 为每个子图预留 MST 边列表

    cout << "subgraphs size: " << num_of_subtrees << endl;
    std::vector<std::vector<int>> subgraph_vertices(num_of_subtrees); // 每个子图包含的“全局顶点索引”列表
    for (int i = 0; i < static_cast<int>(component.size()); i++)
    {
        subgraph_vertices[component[i]].push_back(i); // 将全局顶点 i 放入其所属子图的顶点列表
    }

    // === 3) 创建子图并将子图内顶点（以全局索引）插入到各自子图（转为局部索引）===
    // 说明：为了后续调用 prim_minimum_spanning_tree，需要在子图空间中添加边与权重。
    Graph base(nr_scans);           // “根”图（用于创建子图）
    std::vector<Graph *> subgraphs; // 保存子图指针
    for (int c = 0; c < num_of_subtrees; ++c)
    {
        Graph &sg = base.create_subgraph(); // 基于根图创建一个子图
        for (int gv : subgraph_vertices[static_cast<size_t>(c)])
        {
            // 将“全局顶点索引”加入到子图中（内部会分配一个“局部顶点索引”）
            boost::add_vertex(gv, sg);
        }
        subgraphs.push_back(&sg);
    }

    // === 4)把“保留的全局边”分派到各自子图，并设置权重 ----------
    for (size_t i = 0; i < kept_edges; ++i)
    {
        // 找到该边所属的子图编号（利用起点在 component 中的标记）
        const int u = after_check_graph_pairs[i].first;
        const int v = after_check_graph_pairs[i].second;
        const int c = component[u];

        // 取该子图的“边权重属性图”
        Graph &sg = *subgraphs[static_cast<size_t>(c)];
        boost::property_map<Graph, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, sg);

        // 将全局端点索引映射到子图的“局部顶点索引”
        const size_t src = sg.global_to_local(u);
        const size_t tgt = sg.global_to_local(v);

        boost::graph_traits<Graph>::edge_descriptor e_desc;
        bool inserted = false;
        boost::tie(e_desc, inserted) = boost::add_edge(src, tgt, sg); // 在子图（局部索引）中加边
        weightmap[e_desc] = edge_weights[i];                          // 设置该边的权重（顺序与 after_check_* 对齐）
        (void)inserted;                                               // 抑制未使用告警（逻辑上 inserted 一般应为 true）
    }
    cout << "--> find MST edges: \n";

    // === 4) Prim MST & 建立 LUM 索引映射（全局 <-> 局部）===
    LUM_indices_map_.resize(num_of_subtrees);
    LUM_indices_map_inv_.resize(num_of_subtrees);
    for (int i = 0; i < num_of_subtrees; i++)
    {
        // 为简单起见，初始化为 nr_scans 长度；未使用位置保持默认值 0（后续逻辑仅访问子图内的条目）
        LUM_indices_map_[i].resize(nr_scans, 0);
        LUM_indices_map_inv_[i].resize(nr_scans, 0);
    }

    // Prim MST生成
    // p[i][v_local] = 该子图的 Prim 生成树中 v_local 的父结点（局部索引）；
    // 若 p[i][v]==v，则 v 是根
    std::vector<std::vector<boost::graph_traits<Graph>::vertex_descriptor>> p(num_of_subtrees);
    for (int c = 0; c < num_of_subtrees; c++)
    {
        Graph &sg = *subgraphs[static_cast<size_t>(c)]; // 子图
        const size_t nv = num_vertices(sg);             // 子图顶点个数
        p[c].resize(nv);
        if (nv == 1)
            continue;                                    // 单独一个顶点时无需求 MST
        boost::prim_minimum_spanning_tree(sg, &p[c][0]); // 在子图 c 上做 Prim
    }
    for (int c = 0; c < num_of_subtrees; c++)
    {
        Graph &sg = *subgraphs[static_cast<size_t>(c)]; // 子图
        const size_t nv = num_vertices(sg);             // 子图顶点个数
        // 边索引属性：用于把“局部边 (u,v)”映射到 after_check_graph_pairs 的全局边下标
        boost::property_map<Graph, boost::edge_index_t>::type edgemap = boost::get(boost::edge_index, sg);
        // 获取该子图的权重属性映射
        auto weightmap = get(boost::edge_weight, sg);
        cout << "\tMST: ";
        for (int j = 0; j < nv; j++)
        {
            // 维护 LUM 的双向索引映射
            // local_to_global(j) 将“局部顶点 j”映射回“全局顶点索引”
            LUM_indices_map_[c][sg.local_to_global(j)] = j;     // 全局->局部
            LUM_indices_map_inv_[c][j] = sg.local_to_global(j); // 局部->全局

            if (p[c][j] == static_cast<boost::graph_traits<Graph>::vertex_descriptor>(j))
            {
                // p[i][j]==j 说明 j 是 Prim 的根
                cout << "root: " << p[c][j] << "\n";
                root.push_back(sg.local_to_global(j)); // 记录“全局根索引”
                continue;
            }

            // 当前 MST 边 (p[c][j], j) 在局部子图中的表示
            auto [e_desc, ok] = boost::edge(p[c][j], static_cast<size_t>(j), sg);
            // 获取该边的权重
            float w = weightmap[e_desc];

            // 将 Prim 产生的局部树边 (p[i][j], j) 转换为“全局边下标”
            // boost::edge(u_local, v_local, subgraph) 返回 (edge_descriptor, bool)
            edges[c].push_back(edgemap[e_desc]);

            // 转换为全局索引
            int gu = sg.local_to_global(p[c][j]);
            int gv = sg.local_to_global(j);
            // 输出 MST 边及其权重
            cout << "  edge: " << gu << " -- " << gv << " , weight=" << w << endl;
        }
        cout << endl;
    }
}


 /**
  * @brief 在每个连通子图（connected component）的 **MST 边** 上执行点到点 ICP 微调。
  *
  * @details
  * 输入的 @p edges 为“逐子图的边索引列表”，每条边索引指向
  * @p after_check_graph_pairs / @p after_check_matches 的同一位置（即过滤后保留的边）。
  * 函数以 GROR 粗配准得到的相对变换（@p after_check_matches[edge].transformation）作为 **初始位姿**，
  * 仅对该条边关联的两幅点云做 **点到点 ICP**，并将优化后的变换回写到
  * @p after_check_matches[edge].transformation。
  *
  * 并行策略：对每个子图 f 的边列表使用 OpenMP 并行 `for`；每次迭代仅写回该边在
  * @p after_check_matches 中对应的元素，互不冲突。注意：`std::cout` 的输出在并行区可能交错。
  *
  * @param[in]      edges                    逐子图的 MST 边索引；edges[f][i] 是 after_check_* 的下标。
  * @param[in]      after_check_graph_pairs  过滤后保留的“全局边”（两端为全局扫描索引），与 after_check_matches 对齐。
  * @param[in,out]  after_check_matches      每条保留边的配准结果（含 transformation），将被 ICP 更新。
  * @param[in]      num_of_subtrees          连通子图数量（由 MST 计算阶段得到）。
  * @param[in]      clouds                   参与 ICP 的点云数组（与“全局扫描索引”对齐，通常已按 voxel_size_icp 下采样）。
  *
  * @pre  已完成粗配准与一致性推断，并通过 mstCalculation() 产出 @p edges 与 after_check_*。
  * @post 对每条 MST 边，@p after_check_matches[edge].transformation 被 ICP 结果覆盖。
  *
  * @note
  * - ICP 参数：最大对应距离 = @c voxel_size_icp；最大迭代 @c 10；使用互惠对应；
  *   收敛判据 @c EuclideanFitnessEpsilon = 1e-2。
  * - 本实现将点云复制为 `pcl::PointXYZ` 类型，以与类内别名 `PointT=pcl::PointXYZ` 保持一致。
  * - 若需替换为更高层的匹配器（如基于已保存索引的细配准），可将下方 `if (1)` 作为开关位。
  *
  * @thread_safety  每次循环仅写回 `after_check_matches` 的一个独立元素；打印建议使用 `#pragma omp critical` 包裹以避免交错。
  * @complexity     设子图 f 的 MST 边数为 E_f，则总体约为 sum_f O(E_f * N_icp)，其中 N_icp 为 ICP 迭代成本。
  */
void WHU::HL_MRF::pairwiseICP(std::vector<std::vector<size_t>>& edges,
    std::vector<std::pair<int, int>>& after_check_graph_pairs,
    std::vector<pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>>& after_check_matches,
    int& num_of_subtrees,
    std::vector<pcl::PointCloud<PointT>::Ptr>& clouds)
{
    // 遍历每个连通子图（由 MST 阶段得到）
    for (int f = 0; f < num_of_subtrees; f++)
    {
        // 若该子图没有 MST 边，直接跳过（单节点子图或异常情况）
        if (edges[f].empty())
            continue;

        // 在子图 f 内对每条 MST 边并行执行 ICP
#pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < static_cast<int>(edges[f].size()); i++)
        {
            // 取到全局“边下标”，它同时索引 after_check_graph_pairs / after_check_matches
            const size_t edge_idx = edges[f][static_cast<size_t>(i)];

            // 仅用于日志：打印边号（注意并行输出可能交错）
            std::cout << "pairwise icp process edge: " << edge_idx << std::endl;

            // 预留开关：若未来要接入“保存了对应索引的高层匹配”，可在此切换分支
            if (1)
            {
                // 按照“全局边”找到两端扫描的全局 ID（src, tgt）
                const int src_id = after_check_graph_pairs[edge_idx].first;   // 源扫描索引（全局）
                const int tgt_id = after_check_graph_pairs[edge_idx].second;  // 目标扫描索引（全局）

                // 复制两端点云到临时缓冲（类型 pcl::PointXYZ），避免直接修改原数组
                pcl::PointCloud<pcl::PointXYZ>::Ptr src(new pcl::PointCloud<pcl::PointXYZ>);
                pcl::PointCloud<pcl::PointXYZ>::Ptr tgt(new pcl::PointCloud<pcl::PointXYZ>);
                pcl::copyPointCloud(*clouds[src_id], *src);
                pcl::copyPointCloud(*clouds[tgt_id], *tgt);

                // 这里保留了若干被注释掉的实现（如 CCCoreLib ICP/重叠率等），作为备选思路。
                // 当前版本使用 PCL IterativeClosestPoint 进行点到点 ICP。
                pcl::IterativeClosestPoint<PointT, PointT> icp;

                // 若任一端为空则跳过本条边（常见于前序过滤或数据异常）
                if (!src->empty() && !tgt->empty())
                {
                    // === ICP 参数设置 ===
                    // 源/目标点云
                    icp.setInputSource(src);
                    icp.setInputTarget(tgt);

                    // 最大对应距离（单位：m），取类成员 voxel_size_icp；过小可能不收敛，过大易受外点影响
                    icp.setMaxCorrespondenceDistance(voxel_size_icp);

                    // 启用互惠对应（Reciprocal），通常能提升匹配的稳定性但略增计算
                    icp.setUseReciprocalCorrespondences(true);

                    // 迭代上限与收敛判据（欧氏误差阈值）
                    icp.setMaximumIterations(10);
                    icp.setEuclideanFitnessEpsilon(10e-3); // 1e-2，注意这是欧氏 fitness 的收敛阈值

                    // === 对齐执行 ===
                    // 传入 GROR 的相对位姿作为“初始变换”，能显著加快收敛并提高稳定度
                    icp.align(*src, after_check_matches[edge_idx].transformation);

                    // 将最终估计结果回写到该边的 transformation
                    after_check_matches[edge_idx].transformation = icp.getFinalTransformation();

                    // 打印该边两端的全局索引与最终变换（并行环境下输出顺序可能交错）
                    std::cout << src_id << " -- " << tgt_id << "\n";
                    std::cout << after_check_matches[edge_idx].transformation << "\n";
                }
                else
                {
                    // 若数据为空，给出提示（并携带循环内 i 方便定位）
                    std::cout << "ignore " << i << " because one of clouds is empty!\n";
                }
            } // if(1)
        }     // for edges[f]

        std::cout << "icp complete\n";
    } // for subtrees
}

/**
 * @brief 以每个连通子图的 MST 边为约束，将两两配准（ICP/GROR 等）的相对位姿
 *        “沿树累乘”拼接为**全局（以根为参考）**的姿态解，并输出遍历路径。
 *
 * @details
 * - 输入的 @p edges 是“逐子图（connected component）”的边索引列表；
 *   其中 edges[f][i] 是对 after_check_graph_pairs / after_check_matches 的**统一下标**。
 * - 对于每个子图 f：
 *    1. 若该子图无边（单节点子图），则仅将根结点 root[f] 的位姿置为单位阵，其余置零矩阵，
 *       并推入 @p poses 与 @p nodes_seqs。
 *    2. 否则由 edges[f] 构建子图 f 的无向邻接表 Mst，并构造一张“有向边索引表” map_，
 *       用于在 DFS/拼接阶段根据 (u,v) 查得对应的全局边索引 edge_idx 以及方向标记（u->v 是否为正向）。
 *    3. 调用 depthfirstsearch(...) 从根 root[f] 出发生成所有“从根到各结点”的路径序列 nodes_seq；
 *    4. 调用 combineTransformation(...)：沿每条路径将相对位姿累计为“相对根”的全局位姿，
 *       结果写入 poses_t（长度 = nr_scans），并将其推入输出容器 @p poses。
 *
 * 约定与约束：
 * - @p after_check_graph_pairs[edge_idx] = {u, v} 给出边的两端全局扫描 ID（u、v ∈ [0, nr_scans)）。
 * - @p after_check_matches[edge_idx].transformation 存放 **u→v** 的相对变换（或以此为准的初值），
 *   在 combineTransformation 中会依据 map_ 的方向标记决定使用正向或其逆。
 * - @p poses 的 push_back 维度为“子图级别”，即 poses[f][scan_id] 是该子图内某扫描相对于 root[f] 的位姿；
 *   不在该子图中的 scan_id 位置通常保持为 Zero（未赋值）。
 *
 * @param[in]      edges                    逐子图的边索引列表；edges[f][i] 是 after_check_* 的索引。
 * @param[in]      root                     每个子图的根结点全局索引（作为参考坐标系）。
 * @param[in]      after_check_graph_pairs  过滤后的“全局边（u,v）”数组，与 after_check_matches 对齐。
 * @param[in]      after_check_matches      每条边的配准结果（含 transformation），供拼接使用。
 * @param[in]      num_of_subtrees          连通子图数量。
 * @param[out]     poses                    输出：逐子图的全局位姿数组（长度 nr_scans，root 为 I，其余为累乘结果或 Zero）。
 * @param[out]     nodes_seqs               输出：逐子图的 DFS 路径集合；每个元素是一条从 root 出发到某节点的“节点序列”。
 *
 * @pre  已完成 MST 计算并得到了子图划分（root 与 edges），且 after_check_* 二者相互对齐。
 * @post poses.size() 与 nodes_seqs.size() 各自增加 num_of_subtrees 个元素。
 *
 * @note
 * - poses_t 中使用 `Eigen::Matrix4f::Zero()` 表示“未赋值/未知”，`Identity()` 表示“相对于根的单位位姿”。
 * - Mst_trans 在当前实现中仅预留，未实际使用；如需导出“边上的相对变换”序列，可在 combineTransformation 内部填充。
 * - depthfirstsearch / combineTransformation 为类内辅助方法：前者构建从根出发的路径，后者按路径串接相对变换。
 */
void WHU::HL_MRF::concatenateFineRegistration(std::vector<std::vector<size_t>>& edges,
    std::vector<int>& root,
    std::vector<std::pair<int, int>>& after_check_graph_pairs,
    std::vector<pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>>& after_check_matches,
    int& num_of_subtrees,
    std::vector<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>>& poses,
    std::vector<std::vector<pcl::Indices>>& nodes_seqs)
{
    // 逐子图处理
    for (int f = 0; f < num_of_subtrees; f++)
    {
        std::stringstream ss;
        ss << f;

        //========================
        // 情况 A：该子图没有任何边（单节点或异常）
        //========================
        if (edges[f].size() == 0)
        {
            // poses_t：长度 = nr_scans；非本子图中的节点保持 Zero，本子图根设置为 I
            std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
                poses_t(nr_scans, Eigen::Matrix4f::Zero());
            poses_t[root[f]] = Eigen::Matrix4f::Identity();  // 根结点作为参考坐标系

            poses.push_back(poses_t);  // 输出该子图的位姿解

            // nodes_seq：空（无路径）
            std::vector<pcl::Indices> nodes_seq;
            nodes_seqs.push_back(nodes_seq);

            // （可选）历史调试输出留存，当前注释掉
            /*
            ofstream ofs2(".../concatenate registration" + filename.str() + "_Mst" + ss.str() + ".txt");
            for (auto& pose : poses_t) { std::cout << pose << std::endl; ofs2 << pose << std::endl; }
            ofs2.close();
            */
            continue; // 进入下一个子图
        }

        //========================
        // 情况 B：常规子图（含若干条 MST 边）
        //========================

        // 预留：若希望记录“边级别的相对变换链”，可利用此数组；当前未使用
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> Mst_trans(nr_scans - 1);

        // 构建无向邻接表 Mst（按全局节点索引组织）
        std::vector<pcl::Indices> Mst(nr_scans);
        for (int i = 0; i < static_cast<int>(edges[f].size()); i++)
        {
            // 取到该 MST 边在“全局边数组”中的下标
            const size_t edge_idx = edges[f][static_cast<size_t>(i)];
            // 拿到该边的两个端点（全局扫描 ID）
            const int u = after_check_graph_pairs[edge_idx].first;
            const int v = after_check_graph_pairs[edge_idx].second;

            // 无向图邻接：u<->v
            Mst[u].push_back(v);
            Mst[v].push_back(u);
        }

        // 构建“有向边索引表” map_：
        // map_[u * nr_scans + v] = { edge_idx, true }   表示取正向（u->v）
        // map_[v * nr_scans + u] = { edge_idx, false }  表示取反向（v->u，需要在拼接时取逆）
        std::vector<std::pair<int, bool>> map_(nr_scans * nr_scans);
        for (std::size_t i = 0; i < edges[f].size(); i++)
        {
            const size_t edge_idx = edges[f][i];
            const int u = after_check_graph_pairs[edge_idx].first;
            const int v = after_check_graph_pairs[edge_idx].second;

            map_[u * nr_scans + v] = std::make_pair(static_cast<int>(edge_idx), true);   // 正向
            map_[v * nr_scans + u] = std::make_pair(static_cast<int>(edge_idx), false);  // 反向
        }

        // DFS 生成从 root[f] 出发到各节点的“路径序列集合”
        std::vector<pcl::Indices> nodes_seq;  // nodes_seq[k] 是一条路径（节点索引序列）
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
            poses_t(nr_scans, Eigen::Matrix4f::Zero()); // 输出位姿缓冲（先置零）

        depthfirstsearch(nr_scans, root[f], Mst, map_, nodes_seq);
        std::cout << "depth search complete\n";

        // （可选）打印每条从根到节点的路径；用于验证 DFS 结果
        for (const std::vector<int>& item : nodes_seq)
        {
            std::cout << "road :";
            for (int i : item)
                std::cout << i << "\n";
            std::cout << "\n";
        }

        // 将 pairwise 变换沿每条路径进行累乘，得到相对于根的“全局位姿解”，写入 poses_t
        // 注意：combineTransformation 会使用 map_ 中的方向布尔值决定使用 T(u->v) 还是其逆
        combineTransformation(nr_scans, root[f], nodes_seq, map_, after_check_matches, poses_t);

        std::cout << "root " << root[f] << " is the reference frame :\n";

        // （可选）打印该子图内所有节点的最终位姿（Zero 表示未在该子图中或未赋值）
        // 线下保存可用 ofstream；此处仅输出到控制台
        for (Eigen::Matrix4f& pose : poses_t)
        {
            std::cout << pose << std::endl;
        }

        // 推入输出容器：该子图的全局位姿解与 DFS 路径
        poses.push_back(poses_t);
        nodes_seqs.push_back(nodes_seq);
    }
}

/**
 * @brief 从 root 出发做深度优先，枚举到各叶子的路径序列 nodes_seq。
 * @note visited 用于避免回边；path 保存当前路径。
 */

void WHU::HL_MRF::depthfirstsearch(int nr_scans, int root, std::vector<pcl::Indices> &Mst, std::vector<std::pair<int, bool>> &map_, std::vector<pcl::Indices> &nodes_seq)
{
    std::vector<bool> visited(nr_scans, false);

    pcl::Indices path(1, root);
    visited[root] = true;
    // auto leafs =Mst[root];
    std::vector<int> indices = Mst[root];
    pcl::Indices::iterator it = indices.begin();

    for (int i = 0; i < indices.size(); i++)
    {
        int n = indices[i];
        next(n, Mst, map_, nodes_seq, visited, path);
    }
}

/**
 * @brief 递归 DFS 的下一步：加入当前节点，过滤已访问邻居；若无邻居则记录为一条根到叶的路径。
 */

void WHU::HL_MRF::next(int root, std::vector<pcl::Indices> &Mst, std::vector<std::pair<int, bool>> &map_, std::vector<pcl::Indices> &nodes_seq, std::vector<bool> visited, pcl::Indices path)
{
    visited[root] = true;
    path.push_back(root);
    std::vector<int> indices = Mst[root];
    pcl::Indices::iterator it = indices.begin();
    while (it != indices.end())
    {
        if (visited[*it])
            it = indices.erase(it);
        else
            it++;
    }
    if (indices.size() == 0)
    {
        nodes_seq.push_back(path);
        return;
    }

    for (int i = 0; i < indices.size(); i++)
    {
        int n = indices[i];
        next(n, Mst, map_, nodes_seq, visited, path);
    }
}

/**
 * @brief 根据 DFS 得到的所有路径，逐段连乘边变换，计算每个节点相对 root 的位姿。
 * @note map_ 中的 bool 指示变换方向：true 表示沿 (u->v) 使用 inverse(T_uv)；false 表示使用 T_uv。
 */

void WHU::HL_MRF::combineTransformation(int nr_scans, int root,
                                        std::vector<pcl::Indices> &nodes_seq, std::vector<std::pair<int, bool>> &map_,
                                        std::vector<pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>> &matches,
                                        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &poses)
{
    Eigen::Matrix4f combin = Eigen::Matrix4f::Identity();

    poses[root] = Eigen::Matrix4f::Identity();
    for (int i = 0; i < nodes_seq.size(); i++)
    {
        combin = Eigen::Matrix4f::Identity();
        for (int j = 0; j < nodes_seq[i].size() - 1; j++)
        {
            std::pair<int, bool> edge = map_[nodes_seq[i][j] * nr_scans + nodes_seq[i][j + 1]];

            if (edge.second)
                combin *= inverse(matches[edge.first].transformation);
            else
                combin *= (matches[edge.first].transformation);
            poses[nodes_seq[i][j + 1]] = combin;
        }
    }
}

/**
 * @brief 4x4 刚体变换求逆：R^T 与 -R^T t。
 */

Eigen::Matrix4f WHU::HL_MRF::inverse(Eigen::Matrix4f &mat)
{
    Eigen::Matrix3f R = mat.block(0, 0, 3, 3);
    Eigen::Vector3f t = mat.block(0, 3, 3, 1);

    Eigen::Matrix4f inversed = Eigen::Matrix4f::Identity();
    inversed.block(0, 0, 3, 3) = R.transpose().block(0, 0, 3, 3);
    inversed.block(0, 3, 3, 1) = (-(R.transpose() * t)).block(0, 0, 3, 1);
    return inversed;
}

/**
 * @brief 基于 PCL 的 LUM(Graph-SLAM) 全局细配准：以每个连通子图的初始位姿（相对其根节点）
 *        为初值，迭代估计跨帧对应并优化所有顶点的姿态。
 *
 * @details
 * 处理流程（对每个子图 f）：
 *  1) 若该子图边数 < 2，则直接将 poses[f] 写盘后跳过（单节点或仅一条边，LUM 无实质优化）。
 *  2) 构建 LUM 图：先将根节点云作为参考顶点 addPointCloud(root)，再将子图内其它已赋初值的节点以位姿 pose 加入。
 *  3) 迭代 `LUM_iterations` 次：
 *      a. 用当前 poses[f][i] 更新 LUM 顶点位姿（lum.setPose）。
 *      b. 沿 DFS 路径 nodes_seqs[f] 为每条相邻节点对估计“互惠对应”（reciprocal correspondences）：
 *         - 将 next 节点云按 T = inv(poses[f][curr]) * poses[f][next] 变换到 curr 坐标系；
 *         - 用 `voxel_size_icp` 作为最大对应距离估计互惠对应；
 *         - 以 (idx_next, idx_curr) 的图顶点索引写入 LUM 的边约束（lum.setCorrespondences）。
 *      c. 设置 LUM 的单步求解参数（max iters = 1，收敛阈值 1e-3），调用 `lum.compute()` 优化整图。
 *      d. 用 lum.getTransformation(i) 回写 poses[f][LUM_indices_map_inv_[f][i]]。
 *  4) 优化完成后，将各顶点的最终变换写盘，同时同步回 poses[f]。
 *
 * 关键索引结构：
 *  - LUM_indices_map_[f][scan_id]      : 由“全局扫描 id”映射到“LUM 图内顶点序号（连续）”。
 *  - LUM_indices_map_inv_[f][lum_idx]  : 与上相反，由“LUM 顶点序号”映射回“全局扫描 id”。
 *
 * @param[in,out] poses         逐子图的全局姿态解（作为 LUM 初值与最终回写容器）；poses[f][i] 为 4x4 齐次矩阵。
 *                              未参与本子图或未初始化的元素通常为 Zero 矩阵。
 * @param[in]     edges         逐子图的 MST 边索引（仅用于判断子图是否可做 LUM，内部流程不直接使用）。
 * @param[in]     root          每个子图的根节点全局扫描 id（作为参考帧）。
 * @param[in]     nodes_seqs    逐子图的 DFS 路径集合；nodes_seqs[f][k] 是一条从 root[f] 出发的顶点序列。
 * @param[in]     LUM_indices_map_     f 维度的二维映射：scan_id -> lum_vertex_idx。
 * @param[in]     LUM_indices_map_inv_ f 维度的二维映射：lum_vertex_idx -> scan_id。
 * @param[in]     num_of_subtrees      连通子图数量。
 * @param[in]     clouds       所有扫描的点云数组（与全局 scan id 对齐）。
 *
 * @pre  已通过 GROR/ICP 等生成子图的初始位姿（poses[f]），且 LUM 的索引映射已构建完毕。
 * @post poses[f] 将被 LUM 的优化结果覆盖；会在工程输出目录生成每个子图的最终变换文本。
 *
 * @note
 * - 对应估计使用 reciprocal correspondences，距离阈值为 @c voxel_size_icp（单位：m）。
 * - 并行部分（OpenMP）包含 `#pragma omp critical` 以避免跨线程写 LUM/visited 集时的数据竞争。
 * - 当 `use_LUM == false` 时此函数不做任何操作。
 * - I/O：结果保存到 `output_dir + filename.str() + "_Mst" + <f> + ".txt"`。
 */
void WHU::HL_MRF::LUMoptimization(std::vector<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>>& poses,
    std::vector<std::vector<size_t>>& edges,
    std::vector<int>& root,
    std::vector<std::vector<pcl::Indices>>& nodes_seqs,
    std::vector<std::vector<int>>& LUM_indices_map_,
    std::vector<std::vector<int>>& LUM_indices_map_inv_,
    int num_of_subtrees,
    std::vector<pcl::PointCloud<PointT>::Ptr>& clouds)
{
    if (use_LUM)
    {
        // 复用容器（每个子图的对应关系集合）；目前未持久化，仅示例保留
        std::set<int> visited; // 记录某轮中已作为“next”参与过对应估计的节点，避免重复
        std::vector<std::vector<pcl::CorrespondencesPtr>> correspondences(edges.size());

        //================ LUM 全局细配准（逐子图） ================//
        for (int f = 0; f < num_of_subtrees; f++)
        {
            int q = 0; // 历史遗留/调试计数器（当前未使用）
            std::stringstream ss;
            ss << f;

            // 子图边数过少（<2）则直接落盘 poses[f]，不做 LUM
            if (edges[f].size() < 2)
            {
                std::ofstream ofs3(output_dir + filename.str() + "_Mst" + ss.str() + ".txt");
                std::cout << "...saving results to files" << std::endl;
                for (int i = 0; i < nr_scans; i++)
                {
                    if (poses[f][i].isZero())
                        continue; // 该 scan 不在本子图内或未初始化

                    ofs3 << "affine transformation: \n";
                    ofs3 << poses[f][i] << "\n";
                }
                ofs3.close();
                continue;
            }

            // 构造并初始化 LUM 图优化器
            pcl::registration::LUM<PointT> lum;
            std::cout << "apply lum global matching..\n";

            // 1) 将根节点云作为参考顶点加入（无初值姿态）
            lum.addPointCloud(clouds[root[f]]);

            // 2) 将子图内其它已初始化节点，按“当前全局位姿”作为初值加入 LUM 顶点
            for (int i = 0; i < nr_scans; i++)
            {
                if (i == root[f]) continue;
                if (poses[f][i].isZero()) continue; // 未在子图中或未初始化

                // 将 4x4 齐次矩阵转换为 xyz + rpy（弧度），以满足 LUM 接口
                const Eigen::Transform<float, 3, Eigen::Affine> aff(poses[f][i]);
                Eigen::Vector6f pose; // [x y z roll pitch yaw]
                pcl::getTranslationAndEulerAngles(aff, pose[0], pose[1], pose[2],
                    pose[3], pose[4], pose[5]);
                lum.addPointCloud(clouds[i], pose);
            }

            // 3) LUM 迭代求解（外层控制总迭代次数）
            const int iteration_ = LUM_iterations;
            for (int li = 0; li < iteration_; li++)
            {
                visited.clear(); // 每轮只允许某 next 节点参与一次“被估计”，减少重复对应

                std::cout << "...get correspondences\n";

                // 3.a 用当前 poses 更新 LUM 顶点位姿
                for (int i = 0; i < nr_scans; i++)
                {
                    if (i == root[f]) continue;
                    if (poses[f][i].isZero()) continue;

                    const Eigen::Transform<float, 3, Eigen::Affine> aff(poses[f][i]);
                    Eigen::Vector6f pose;
                    pcl::getTranslationAndEulerAngles(aff, pose[0], pose[1], pose[2],
                        pose[3], pose[4], pose[5]);
                    // 通过映射表将“全局 scan id”定位到 LUM 顶点序号
                    lum.setPose(LUM_indices_map_[f][i], pose);
                }

                // 3.b 沿 DFS 路径为相邻节点对估计“互惠对应”，并写入 LUM 的边
                std::vector<pcl::CorrespondencesPtr> correspondences_t; // 当前轮的临时集合（保留变量以便扩展）

                for (int i = 0; i < static_cast<int>(nodes_seqs[f].size()); i++)
                {
                    // 对路径上的相邻对 (curr -> next) 并行估计
#pragma omp parallel for num_threads(num_threads)
                    for (int j = 0; j < static_cast<int>(nodes_seqs[f][i].size()) - 1; j++)
                    {
                        const int curr = nodes_seqs[f][i][j];
                        const int next = nodes_seqs[f][i][j + 1];

                        // 本轮中若 next 尚未被作为“源”处理，则进行一次对应估计
                        if (std::find(visited.begin(), visited.end(), next) == visited.end())
                        {
                            pcl::registration::CorrespondenceEstimation<PointT, PointT> correspondence_estimate;
                            pcl::CorrespondencesPtr temp(new pcl::Correspondences);
                            pcl::PointCloud<PointT>::Ptr src_in_curr(new pcl::PointCloud<PointT>());

                            // 将 next 的点云用 T = inv(Pose(curr)) * Pose(next) 变换到 curr 坐标系
                            // 这里 inverse(...) 为 4x4 齐次姿态求逆；若使用 Eigen 也可 poses[f][curr].inverse()
                            pcl::transformPointCloud(
                                *clouds[next], *src_in_curr,
                                inverse(poses[f][curr]) * poses[f][next]
                            );

                            // 源 = 变换后的 next（在 curr 坐标系），目标 = curr 原始云
                            correspondence_estimate.setInputSource(src_in_curr);
                            correspondence_estimate.setInputTarget(clouds[curr]);

                            // 互惠对应估计；最大对应距离 = voxel_size_icp
                            correspondence_estimate.determineReciprocalCorrespondences(*temp, voxel_size_icp);

                            // 将对应与“访问标记”写入共享状态，需临界区保护
#pragma omp critical
                            {
                                visited.insert(next);
                                std::cout << i << ":" << "correspondences sizes: " << temp->size() << "\n";
                                // 将 (next -> curr) 的对应放入 LUM：参数为 LUM 顶点索引
                                lum.setCorrespondences(
                                    LUM_indices_map_[f][next],
                                    LUM_indices_map_[f][curr],
                                    temp
                                );
                            }
                        }
                    }
                }

                // 3.c LUM 单步优化（外层循环控制总步数）
                lum.setMaxIterations(1);
                lum.setConvergenceThreshold(0.001f);
                std::cout << "perform lum optimization...\n";
                lum.compute();

                // 3.d 将 LUM 的当前顶点解回写到 poses[f]
                //     通过 inv-map 将 lum 顶点序号映射回全局 scan id
                for (int i = 0; i < lum.getNumVertices(); i++)
                {
                    poses[f][LUM_indices_map_inv_[f][i]] = lum.getTransformation(i).matrix();
                }
            } // for li (LUM_iterations)

            // 4) 最终结果落盘（并再次同步一遍，保证一致）
            std::ofstream ofs3(output_dir + filename.str() + "_Mst" + ss.str() + ".txt");
            std::cout << "...saving results to files" << std::endl;
            for (int i = 0; i < lum.getNumVertices(); i++)
            {
                ofs3 << "affine transformation: \n";
                ofs3 << lum.getTransformation(i).matrix();
                poses[f][LUM_indices_map_inv_[f][i]] = lum.getTransformation(i).matrix();
                if (i == (lum.getNumVertices() - 1)) break;
                ofs3 << "\n";
            }
            ofs3.close();

            // 可视化调试（保留注释）
            // pcl::visualization::PCLVisualizer vis;
            // vis.addPointCloud(cloud_out, "clouds");
            // vis.spin();
        } // for f
    } // if(use_LUM)
}

void WHU::HL_MRF::solveGlobalPose()
{
    for (int i = Hierarchical_block_poses.size() - 1; i > 0; i--)
    {
        int count = 0;
        for (int j = 0; j < Hierarchical_block_poses[i].size(); j++)
        {
            for (int k = 0; k < Hierarchical_block_poses[i][j].size(); k++)
            {
                for (int m = 0; m < Hierarchical_block_poses[i - 1][count].size(); m++)
                {
                    Hierarchical_block_poses[i - 1][count][m] = Hierarchical_block_poses[i][j][k] * Hierarchical_block_poses[i - 1][count][m];
                }
                count++;
            }
        }
    }
    int count = 0;
    for (int i = 0; i < Hierarchical_block_poses[0].size(); i++)
    {
        for (int j = 0; j < Hierarchical_block_poses[0][i].size(); j++)
        {
            global_poses[count] = Hierarchical_block_poses[0][i][j];
            count++;
        }
    }
}

void WHU::HL_MRF::eliminateClosestPoints(pcl::PointCloud<PointT>::Ptr &src,
                                         pcl::PointCloud<PointT>::Ptr &tgt,
                                         Eigen::Matrix4f &trans,
                                         pcl::PointCloud<pcl::FPFHSignature33>::Ptr &fpfs,
                                         pcl::PointCloud<pcl::FPFHSignature33>::Ptr &fpft)
{
    pcl::CorrespondencesPtr corr(new pcl::Correspondences);
    pcl::registration::CorrespondenceEstimation<PointT, PointT> es;
    es.setInputSource(src);
    es.setInputTarget(tgt);
    es.determineReciprocalCorrespondences(*corr, voxel_size_icp);

    int before = src->points.size();
    pcl::PointCloud<PointT>::Ptr after_earse(new pcl::PointCloud<PointT>());
    int count = 0;
    for (int i = 0; i < src->points.size(); i++)
    {
        if (i == (*corr)[count].index_query)
        {
            count++;
            continue;
        }
        after_earse->points.push_back(src->points[i]);
    }
    src->clear();
    src = after_earse;
    int after = src->points.size();
    cout << before - after << "points have been removed\n";
    count = 0;
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr after_earse_feature(new pcl::PointCloud<pcl::FPFHSignature33>());
    if (fpfs.get() != nullptr)
    {
        for (int i = 0; i < fpfs->points.size(); i++)
        {
            if (i == (*corr)[count].index_query)
            {
                count++;
                continue;
            }
            after_earse_feature->points.push_back(fpfs->points[i]);
        }
        fpfs->clear();
        fpfs = after_earse_feature;
    }
}

double WHU::HL_MRF::getRMSE(pcl::PointCloud<PointT>::Ptr src, pcl::PointCloud<PointT>::Ptr tgt, Eigen::Matrix4f &trans, double max)
{
    double fitness_score = 0.0;

    // Transform the input dataset using the final transformation
    pcl::PointCloud<PointT> transformed;
    transformPointCloud(*src, transformed, trans);

    std::vector<int> nn_indices(1);
    std::vector<float> nn_dists(1);
    pcl::KdTreeFLANN<PointT>::Ptr tree_(new pcl::KdTreeFLANN<PointT>());
    tree_->setInputCloud(tgt);

    // For each point in the source dataset
    int nr = 0;
    for (std::size_t i = 0; i < transformed.points.size(); ++i)
    {
        // Find its nearest neighbor in the target
        tree_->nearestKSearch(transformed.points[i], 1, nn_indices, nn_dists);

        // Deal with occlusions (incomplete targets)
        if (nn_dists[0] <= max)
        {
            // Add to the fitness score
            fitness_score += nn_dists[0];
            nr++;
        }
    }

    if (nr > 0)
        return (fitness_score / nr);
    return (std::numeric_limits<double>::max());
}

void WHU::HL_MRF::performMultiviewRegistration()
{
    auto begin = std::chrono::steady_clock::now();

    // ====== Step 1. 读取输入点云文件 ======
    // 若未设置输入路径，直接退出
    if (this->PLYpath.empty())
    {
        std::cerr << "no input paths";
        return;
    }
    // 从输入路径读取所有 .ply/.pcd 文件路径，排序后存入 files[]
    if (!readPLYfiles())
    {
        std::cerr << "Input path has no files: " << PLYpath << std::endl;
        return;
    }
    scans_left = files.size();         // 剩余的扫描数量
    global_poses.resize(files.size()); // 为最终全局位姿分配空间
    scans_left_last = 0;               // 上一轮剩余扫描数（用于判断收敛）

    /* ====== Step 2. 层次化多视角配准 ======
     * - Loop 1: 块内配准（internal block registration）
     * - Loop 2: 块与块之间配准（block-to-block registration）
     * - 终止条件：
     *   1) scans_left == 1 ：所有扫描已合并；
     *   2) scans_left 未减少：有些扫描无法继续合并。
     */
    while (scans_left != 1 && scans_left_last != scans_left)
    {
        scans_left_last = scans_left;
        init(); // 初始化分块数量 part_loop / 余数 part_num_rest 等

        // 保存当前大循环（bigloop）下，每个子块的位姿结果
        std::vector<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>> block;

        for (int loop = 0; loop < part_loop; loop++)
        {
            // ====== Step 2.1 构建扫描对 ======
            blockPartition(loop);    // 生成块内 pairs (i,j)
            nr_pairs = pairs.size(); // 当前块的边数

            // ====== 临时变量声明 ======
            // (1) 块内点云与特征
            std::vector<pcl::PointCloud<PointT>::Ptr> keypoint_clouds;
            std::vector<pcl::PointCloud<pcl::FPFHSignature33>::Ptr> keypoint_clouds_feature;
            std::vector<pcl::PointCloud<PointT>::Ptr> clouds;
            // (2) 粗配准（GROR）
            std::vector<int> pairs_best_count(nr_pairs);
            std::vector<pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>> candidate_matches(nr_pairs);
            std::set<int> rejected_pairs_; // 被判为错误匹配的边
            // (3) 细配准（ICP+LUM）
            int num_of_subtrees;
            std::vector<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>> poses;

            // ====== Step 2.2 准备点云 ======
            transferVaribles(keypoint_clouds, keypoint_clouds_feature, clouds, loop);
            if (bigloop == 0) // 第一层循环才做预处理：下采样、ISS关键点、FPFH
            {
                preprocessing(keypoint_clouds, keypoint_clouds_feature, clouds, loop);
            }

            // ====== Step 2.3 粗配准（pairwise GROR） ======
            coarseRgistration(keypoint_clouds, keypoint_clouds_feature, clouds, pairs_best_count, candidate_matches);

            // ====== Step 2.4 全局一致性选择（ScanGraphInference） ======
            globalCoarseRegistration(rejected_pairs_, candidate_matches, pairs_best_count, nr_scans);

            // ====== Step 2.5 全局细配准（MST+ICP+LUM） ======
            globalFineRegistration(rejected_pairs_, candidate_matches, clouds, poses, num_of_subtrees);

            // ====== Step 2.6 根据扫描数量分情况合并 ======
            if (nr_scans >= 3)
            {
                // --- 情况 A：块内扫描数 >=3，使用 MST+ICP+LUM ---
                // 保存每个子树的位姿结果
                for (int f = 0; f < num_of_subtrees; f++)
                {
                    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> temp0;
                    for (int i = 0; i < poses[f].size(); i++)
                    {
                        if (poses[f][i].isZero())
                            continue;
                        temp0.push_back(poses[f][i]);
                    }
                    block.push_back(temp0);
                }

                // 将同一子树内的扫描点云变换到统一坐标系并合并
                for (int f = 0; f < num_of_subtrees; f++)
                {
                    pcl::PointCloud<PointT>::Ptr part_sum(new pcl::PointCloud<PointT>);
                    pcl::PointCloud<PointT>::Ptr part_sum_keypoint(new pcl::PointCloud<PointT>);
                    pcl::PointCloud<pcl::FPFHSignature33>::Ptr part_sum_keypoint_feature(new pcl::PointCloud<pcl::FPFHSignature33>);

                    for (int i = 0; i < poses[f].size(); i++)
                    {
                        if (poses[f][i].isZero())
                            continue;
                        // 将点云/关键点/特征变换到位姿 poses[f][i]
                        pcl::transformPointCloud(*clouds[i], *clouds[i], poses[f][i]);
                        pcl::transformPointCloud(*keypoint_clouds[i], *keypoint_clouds[i], poses[f][i]);

                        // 累积合并
                        *part_sum += *clouds[i];
                        *part_sum_keypoint += *keypoint_clouds[i];
                        *part_sum_keypoint_feature += *keypoint_clouds_feature[i];

                        // 清空原始，释放内存
                        clouds[i]->clear();
                        keypoint_clouds[i]->clear();
                        keypoint_clouds_feature[i]->clear();
                    }

                    // 将结果保存到 part_clouds[] 等容器
                    if (bigloop == 0)
                    {
                        part_clouds.push_back(part_sum);
                        part_keypoint_clouds.push_back(part_sum_keypoint);
                        part_keypoint_feature_clouds.push_back(part_sum_keypoint_feature);
                    }
                    else
                    {
                        part_clouds[all_count] = part_sum;
                        part_keypoint_clouds[all_count] = part_sum_keypoint;
                        part_keypoint_feature_clouds[all_count] = part_sum_keypoint_feature;
                    }

                    // 可选：可视化检查
                    if (check_blcok)
                    {
                        pcl::visualization::PCLVisualizer vis;
                        vis.addPointCloud<PointT>(part_clouds[all_count]);
                        vis.spin();
                    }
                    all_count++;
                }
            }
            else
            {
                // ====== 情况 B：块内扫描数 < 3 ======

                // --- Case B1: 块内只有 2 个扫描 ---
                if (nr_scans == 2)
                {
                    // 如果 GROR 粗配准找到的内点数超过阈值，说明匹配可靠
                    if (pairs_best_count[0] > max_consensus_set)
                    {
                        cout << "apply  ICP" << endl;

                        // 将两个点云拷贝出来，用于ICP精配准
                        pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>);
                        pcl::PointCloud<pcl::PointXYZ>::Ptr temp2(new pcl::PointCloud<pcl::PointXYZ>);
                        copyPointCloud(*clouds[0], *temp);
                        copyPointCloud(*clouds[1], *temp2);

                        // 使用 PCL ICP 进一步优化 candidate_matches[0].transformation
                        pcl::IterativeClosestPoint<PointT, PointT> icp;
                        icp.setInputSource(temp);
                        icp.setInputTarget(temp2);
                        icp.setMaxCorrespondenceDistance(voxel_size_icp); // ICP 搜索半径
                        icp.setUseReciprocalCorrespondences(true);
                        icp.setMaximumIterations(10);
                        icp.setEuclideanFitnessEpsilon(10e-4);
                        icp.align(*temp, candidate_matches[0].transformation);
                        candidate_matches[0].transformation = icp.getFinalTransformation();

                        // ====== 记录配准后的位姿 ======
                        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> temp0;
                        temp0.push_back(Eigen::Matrix4f::Identity());                  // 第一个点云作为参考
                        temp0.push_back(inverse(candidate_matches[0].transformation)); // 第二个点云相对第一个的变换
                        block.push_back(temp0);

                        // ====== 合并两个点云 ======
                        pcl::PointCloud<PointT>::Ptr part_sum(new pcl::PointCloud<PointT>);
                        pcl::PointCloud<PointT>::Ptr part_sum_keypoint(new pcl::PointCloud<PointT>);
                        pcl::PointCloud<pcl::FPFHSignature33>::Ptr part_sum_keypoint_feature(new pcl::PointCloud<pcl::FPFHSignature33>);

                        // 将第2个点云按反变换对齐到第1个点云坐标系
                        pcl::transformPointCloud(*clouds[1], *clouds[1], inverse(candidate_matches[0].transformation));
                        pcl::transformPointCloud(*keypoint_clouds[1], *keypoint_clouds[1], inverse(candidate_matches[0].transformation));

                        // 合并两份点云及其特征
                        *part_sum += *clouds[0] + *clouds[1];
                        *part_sum_keypoint += *keypoint_clouds[0] + *keypoint_clouds[1];
                        *part_sum_keypoint_feature += *keypoint_clouds_feature[0] + *keypoint_clouds_feature[1];

                        // 保存结果
                        if (bigloop == 0)
                        {
                            part_clouds.push_back(part_sum);
                            part_keypoint_clouds.push_back(part_sum_keypoint);
                            part_keypoint_feature_clouds.push_back(part_sum_keypoint_feature);
                        }
                        else
                        {
                            part_clouds[all_count] = std::move(part_sum);
                            part_keypoint_clouds[all_count] = std::move(part_sum_keypoint);
                            part_keypoint_feature_clouds[all_count] = std::move(part_sum_keypoint_feature);
                        }

                        // 输出配准结果到文本
                        std::ofstream ofs3(output_dir + filename.str() + "_Mst" + ".txt");
                        cout << "...saving results to files" << endl;
                        ofs3 << "affine transformation: \n";
                        ofs3 << Eigen::Matrix4f::Identity() << "\n";
                        ofs3 << "affine transformation: \n";
                        ofs3 << inverse(candidate_matches[0].transformation);
                        ofs3.close();

                        all_count++;
                    }
                    else
                    {
                        // 若 GROR 内点数过小，不足以支撑ICP，认为粗配准失败
                        // 此时直接保存原始两个点云，不做变换
                        if (bigloop == 0)
                        {
                            part_clouds.push_back(clouds[0]);
                            part_keypoint_clouds.push_back(keypoint_clouds[0]);
                            part_keypoint_feature_clouds.push_back(keypoint_clouds_feature[0]);
                            part_clouds.push_back(clouds[1]);
                            part_keypoint_clouds.push_back(keypoint_clouds[1]);
                            part_keypoint_feature_clouds.push_back(keypoint_clouds_feature[1]);
                        }
                        else
                        {
                            part_clouds[all_count] = std::move(clouds[0]);
                            part_keypoint_clouds[all_count] = std::move(keypoint_clouds[0]);
                            part_keypoint_feature_clouds[all_count] = std::move(keypoint_clouds_feature[0]);
                            part_clouds[all_count + 1] = std::move(clouds[1]);
                            part_keypoint_clouds[all_count + 1] = std::move(keypoint_clouds[1]);
                            part_keypoint_feature_clouds[all_count + 1] = std::move(keypoint_clouds_feature[1]);
                        }

                        // 记录两个点云都使用单位矩阵
                        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> temp0;
                        temp0.push_back(Eigen::Matrix4f::Identity());
                        temp0.push_back(Eigen::Matrix4f::Identity());
                        block.push_back(temp0);

                        // 输出结果到文本
                        std::ofstream ofs3(output_dir + filename.str() + "_Mst0" + ".txt");
                        ofs3 << "affine transformation: \n";
                        ofs3 << Eigen::Matrix4f::Identity();
                        ofs3.close();

                        std::ofstream ofs4(output_dir + filename.str() + "_Mst1" + ".txt");
                        ofs4 << "affine transformation: \n";
                        ofs4 << Eigen::Matrix4f::Identity();
                        ofs4.close();

                        all_count++;
                        all_count++;
                    }
                }
                else
                {
                    // --- Case B2: 块内只有 1 个扫描 ---
                    // 不需要配准，直接保存
                    if (bigloop == 0)
                    {
                        part_clouds.push_back(clouds[0]);
                        part_keypoint_clouds.push_back(keypoint_clouds[0]);
                        part_keypoint_feature_clouds.push_back(keypoint_clouds_feature[0]);
                    }
                    else
                    {
                        part_clouds[all_count] = std::move(clouds[0]);
                        part_keypoint_clouds[all_count] = std::move(keypoint_clouds[0]);
                        part_keypoint_feature_clouds[all_count] = std::move(keypoint_clouds_feature[0]);
                    }

                    // 位姿恒为单位矩阵
                    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> temp0;
                    temp0.push_back(Eigen::Matrix4f::Identity());
                    block.push_back(temp0);

                    // 保存 Identity 变换
                    std::ofstream ofs3(output_dir + filename.str() + "_Mst0" + ".txt");
                    ofs3 << "affine transformation: \n";
                    ofs3 << Eigen::Matrix4f::Identity();
                    ofs3.close();

                    all_count++;
                }
            }

        } // end for (loop)

        // 将当前层的结果保存
        Hierarchical_block_poses.push_back(block);

        // ====== Step 2.7 统计剩余扫描数，准备下一轮 ======
        scans_left = 0;
        for (int i = 0; i < part_clouds.size(); i++)
        {
            if (part_clouds[i].get() == nullptr)
                continue;
            scans_left++;
        }
        part_clouds.resize(scans_left);
        part_keypoint_clouds.resize(scans_left);
        part_keypoint_feature_clouds.resize(scans_left);
        bigloop++; // 进入下一层循环
    } // end while

    // ====== Step 3. 导出结果 ======
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> past = std::chrono::duration_cast<std::chrono::duration<double>>(end - begin);
    int min = past.count() / 60;
    double sec = past.count() - 60 * min;
    cout << "total cost time:" << min << "min" << sec << "sec\n";

    // 保存最终合并后的点云
    pcl::io::savePCDFile(output_dir + "/aligned_point_cloud.pcd", *part_clouds[0], true);

    // 求解并保存最终全局位姿
    try
    {
        solveGlobalPose();
    }
    catch (std::exception &e)
    {
        e.what();
    }

    std::ofstream ofs3(output_dir + "/global_poses.txt");
    cout << "...saving final results to files" << endl;
    for (int i = 0; i < global_poses.size(); i++)
    {
        ofs3 << i + 1 << ": \n";
        ofs3 << global_poses[i] << "\n";
    }
    ofs3.close();
}
