#include "./include/gror_pre.h"

void GrorPre::voxelGridFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudVG, double inlTh)
{
    // format for filtering
    pcl::PCLPointCloud2::Ptr cloud2(new pcl::PCLPointCloud2());
    pcl::PCLPointCloud2::Ptr cloudVG2(new pcl::PCLPointCloud2());
    pcl::toPCLPointCloud2(*cloud, *cloud2);
    // set up filtering parameters
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    sor.setInputCloud(cloud2);
    sor.setLeafSize(inlTh, inlTh, inlTh);
    // filtering process
    sor.filter(*cloudVG2);
    pcl::fromPCLPointCloud2(*cloudVG2, *cloudVG);
}

/**
 * @brief 提取点云的 ISS (Intrinsic Shape Signatures) 关键点
 *
 * 该函数基于 PCL 提供的 ISSKeypoint3D 算子实现关键点检测，
 * 通过计算局部协方差矩阵的特征值来判断点的显著性，并使用非极大值抑制保证关键点分布的稀疏性。
 *
 * @param[in] cloud       输入点云（原始点云）
 * @param[out] ISS        提取的关键点点云（坐标结果）
 * @param[out] ISS_Idx    关键点对应的原始点索引集合
 * @param[in] resolution  点云分辨率（通常为体素下采样后的体素大小），用于控制邻域半径与阈值设置
 *
 * @note
 * 1. 参数设置依赖于点云的分辨率，需根据数据集场景调整。
 * 2. 非极大值抑制半径在不同场景可调节（如办公室/铁路环境）。
 */
void GrorPre::issKeyPointExtration(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                   pcl::PointCloud<pcl::PointXYZ>::Ptr ISS,
                                   pcl::PointIndicesPtr ISS_Idx,
                                   double resolution)
{
    // -------------------- 参数设置 --------------------
    // ISS关键点检测的核心参数都依赖于点云分辨率 resolution

    // 邻域搜索半径，用于计算协方差矩阵特征值（显著性判断）
    double iss_salient_radius_ = 6 * resolution;

    // 非极大值抑制半径，用于避免提取过于密集的关键点（选择局部最显著的点）
    double iss_non_max_radius_ = 4 * resolution;
    // 代码中还给出了不同场景的经验值：
    // double iss_non_max_radius_ = 2 * resolution; // for office
    // double iss_non_max_radius_ = 9 * resolution; // for railway

    // 特征值比阈值，用于判断点是否是关键点
    double iss_gamma_21_(0.975); // λ2 / λ1 < gamma21
    double iss_gamma_32_(0.975); // λ3 / λ2 < gamma32

    // 至少需要的邻居数，保证计算协方差矩阵和特征值的稳定性
    double iss_min_neighbors_(4);

    // 线程数（PCL的ISS算子本身支持多线程加速）
    int iss_threads_(1);

    // -------------------- 构建KD树搜索结构 --------------------
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

    // -------------------- 初始化ISS检测器 --------------------
    pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_detector;

    iss_detector.setSearchMethod(tree);                 // 设置搜索方法为KD树
    iss_detector.setSalientRadius(iss_salient_radius_); // 设置显著性计算半径
    iss_detector.setNonMaxRadius(iss_non_max_radius_);  // 设置非极大值抑制半径
    iss_detector.setThreshold21(iss_gamma_21_);         // 设置特征值比阈值 λ2/λ1
    iss_detector.setThreshold32(iss_gamma_32_);         // 设置特征值比阈值 λ3/λ2
    iss_detector.setMinNeighbors(iss_min_neighbors_);   // 设置最小邻居数量
    iss_detector.setNumberOfThreads(iss_threads_);      // 设置线程数
    iss_detector.setInputCloud(cloud);                  // 输入点云

    // -------------------- 执行ISS关键点检测 --------------------
    iss_detector.compute(*ISS); // 结果存储在 ISS 点云中

    // -------------------- 保存关键点索引 --------------------
    // 除了关键点坐标点云，还可以获取关键点对应的原始点索引
    ISS_Idx->indices = iss_detector.getKeypointsIndices()->indices;
    ISS_Idx->header = iss_detector.getKeypointsIndices()->header;
}

/**
 * @brief 计算输入点云关键点的 FPFH (Fast Point Feature Histograms) 特征
 *
 * 该函数首先估计点云的法向量，然后基于法向量在关键点位置处计算 FPFH 特征。
 * FPFH 是一种局部几何特征描述子，常用于点云配准、关键点匹配等任务。
 *
 * @param[in]  cloud       输入点云（通常为经过体素下采样的点云）
 * @param[in]  resolution  点云分辨率（通常与体素大小一致，用于确定邻域搜索半径）
 * @param[in]  iss_Idx     ISS 关键点在输入点云中的索引
 * @param[out] fpfh_out    输出的 FPFH 特征点云，每个关键点对应一个 33 维直方图
 *
 * ### 算法流程
 * 1. 使用半径 `3 × resolution` 估计点云法向量
 * 2. 使用半径 `8 × resolution` 在关键点邻域内计算 FPFH
 * 3. 支持 OMP 多线程加速，默认线程数设置为 16
 *
 * ### 输出结果
 * - `fpfh_out` 包含所有关键点的 FPFH 特征
 *
 * @note
 * - FPFH 对点云的几何结构具有较强的判别能力，适用于点云匹配。
 * - 半径选择需与点云密度相适应，否则特征可能过稀或过密。
 */
void GrorPre::fpfhComputation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                              double resolution,
                              pcl::PointIndicesPtr iss_Idx,
                              pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_out)
{
    //===================== 1. 法向量估计 =====================//
    pcl::PointCloud<pcl::Normal>::Ptr normal(new pcl::PointCloud<pcl::Normal>());           ///< 存放估计出的法向量
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>()); ///< KdTree 空间搜索结构

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne; ///< 法向量估计器
    ne.setInputCloud(cloud);                              ///< 设置输入点云
    ne.setSearchMethod(tree);                             ///< 设置邻域搜索方法为 KdTree
    ne.setRadiusSearch(3 * resolution);                   ///< 邻域半径 = 3 × 分辨率
    ne.compute(*normal);                                  ///< 计算法向量并存储到 normal

    //===================== 2. FPFH 特征计算 =====================//
    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_est; ///< 基于 OMP 的 FPFH 特征估计器
    fpfh_est.setInputCloud(cloud);                                                     ///< 设置输入点云
    fpfh_est.setInputNormals(normal);                                                  ///< 设置法向量
    fpfh_est.setSearchMethod(tree);                                                    ///< 使用 KdTree 搜索邻域
    fpfh_est.setRadiusSearch(8 * resolution);                                          ///< 邻域半径 = 8 × 分辨率（比法向量估计更大，获取更多邻域几何信息）
    fpfh_est.setNumberOfThreads(16);                                                   ///< 使用 16 线程并行加速计算
    fpfh_est.setIndices(iss_Idx);                                                      ///< 仅在 ISS 关键点位置计算特征
    fpfh_est.compute(*fpfh_out);                                                       ///< 计算并输出关键点对应的 FPFH 特征
}

/**
 * @brief 在源点云和目标点云的 FPFH 特征之间搜索对应关系（双向验证）。
 *
 * 该函数利用最近邻搜索（KdTree）在源点云特征和目标点云特征之间建立匹配关系。
 * 它采用了**互验证（reciprocal correspondences）**的策略：
 * - 先从源点云特征出发，找到目标点云的若干候选匹配点；
 * - 再从目标点云出发，检查这些候选点是否也将源点作为其近邻之一；
 * - 只有通过双向检查的匹配对才被保留。
 *
 * @param[in]  fpfhs    源点云的 FPFH 特征集合
 * @param[in]  fpfht    目标点云的 FPFH 特征集合
 * @param[out] corr     输出的匹配对应关系（源索引、目标索引、距离）
 * @param[in]  max_corr 每个源点在目标点中允许的最大候选匹配数量
 * @param[out] corr_NOs 每个源点被匹配的次数统计
 * @param[out] corr_NOt 每个目标点被匹配的次数统计
 */
void GrorPre::correspondenceSearching(pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs,
                                      pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfht,
                                      pcl::Correspondences &corr,
                                      int max_corr,
                                      std::vector<int> &corr_NOs,
                                      std::vector<int> &corr_NOt)
{
    // 每个源点最多寻找的对应点个数（避免过多候选造成冗余）
    int n = std::min(max_corr, (int)fpfht->size());

    // 清空输出对应关系
    corr.clear();

    // 初始化统计数组：corr_NOs[i] 表示源点 i 被匹配到的次数
    corr_NOs.assign(fpfhs->size(), 0);

    // corr_NOt[j] 表示目标点 j 被匹配到的次数
    corr_NOt.assign(fpfht->size(), 0);

    // === 构建KdTree用于快速最近邻搜索 ===
    pcl::KdTreeFLANN<pcl::FPFHSignature33> treeS;
    treeS.setInputCloud(fpfhs); // 源特征树

    pcl::KdTreeFLANN<pcl::FPFHSignature33> treeT;
    treeT.setInputCloud(fpfht); // 目标特征树

    // === 遍历每一个源点特征，寻找候选匹配 ===
    for (size_t i = 0; i < fpfhs->size(); i++)
    {
        // 存储 i 在目标点云中的候选近邻索引和对应距离
        std::vector<int> corrIdxTmp(n);
        std::vector<float> corrDisTmp(n);

        // 从目标特征集合中，找到源点 i 的 n 个最近邻
        treeT.nearestKSearch(*fpfhs, i, n, corrIdxTmp, corrDisTmp);

        // 遍历这些候选近邻，做互验证（reciprocal check）
        for (size_t j = 0; j < corrIdxTmp.size(); j++)
        {
            bool removeFlag = true;        // 标记是否丢弃该匹配
            int searchIdx = corrIdxTmp[j]; // 目标点索引

            // 再从目标点 searchIdx 出发，寻找其在源点中的 n 个近邻
            std::vector<int> corrIdxTmpT(n);
            std::vector<float> corrDisTmpT(n);
            treeS.nearestKSearch(*fpfht, searchIdx, n, corrIdxTmpT, corrDisTmpT);

            // 检查源点 i 是否在这些近邻中出现
            for (size_t k = 0; k < n; k++)
            {
                if (corrIdxTmpT.data()[k] == i)
                {
                    removeFlag = false; // 说明 i <-> searchIdx 是双向的
                    break;
                }
            }

            // 只有通过双向验证的匹配才被保留
            if (removeFlag == false)
            {
                pcl::Correspondence corrTabTmp;
                corrTabTmp.index_query = i;             // 源点索引
                corrTabTmp.index_match = corrIdxTmp[j]; // 目标点索引
                corrTabTmp.distance = corrDisTmp[j];    // 特征距离
                corr.push_back(corrTabTmp);

                // 更新匹配次数统计
                corr_NOs[i]++;
                corr_NOt[corrIdxTmp[j]]++;
            }
        }
    }
}

void GrorPre::grorPreparation(pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloudS, pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloudT, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudS, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudT, pcl::PointCloud<pcl::PointXYZ>::Ptr issS, pcl::PointCloud<pcl::PointXYZ>::Ptr issT, pcl::CorrespondencesPtr corr, double resolution)
{
    int max_corr = 5; // neighbor number in descriptor searching
    auto t = std::chrono::system_clock::now();
    /*=============down sample point cloud by voxel grid filter=================*/
    std::cout << "/*voxel grid sampling......" << resolution << std::endl;
    GrorPre::voxelGridFilter(origin_cloudS, cloudS, resolution);
    GrorPre::voxelGridFilter(origin_cloudT, cloudT, resolution);

    auto t1 = std::chrono::system_clock::now();
    std::cout << "/*Down!: time consumption of cloud down sample : " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t).count()) / 1000.0 << std::endl;
    std::cout << "/*=================================================*/" << std::endl;

    /*=========================extract iss key points===========================*/
    std::cout << "/*extracting ISS keypoints......" << std::endl;
    pcl::PointIndicesPtr iss_IdxS(new pcl::PointIndices);
    pcl::PointIndicesPtr iss_IdxT(new pcl::PointIndices);
    GrorPre::issKeyPointExtration(cloudS, issS, iss_IdxS, resolution);
    GrorPre::issKeyPointExtration(cloudT, issT, iss_IdxT, resolution);
    auto t2 = std::chrono::system_clock::now();
    std::cout << "/*Down!: time consumption of iss key point extraction: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()) / 1000.0 << std::endl;
    std::cout << "/*=================================================*/" << std::endl;

    /*======================fpfh descriptor computation=========================*/
    std::cout << "/*fpfh descriptor computation......" << std::endl;
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhS(new pcl::PointCloud<pcl::FPFHSignature33>());
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhT(new pcl::PointCloud<pcl::FPFHSignature33>());
    GrorPre::fpfhComputation(cloudS, resolution, iss_IdxS, fpfhS);
    GrorPre::fpfhComputation(cloudT, resolution, iss_IdxT, fpfhT);
    auto t3 = std::chrono::system_clock::now();
    std::cout << "/*Down!: time consumption of fpfh descriptor computation: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()) / 1000.0 << std::endl;
    std::cout << "/*size of issS = " << issS->size() << "; size of issT = " << issT->size() << std::endl;
    std::cout << "/*=================================================*/" << std::endl;

    /*========================correspondences matching=========================*/
    std::cout << "/*matching correspondences..." << std::endl;
    std::vector<int> corr_NOS, corr_NOT;
    GrorPre::correspondenceSearching(fpfhS, fpfhT, *corr, max_corr, corr_NOS, corr_NOT);
    auto t4 = std::chrono::system_clock::now();
    std::cout << "/*Down!: time consumption of matching correspondences: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()) / 1000.0 << std::endl;
    std::cout << "/*number of correspondences= " << corr->size() << std::endl;
    std::cout << "/*=================================================*/" << std::endl;
}

void GrorPre::grorPreparation(pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloudS, pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloudT, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudS, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudT, pcl::PointCloud<pcl::PointXYZ>::Ptr issS, pcl::PointCloud<pcl::PointXYZ>::Ptr issT, Eigen::Vector3f &centerS, Eigen::Vector3f &centerT, pcl::CorrespondencesPtr corr, double resolution)
{
    int max_corr = 5; // neighbor number in descriptor searching
    auto t = std::chrono::system_clock::now();
    /*=============down sample point cloud by voxel grid filter=================*/
    std::cout << "/*voxel grid sampling......" << resolution << std::endl;
    GrorPre::voxelGridFilter(origin_cloudS, cloudS, resolution);
    GrorPre::voxelGridFilter(origin_cloudT, cloudT, resolution);

    auto t1 = std::chrono::system_clock::now();
    std::cout << "/*Down!: time consumption of cloud down sample : " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t).count()) / 1000.0 << std::endl;
    std::cout << "/*=================================================*/" << std::endl;

    /*=========================extract iss key points===========================*/
    std::cout << "/*extracting ISS keypoints......" << std::endl;
    pcl::PointIndicesPtr iss_IdxS(new pcl::PointIndices);
    pcl::PointIndicesPtr iss_IdxT(new pcl::PointIndices);
    GrorPre::issKeyPointExtration(cloudS, issS, iss_IdxS, resolution);
    GrorPre::issKeyPointExtration(cloudT, issT, iss_IdxT, resolution);
    auto t2 = std::chrono::system_clock::now();
    std::cout << "/*Down!: time consumption of iss key point extraction: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()) / 1000.0 << std::endl;
    std::cout << "/*=================================================*/" << std::endl;
    // translating the center of both point clouds to the origin
    pcl::PointXYZ ps, pt;
    pcl::computeCentroid(*issS, ps);

    for (int i = 0; i < issS->size(); i++)
    {
        issS->points[i].x -= ps.x;
        issS->points[i].y -= ps.y;
        issS->points[i].z -= ps.z;
    }
    pcl::computeCentroid(*issT, pt);
    for (int i = 0; i < issT->size(); i++)
    {
        issT->points[i].x -= pt.x;
        issT->points[i].y -= pt.y;
        issT->points[i].z -= pt.z;
    }
    centerS = Eigen::Vector3f(ps.x, ps.x, ps.x);
    centerT = Eigen::Vector3f(pt.x, pt.x, pt.x);
    /*======================fpfh descriptor computation=========================*/
    std::cout << "/*fpfh descriptor computation......" << std::endl;
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhS(new pcl::PointCloud<pcl::FPFHSignature33>());
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhT(new pcl::PointCloud<pcl::FPFHSignature33>());
    GrorPre::fpfhComputation(cloudS, resolution, iss_IdxS, fpfhS);
    GrorPre::fpfhComputation(cloudT, resolution, iss_IdxT, fpfhT);
    auto t3 = std::chrono::system_clock::now();
    std::cout << "/*Down!: time consumption of fpfh descriptor computation: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()) / 1000.0 << std::endl;
    std::cout << "/*size of issS = " << issS->size() << "; size of issT = " << issT->size() << std::endl;
    std::cout << "/*=================================================*/" << std::endl;

    /*========================correspondences matching=========================*/
    std::cout << "/*matching correspondences..." << std::endl;
    std::vector<int> corr_NOS, corr_NOT;
    GrorPre::correspondenceSearching(fpfhS, fpfhT, *corr, max_corr, corr_NOS, corr_NOT);
    auto t4 = std::chrono::system_clock::now();
    std::cout << "/*Down!: time consumption of matching correspondences: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()) / 1000.0 << std::endl;
    std::cout << "/*number of correspondences= " << corr->size() << std::endl;
    std::cout << "/*=================================================*/" << std::endl;
}

void GrorPre::centroidTransMatCompute(Eigen::Matrix4f &T, const Eigen::Vector3f &vS, const Eigen::Vector3f &vT)
{
    Eigen::Vector3f t = T.block(0, 3, 3, 1);
    Eigen::Matrix3f R = T.block(0, 0, 3, 3);

    Eigen::Transform<float, 3, Eigen::Affine> a3f_truth(R);
    Eigen::Vector3f centerSt(0, 0, 0);
    pcl::transformPoint(vS, centerSt, a3f_truth);

    t = t - vT + centerSt;

    T.block(0, 3, 3, 1) = t;
}
