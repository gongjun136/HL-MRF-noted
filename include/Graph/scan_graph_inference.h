/*
 * Copyright(c) 2020, SCHOOL OF GEODESY AND GEOMATIC, WUHAN UNIVERSITY
 * WUHAN, CHINA
 * All Rights Reserved
 * Authors: Hao Wu, Pengcheng Wei, et al.
 * Do not hesitate to contact the authors if you have any question or find any bugs
 * Email: haowu2021@whu.edu.cn
 * Thanks to the work P.W.Theiler, et al.
 *
*/

#include <vector>
#include <iostream>
#include <set>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pcl/registration/matching_candidate.h>
#include "transformation_graph.h"

class ScanGraphInference
{

public:
    ScanGraphInference();
    ~ScanGraphInference();

    /** \brief inference invaild edges.
     * \param[in] length the length of loops
     * \returns  UES
     */
    std::set<int> inference(int length, int t_mcs);

    /**
     * @brief 设置点云扫描对（图中的边集合）
     *
     * 本函数用于将点云配准问题建模为一个图 G_s(V, E)，其中 V 表示扫描（节点），E 表示扫描对（边）。
     * 输入参数 pairs 代表所有待考虑的扫描对 (v_i, v_j)，每个 pair 对应一条边。
     *
     * @param[in] pairs 扫描对集合，每个元素是 (i, j)，表示第 i 个点云和第 j 个点云之间存在配准关系
     * @note 该函数仅保存扫描对信息，后续会结合 transformations 使用
     */
    void setScanPairs(std::vector<std::pair<int, int>> pairs) { edges = pairs; }

    /**
     * @brief 设置扫描对之间的初始配准候选变换
     *
     * 本函数接收与 edges 对应的变换估计结果，保存到 transformations 中。
     * 每个 MatchingCandidate 包含两部分信息：
     *  - 配准的边 (scan pair)
     *  - 对应的变换矩阵 (transformation)
     *
     * @param[in] trans 存储每条边的配准候选信息（MatchingCandidate）
     * @warning 输入的 transformations 数量必须与 edges 数量相同，否则 inference() 会报错
     */
    void setMatchingCandidates(std::vector<pcl::registration::MatchingCandidate,
        Eigen::aligned_allocator<pcl::registration::MatchingCandidate>> &trans)
    {
        transformations = trans;
    }


    /**
     * @brief 设置 MCS（Maximum Clique Size，最大团大小）结果
     *
     * 本函数保存每条边在图结构推理中的 MCS 值，用于后续结合不确定边 (UES) 做边的裁剪。
     *
     * @param[in] vmcs 整型向量，VMCS[i] 表示第 i 条边对应的最大团大小
     * @note 在 inference() 的最后阶段，会结合 t_mcs 阈值过滤掉 MCS 太小的边
     */
    void setMCS(std::vector<int> vmcs)
    {
        VMCS = vmcs;
    }


    /**
     * @brief 设置预定义的旋转精度阈值
     *
     * 在进行闭环验证 (loop closure) 时，会计算旋转误差。
     * 如果旋转误差超过设定阈值（结合 sqrt(n) 和放大因子），则认为闭环无效。
     *
     * @param[in] accuracy 旋转精度阈值（弧度制，默认值为 0.087 ≈ 5°）
     * @note 该参数影响 loopClosure() 中的有效性判定
     */
    inline void setRotationAccuracy(float accuracy)
    {
        rotation_accuracy = accuracy;
    }


    /**
     * @brief 设置预定义的平移精度阈值
     *
     * 在进行闭环验证 (loop closure) 时，会计算平移误差。
     * 如果平移误差超过设定阈值（结合 sqrt(n) 和放大因子），则认为闭环无效。
     *
     * @param[in] accuracy 平移精度阈值（默认值为 0.5 m）
     * @note 该参数影响 loopClosure() 中的有效性判定
     */
    inline void setTranslationAccuracy(float accuracy)
    {
        translation_accuracy = accuracy;
    }


private:
    void eliminateUncertaintyEdges();

    bool loopClosure(Eigen::Matrix4f &trans, int n);

    void nextNode(Eigen::Matrix4f &cur_trans, std::vector<std::pair<int, bool>> &loop, int position);

    void combineTransformation(Eigen::Matrix4f &a, Eigen::Matrix4f b, Eigen::Matrix4f &c, bool inverse);

private:
    /** \brief Scan graph. */
    pcl::registration::TransformationGraph<float> graph;

    // 点云扫描对
    std::vector<std::pair<int, int>> edges;

    /** \brief rotation accuracy. */
    float rotation_accuracy;

    /** \brief translation accuracy. */
    float translation_accuracy;

    // 扫描对之间的初始配准候选变换
    std::vector<pcl::registration::MatchingCandidate,Eigen::aligned_allocator<pcl::registration::MatchingCandidate>> transformations;

    std::vector<int> VMCS;

    /** \brief detected loops. */
    std::vector<std::vector<std::pair<int, bool>>> loops;

    /** \brief Uncertain edges set. */
    std::vector<std::vector<std::pair<int, bool>>> UES;

    /** \brief Uncertain edges after knowledge sharing. */
    std::vector<std::vector<std::pair<int, bool>>> UES_pruned;

    /** \brief Vaild edges set. */
    std::vector<std::vector<std::pair<int, bool>>> VES;

    // 无效边
    std::set<int> IES;

    /** \brief Vaild edges. */
    std::set<int> VE;
};
