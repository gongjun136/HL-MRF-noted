/*
 * Software License Agreement
 *
 *  Copyright (c) P.W.Theiler, 2015
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef IMPL_TRANSFORMATION_GRAPH_H_
#define IMPL_TRANSFORMATION_GRAPH_H_

#include "transformation_graph.h"

/**
 * @brief 在变换图 (TransformationGraph) 中检测环路 (loop closure)
 *
 * @tparam Scalar       数值类型（如 float 或 double）
 * @param[out] loops    输出的环路集合，每个环路是由边索引 (DirectedIndices) 组成
 * @param[in] max_loop_size 最大环路长度限制
 *
 * @return int          返回检测到的环路数量
 *
 * @details
 * 该函数通过深度优先搜索 (DFS) 找到图中的所有环路，并进行如下处理：
 * 1. 将节点环路 (node loop) 转换为边环路 (edge loop)；
 * 2. 去除重复环路；
 * 3. 剔除冗余的大环路（如果已包含在较小环路中）。
 */
template <typename Scalar>
int pcl::registration::TransformationGraph<Scalar>::detectLoops(
    DirectedIndicesList &loops,
    int max_loop_size)
{
    // 1. 初始化图的邻接表和映射表 (graph_, map_)
    initCompute();

    const std::size_t nr_clouds = graph_.size(); ///< 图中节点数（点云数量）
    const std::size_t nr_pairs = pairs_.size();  ///< 图中边数（点云对数量）

    // 2. 使用 DFS 搜索环路，得到的是基于“节点序列”的环路集合
    IndicesList node_loops;
    depthFirstSearch(node_loops, max_loop_size);

    // 3. 将“节点环路”转换为“边环路”（因为最终需要的是边对应的变换约束）
    //    - loops 存储的是边的索引
    const std::size_t nr_loops = node_loops.size();
    loops.resize(nr_loops);

    for (std::size_t l = 0; l < nr_loops; l++)
    {
        const std::size_t loop_size = node_loops[l].size() - 1; ///< 环路长度（节点数-1 = 边数）
        loops[l].resize(loop_size);

        for (std::size_t i = 0; i < loop_size; i++)
        {
            // map_ 用于从 (node_i, node_j) 映射到边的索引，以及边的方向
            loops[l][i] = map_[node_loops[l][i] * nr_clouds + node_loops[l][i + 1]];
        }
    }

    // 4. 去除重复环路（判断是否由相同边组成，环路的方向不考虑）
    for (DirectedIndicesList::iterator it_a = loops.begin();
         it_a != loops.end() - 1 && it_a != loops.end(); it_a++)
    {
        for (DirectedIndicesList::iterator it_b = it_a + 1; it_b != loops.end();)
        {
            if (hasSameElements(*it_a, *it_b)) ///< 如果两个环路的边集合相同
                it_b = loops.erase(it_b);      ///< 删除重复的环路
            else
                it_b++;
        }
    }

    // 5. 剔除冗余大环路
    //    思路：如果一个大环路和已有的小环路共享 ≥2 条边，则认为冗余，删除该大环路
    std::sort(loops.begin(), loops.end(), smaller); ///< 按环路长度从小到大排序

    Eigen::MatrixXi lookup = Eigen::MatrixXi::Zero(nr_pairs, nr_pairs); ///< 记录已有环路中的边对关系

    for (DirectedIndicesList::iterator it = loops.begin(); it != loops.end();)
    {
        if (it->size() <= max_loop_size)
        {
            // 将当前环路的边添加进 lookup，保留该环路
            addLoopToLookup(*it++, lookup);
        }
        else
        {
            // 如果当前大环路和已有环路共享的边数 < 2，则保留；否则删除
            if (compareLoopToLookup(*it, lookup) < 2)
                it++;
            else
                it = loops.erase(it);
        }
    }

    // 6. 返回最终环路数量
    return (static_cast<int>(loops.size()));
};

/**
 * @brief 初始化 TransformationGraph 的图结构和映射表
 *
 * @tparam Scalar       数值类型（float 或 double）
 *
 * @details
 * 该函数的主要功能是：
 * 1. 根据点云对 (pairs_) 构建无向图 graph_；
 * 2. 对每个节点的连接关系进行排序，便于后续遍历；
 * 3. 初始化映射表 map_，用于快速查询“节点对 → 边索引及方向”；
 * 4. 避免重复计算，通过标志位 graph_is_updated_ 控制是否需要重新构建。
 *
 * 映射表 map_ 的关键思想：
 * - 每个有序对 (idx1, idx2) 映射到一个 pair：
 *   - 第一个元素 = 边索引（对应 pairs_ 的下标）
 *   - 第二个元素 = 是否为正向 (true 表示 idx1→idx2 与 pairs_ 一致；false 表示反向)
 */
template <typename Scalar>
void pcl::registration::TransformationGraph<Scalar>::initCompute()
{
    // 如果图已经更新过，则直接返回，避免重复计算
    if (graph_is_updated_)
        return;

    // 找到 pairs_ 中的最大节点索引，用于确定图中节点总数
    Pairs::const_iterator it = std::max_element(
        pairs_.begin(), pairs_.end(), bigger_pair_element);

    const std::size_t nr_clouds = static_cast<std::size_t>(
        std::max(it->first, it->second) + 1); // 节点总数

    // 调整 graph_ 大小，每个节点维护一个邻接表
    graph_.resize(nr_clouds);

    // === 1. 构建无向图 adjacency list ===
    const std::size_t nr_pairs = pairs_.size();
    for (std::size_t i = 0; i < nr_pairs; i++)
    {
        // 双向连接：first → second, second → first
        graph_[pairs_[i].first].push_back(pairs_[i].second);
        graph_[pairs_[i].second].push_back(pairs_[i].first);
    }

    // === 2. 将每个节点的邻接表排序，保证遍历时顺序一致 ===
    for (std::size_t i = 0; i < nr_clouds; i++)
        std::sort(graph_[i].begin(), graph_[i].end());

    // === 3. 初始化映射表 map_ ===
    // map_ 的大小 = nr_clouds * nr_clouds（笛卡尔积），
    // 用于快速定位“节点对 → 边索引”
    map_.resize(nr_clouds * nr_clouds);

    // 填充映射表：
    //   - 正向 (first→second) 存 true
    //   - 反向 (second→first) 存 false
    for (std::size_t i = 0; i < nr_pairs; i++)
    {
        map_[pairs_[i].first * nr_clouds + pairs_[i].second] =
            std::make_pair(static_cast<int>(i), true);

        map_[pairs_[i].second * nr_clouds + pairs_[i].first] =
            std::make_pair(static_cast<int>(i), false);
    }

    // === 4. 设置更新标志，避免重复构建 ===
    graph_is_updated_ = true;
}

/**
 * @brief 在 TransformationGraph 中执行深度优先搜索 (DFS)，用于检测可能的环路
 *
 * @tparam Scalar         数值类型（float 或 double）
 * @param[out] loops      存储搜索到的环路，每个环路由节点索引序列表示
 * @param[in]  max_loop_size  最大允许的环路长度（节点数限制）
 *
 * @return 搜索到的环路数量
 *
 * @details
 * 1. 遍历图中每个节点作为起始点；
 * 2. 从当前节点出发，沿着未访问的邻接节点进行深度优先遍历；
 * 3. 将路径记录到 loops 中（后续会在 detectLoops 中转换成边序列）。
 *
 * 关键点：
 * - visited 数组控制哪些节点已被探索，避免重复；
 * - removeVisited 用于在递归前移除已访问过的邻居节点；
 * - nextDepth 递归探索时会检查是否满足环路闭合条件；
 * - 每次以某个 node 为起点，搜索完成后才标记该节点为已访问，
 *   这样保证能检测到回到起点的环路（loop closure）。
 */
template <typename Scalar>
int pcl::registration::TransformationGraph<Scalar>::depthFirstSearch(
    IndicesList& loops,
    int max_loop_size)
{
    // === 获取节点总数 ===
    const std::size_t nr_nodes = graph_.size();

    // 记录每个节点是否已访问
    std::vector<bool> visited(nr_nodes, false);

    // === 遍历图中的每个节点，作为 DFS 起点 ===
    for (std::size_t node = 0; node < nr_nodes; node++)
    {
        // 初始化路径，仅包含当前起点
        Indices path(1, static_cast<int>(node));

        // 获取当前节点的邻居节点列表
        Indices indices = graph_[node];

        // 去掉已访问过的节点（防止走回头路）
        removeVisited(visited, indices);

        const std::size_t nr_adjacent = indices.size();

        // === 遍历所有未访问的邻居，递归探索更深的路径 ===
        for (std::size_t a = 0; a < nr_adjacent; a++)
            nextDepth(indices[a], path, visited, max_loop_size, loops);

        // === 最后才把当前节点标记为已访问 ===
        // 这样保证 DFS 可以检测到环路闭合回到起点
        visited[node] = true;
    }

    // 返回搜索到的环路数量
    return (static_cast<int>(loops.size()));
};

///////////////////////////////////////////////////////////////////////////////////////////
template <typename Scalar>
void pcl::registration::TransformationGraph<Scalar>::removeVisited(
    const std::vector<bool> &visited,
    Indices &indices)
{
    Indices::iterator it = indices.begin();
    while (it != indices.end())
    {
        if (visited[*it])
            it = indices.erase(it);
        else
            it++;
    }
};

/**
 * @brief 深度优先递归搜索，用于扩展路径并检测环路
 *
 * @tparam Scalar            数值类型（float 或 double）
 * @param[in]  node          当前访问的节点索引
 * @param[in]  path          当前路径（节点序列），按值传递，每次递归生成一份新路径
 * @param[in]  visited       已访问节点的标记数组（按值传递，每层递归独立副本）
 * @param[in]  max_loop_size 最大允许环路长度（超过则提前中止）
 * @param[out] loops         存储检测到的环路（路径序列）
 *
 * @return true  表示找到有效环路并且递归可以提前终止
 * @return false 没有找到环路
 *
 * @details
 * 该函数是 `depthFirstSearch` 的递归子函数，执行如下逻辑：
 * 1. 将当前节点加入 path，并标记为已访问；
 * 2. 若回到起始节点 (path[0]) 且 path.size()>3，说明找到一个环路，存储到 loops；
 * 3. 若回到起始节点但环路太短（如 2 或 3 个节点），丢弃；
 * 4. 遍历当前节点的未访问邻居，递归继续探索；
 * 5. 若找到环路且路径已达到 max_loop_size，可提前终止。
 */
template <typename Scalar>
bool pcl::registration::TransformationGraph<Scalar>::nextDepth(
    std::size_t node,
    Indices path,
    std::vector<bool> visited,
    int max_loop_size,
    IndicesList& loops)
{
    // === 1. 将当前节点加入路径，并标记已访问 ===
    path.push_back(static_cast<int>(node));
    visited[node] = true;

    // === 2. 检查是否形成环路 ===
    if (node == path[0] && path.size() > 3)   // 回到起点，且长度>3，构成有效环路
    {
        loops.push_back(path);                // 保存该环路
        return (true);                        // 返回成功，通知上层递归
    }

    // === 3. 检查是否回到起点但环路太短 ===
    if (node == path[0])  // 但长度 <= 3，例如起点->邻居->起点，不算有效环
        return (false);

    // === 4. 遍历所有未访问的邻居，递归探索更深路径 ===
    Indices indices = graph_[node];          // 当前节点的邻接表
    removeVisited(visited, indices);         // 去掉已访问过的节点
    const std::size_t nr_adjacent = indices.size();

    for (std::size_t a = 0; a < nr_adjacent; a++)
    {
        // 递归探索邻居
        if (nextDepth(indices[a], path, visited, max_loop_size, loops))
        {
            // 如果找到环路，但路径未超过 max_loop_size，则继续探索
            if (path.size() < max_loop_size)
                continue;

            // 否则达到最大长度，提前终止递归
            return (true);
        }
    }

    // === 5. 未找到环路 ===
    return (false);
};

/**
 * @brief 判断两个有向边序列是否包含相同的元素
 *
 * @tparam Scalar   数值类型（float 或 double）
 * @param[in] a     有向边序列 A（每个元素通常是 pair<int,bool>，存储边索引和方向）
 * @param[in] b     有向边序列 B
 *
 * @return true     两个序列的元素集合相同（忽略顺序）
 * @return false    两个序列大小不同，或包含不同的元素
 *
 * @details
 * - 首先检查两个序列的大小是否一致；
 * - 若一致，则分别对两个序列排序（确保比较时顺序无关）；
 * - 逐一比较排序后的元素，如果任一元素不同，则返回 false；
 * - 全部相同则返回 true。
 *
 * ⚠️ 注意：
 * - 这里比较时仅检查 `.first`（边的索引），而忽略了 `.second`（方向标记），
 *   说明该函数只关心环路由哪些边组成，而不关心边的方向。
 */
template <typename Scalar>
bool pcl::registration::TransformationGraph<Scalar>::hasSameElements(
    DirectedIndices a,
    DirectedIndices b)
{
    // 1. 大小不同，必然不相等
    if (a.size() != b.size())
        return (false);

    // 2. 对两个序列排序（保证比较时顺序一致）
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());

    const std::size_t nr_elements = a.size();

    // 3. 逐一比较每个元素（只比较 pair.first，忽略方向）
    for (std::size_t i = 0; i < nr_elements; i++)
        if (a[i].first != b[i].first)
            return (false);

    // 4. 全部相等，说明两个序列包含相同元素
    return (true);
};

/**
 * @brief 将当前环路中所有边对组合记录到查找表 lookup 中
 *
 * @tparam Scalar    数值类型（float 或 double）
 * @param[in] loop   一个环路，存储为边索引序列（DirectedIndices，每个元素为 pair<int,bool>）
 *                   - `.first` 表示边的索引（唯一 ID）
 *                   - `.second` 表示边的方向（这里不关心）
 * @param[in,out] lookup  边对查找表（对称矩阵，元素为 0/1）
 *
 * @details
 * - 遍历环路中的所有边对组合 (i, j)，即环路中的任意两条边；
 * - 在 lookup 矩阵中将 (i,j) 和 (j,i) 标记为 1，表示这两条边在某个环路中共现；
 * - lookup 矩阵的大小为 [nr_pairs × nr_pairs]，其中 nr_pairs 是图中所有边的总数；
 * - 该函数用于构建“边共现关系”，后续可用于约束较大环路的去冗余操作。
 */
template <typename Scalar>
void pcl::registration::TransformationGraph<Scalar>::addLoopToLookup(
    const DirectedIndices& loop,
    Eigen::MatrixXi& lookup)
{
    // 遍历环路中所有边的组合 (外层固定一条边 it_out)
    for (DirectedIndices::const_iterator it_out = loop.begin(); it_out != loop.end() - 1; it_out++)
        // 内层取另一条边 it_in（确保不重复组合，避免 (i,j) 与 (j,i) 重复）
        for (DirectedIndices::const_iterator it_in = it_out + 1; it_in != loop.end(); it_in++)
        {
            // 在 lookup 矩阵中标记这两条边 (i,j) 共现
            lookup(it_out->first, it_in->first) = 1;
            lookup(it_in->first, it_out->first) = 1;  // 对称存储
        }
};


///////////////////////////////////////////////////////////////////////////////////////////
template <typename Scalar>
int pcl::registration::TransformationGraph<Scalar>::compareLoopToLookup(
    const DirectedIndices &loop,
    const Eigen::MatrixXi &lookup)
{
    int nr_common_edges = 0;

    // loop over all pair combinations in the present loop
    for (DirectedIndices::const_iterator it_out = loop.begin(); it_out != loop.end() - 1; it_out++)
        for (DirectedIndices::const_iterator it_in = it_out + 1; it_in != loop.end(); it_in++)
        {
            // if pair combination is present in lookup, count as common
            if (lookup(it_out->first, it_in->first) == 1 || lookup(it_in->first, it_out->first) == 1)
                nr_common_edges++;
        }

    return (nr_common_edges);
};

///////////////////////////////////////////////////////////////////////////////////////////

#endif // IMPL_TRANSFORMATION_GRAPH_H_