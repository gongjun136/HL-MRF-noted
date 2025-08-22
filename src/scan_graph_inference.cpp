#include "../include/Graph/scan_graph_inference.h"

ScanGraphInference::ScanGraphInference(/* args */) : rotation_accuracy(0.087),
                                                     translation_accuracy(0.5)
{
}

ScanGraphInference::~ScanGraphInference()
{
}

/**
 * @brief 扫描图推理主函数（推理无效边集合 IES）
 *
 * 本函数是整个 ScanGraphInference 的核心流程：
 * 1. 检查输入的边集合和变换候选集合是否合法；
 * 2. 构建扫描图并进行环路检测；
 * 3. 遍历所有环路，调用 nextNode() 检查环路中边的连贯性；
 * 4. 调用 eliminateUncertaintyEdges() 消除不确定边 (UES)；
 * 5. 结合 MCS (Maximum Clique Size) 阈值，筛选出最终的无效边集合 IES。
 *
 * @param[in] length   检测环路的最大长度（限制搜索深度，避免图太大时计算量过高）
 * @param[in] t_mcs    MCS 阈值，若某条边的 MCS < t_mcs，则判定该边为无效边
 * @return std::set<int> 无效边索引集合 IES（Invalid Edge Set）
 */
std::set<int> ScanGraphInference::inference(int length, int t_mcs)
{
    // ================== Step 1. 初始化检查 ==================
    if (edges.size() < 3)
    {
        // 如果边数太少，无法形成有效环路
        std::cerr << "--> ERROR: graph is too small to inference!\n";
        IES.clear();
        return IES;
    }

    if (edges.size() != transformations.size())
    {
        // 若边数与对应的变换候选数量不一致，说明数据错误
        std::cerr << "edges: " << edges.size() << "\t transformations: " << transformations.size() << "\n";
        std::cerr << "--> ERROR: transformations should be the same size as nodes.\n";
        IES.clear();
        return IES;
    }

    // ================== Step 2. 构建图并进行环路检测 ==================
    graph.setPointCloudPairs(edges);         ///< 将边集合传入图结构
    graph.detectLoops(loops, length);        ///< 检测环路，结果存储在 loops 中

    // ================== Step 3. 遍历每个环路，执行连贯性验证 ==================
    for (std::vector<std::vector<std::pair<int, bool>>>::iterator loop = loops.begin(); loop != loops.end(); loop++)
    {
        int position = 0;
        Eigen::Matrix4f trans = Eigen::Matrix4f::Identity(); ///< 初始变换为单位矩阵
        nextNode(trans, *loop, position);                    ///< 遍历环路节点，累计变换并检查一致性
    }

    // ================== Step 4. 消除不确定边 (UES) ==================
    eliminateUncertaintyEdges();

    // ================== Step 5. 结合 MCS 阈值筛选无效边 ==================
    for (int i = 0; i < UES_pruned.size(); i++)
    {
        int count = 0;
        for (int j = 0; j < UES_pruned[i].size(); j++)
        {
            // 如果某条边的最大团大小 < 阈值，则判定该边无效
            if (VMCS[UES_pruned[i][j].first] < t_mcs)
            {
                IES.insert(UES_pruned[i][j].first); ///< 加入无效边集合
                count++;
            }
        }

        // 这里原本设计了“环路无效但没有边被删除”的兜底逻辑，但已被注释掉
        // if (count == 0) { ... }
    }

    return IES; ///< 返回最终的无效边集合
}

void ScanGraphInference::eliminateUncertaintyEdges()
{
    for (int i = 0; i < VES.size(); i++)
    {
        for (int j = 0; j < VES[i].size(); j++)
        {
            VE.insert(VES[i][j].first);
        }
    }

    // knowledge sharing
    std::vector<std::vector<std::pair<int, bool>>> UES_pruned_temp;
    for (int i = 0; i < UES.size(); i++)
    {
        std::vector<std::pair<int, bool>> temp;
        for (int j = 0; j < UES[i].size(); j++)
        {
            int id = UES[i][j].first;
            if (std::find(VE.begin(), VE.end(), id) == VE.end())
            {
                temp.push_back(UES[i][j]);
            }
        }
        if (temp.size() != 0)
        {
            UES_pruned_temp.push_back(temp);
        }
    }

    // simple reasoning

    for (int i = 0; i < UES_pruned_temp.size(); i++)
    {
        if (UES_pruned_temp[i].size() == 1)
        {
            IES.insert(UES_pruned_temp[i][0].first);
        }
        else
        {
            UES_pruned.push_back(UES_pruned_temp[i]);
        }
    }
}

bool ScanGraphInference::loopClosure(Eigen::Matrix4f &trans, int n)
{
    Eigen::Affine3f rot;
    rot = trans.block<3, 3>(0, 0);
    const float rotation = Eigen::AngleAxisf(rot.rotation()).angle();
    const float translation = trans.rightCols<1>().head(3).norm();

    const float acc_tol_factor = 2.0f;
    const float rotation_ratio = rotation / (rotation_accuracy * sqrtf(float(n)) * acc_tol_factor);
    const float translation_ratio = translation / (translation_accuracy * sqrtf(float(n)) * acc_tol_factor);

    const float mean_ratio = 0.5 * (rotation_ratio + translation_ratio);

    return mean_ratio < 1;
}

void ScanGraphInference::nextNode(Eigen::Matrix4f &cur_trans, std::vector<std::pair<int, bool>> &loop, int position)
{
    Eigen::Matrix4f new_trans;
    std::pair<int, bool> edge = loop[position];
    Eigen::Matrix4f &next_trans = transformations[edge.first].transformation;
    combineTransformation(cur_trans, next_trans, new_trans, !edge.second);
    if (position < loop.size() - 1)
        nextNode(new_trans, loop, position + 1);
    else
    {
        bool vaildity = loopClosure(new_trans, position + 1);

        if (vaildity)
            VES.push_back(loop);
        else
        {
            UES.push_back(loop);
        }
    }
}

void ScanGraphInference::combineTransformation(Eigen::Matrix4f &a, Eigen::Matrix4f b, Eigen::Matrix4f &c, bool inverse)
{
    c.setZero();
    if (b.isZero() || a.isZero())
        return;

    c = b;
    if (!inverse)
        c *= a;
    else
        c = b.colPivHouseholderQr().solve(a);
    return;
}