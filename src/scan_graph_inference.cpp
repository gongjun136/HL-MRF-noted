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

/**
 * @brief 消除不确定边 (Uncertainty Edges) 并更新内部集合。
 *
 * 该函数的主要流程：
 * 1. 遍历所有有效环路集合 VES，将其中所有边的索引加入 VE（Valid Edges 集合）。
 * 2. 对无效环路集合 UES 进行“知识共享”式的裁剪：
 *    - 如果某条无效环路中的边已经在 VE 中出现（即被确认有效），则从该无效环路中剔除这些边；
 *    - 剩余的部分存入临时容器 UES_pruned_temp。
 * 3. 简单推理：
 *    - 如果某条裁剪后的无效环路只剩下 1 条边，则说明该边本身就是“孤立的不确定边”，直接加入 IES（Induced Edges Set）。
 *    - 否则，将该环路加入最终的 UES_pruned（保留的无效环路集合）。
 *
 * @note
 * - VE：有效边集合（从所有有效环路中提取的边索引）。
 * - UES：原始无效环路集合。
 * - UES_pruned_temp：中间变量，用于保存剔除 VE 中边之后的无效环路。
 * - UES_pruned：最终保留的无效环路集合。
 * - IES：由简单推理得到的“单独不确定边”集合。
 */
void ScanGraphInference::eliminateUncertaintyEdges()
{
    // ===== Step 1: 从所有有效环路 VES 中提取边索引，加入 VE =====
    for (int i = 0; i < VES.size(); i++)
    {
        for (int j = 0; j < VES[i].size(); j++)
        {
            // VES[i][j].first 表示边的索引值
            VE.insert(VES[i][j].first);
        }
    }

    // ===== Step 2: “知识共享”裁剪 UES =====
    std::vector<std::vector<std::pair<int, bool>>> UES_pruned_temp;
    for (int i = 0; i < UES.size(); i++)
    {
        std::vector<std::pair<int, bool>> temp;
        for (int j = 0; j < UES[i].size(); j++)
        {
            int id = UES[i][j].first;

            // 如果该边 id 没有出现在 VE 中（即不是有效边），则保留它
            if (std::find(VE.begin(), VE.end(), id) == VE.end())
            {
                temp.push_back(UES[i][j]);
            }
        }

        // 如果裁剪后环路还有剩余边，则保存下来
        if (temp.size() != 0)
        {
            UES_pruned_temp.push_back(temp);
        }
    }

    // ===== Step 3: 简单推理 =====
    for (int i = 0; i < UES_pruned_temp.size(); i++)
    {
        if (UES_pruned_temp[i].size() == 1)
        {
            // 如果裁剪后只剩 1 条边，说明该边是孤立的不确定边 -> 加入 IES
            IES.insert(UES_pruned_temp[i][0].first);
        }
        else
        {
            // 否则将其作为多边的不确定环路保留下来
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
/**
 * @brief 递归沿环路累计位姿，并在末端进行闭环一致性判定。
 *
 * 从起始累计变换 @p cur_trans 出发，依据 @p loop[position] 指定的“边索引与方向”，
 * 将该边对应的相对变换与当前累计变换做复合（若方向为反向则使用该边变换的逆），
 * 得到新的累计变换并递归推进。到达环路末端时，调用 `loopClosure()` 对累计误差进行
 * 判定，并将整条环路分别加入 VES（有效）或 UES（无效）。
 *
 * 复合计算通过 `combineTransformation(cur_trans, next_trans, new_trans, !direction)` 完成：
 * 当 `direction==true` 表示按存储方向前进（不取逆），`direction==false` 表示沿反向前进（取逆）。
 *
 * @param[in]  cur_trans  从环路起点累计到“当前节点”的齐次变换（4×4，刚体位姿；本函数不修改其内容）。
 * @param[in]  loop       环路边序列；元素为 `(edge_index, direction)`。
 *                        - `edge_index`：在 `transformations` 中的边索引；
 *                        - `direction==true`：按存储方向使用该边；`false`：使用该边的逆变换。
 * @param[in]  position   当前处理到的环路位置（0 ≤ position < loop.size()）。
 *
 * @pre  `!loop.empty()` 且 `0 ≤ position < loop.size()`；所有 `edge_index` 均为有效下标；
 *       当 `direction==false` 时，对应边变换可逆。
 * @post 若本次调用处理到环路末端，则按 `loopClosure()` 的判定结果将整条环路加入 `VES` 或 `UES`。
 *
 * @note 该函数为深度为 |loop| 的递归过程，时间复杂度 O(|loop|)；空间复杂度由递归深度决定。
 * @see  combineTransformation(), loopClosure(), VES, UES, transformations
 */
void ScanGraphInference::nextNode(Eigen::Matrix4f& cur_trans, std::vector<std::pair<int, bool>>& loop, int position)
{
    // new_trans：从环路起点累计到“下一个节点”的复合变换（当前步计算的结果）
    Eigen::Matrix4f new_trans;

    // 当前要处理的“边”描述（edge.first 为边在 transformations 中的索引，edge.second 为方向标记）
    // 约定：edge.second == true 表示按 transformations 中存储的“正向”使用该边；
    //      edge.second == false 表示沿相反方向使用（需要对该边的变换取逆）。
    std::pair<int, bool> edge = loop[position];

    // 取出该边的候选位姿变换（4x4 齐次矩阵），注意这里是“边”的相对位姿，不是节点的绝对位姿
    Eigen::Matrix4f& next_trans = transformations[edge.first].transformation;

    // 将“当前累计变换”与“本边变换”做一次复合，得到指向下一个节点的累计变换 new_trans：
    //  - 当 inverse=false（即 edge.second==true 时，传进去的是 !edge.second==false）：
    //      new_trans = next_trans * cur_trans                  （按存储方向前进）
    //  - 当 inverse=true（即 edge.second==false 时，传进去的是 !edge.second==true）：
    //      new_trans = next_trans^{-1} * cur_trans             （沿反向走，需要用该边变换的逆）
    //  combineTransformation 内部通过 QR 求解 b * X = a 的方式实现 “b^{-1} * a”
    combineTransformation(cur_trans, next_trans, new_trans, !edge.second);

    // 若还未走到该环路的最后一条边，则递归地处理下一条边，继续累计复合变换
    if (position < loop.size() - 1)
        nextNode(new_trans, loop, position + 1);
    else
    {
        // 走完整个环路后，使用累计变换 new_trans 做闭环一致性检验：
        // n = position + 1 为参与闭环的边数；loopClosure 会把旋转/平移误差按 sqrt(n) 与精度阈值缩放后求平均比值
        // （mean_ratio<1 认为该环路有效）。见同文件 loopClosure 的实现。
        bool vaildity = loopClosure(new_trans, position + 1);

        // 根据闭环有效性，将该环路整体分入：
        //  - VES（Valid Edge Set 的环路集合）：闭环约束一致、可被认为“自洽”的环路
        //  - UES（Uncertain Edge Set 的环路集合）：闭环不一致、存在冲突的环路（后续会进一步裁剪/推理）
        if (vaildity)
            VES.push_back(loop);
        else
        {
            UES.push_back(loop);
        }
    }
}

/**
 * @brief 组合两个位姿变换矩阵（4x4 齐次矩阵），支持正向与逆向复合。
 *
 * 此函数用于将变换矩阵 @p a 与 @p b 进行复合运算，并将结果存入 @p c。
 * 根据参数 @p inverse 的取值，选择不同的计算方式：
 * - 当 inverse == false：执行正向复合，c = b * a；
 * - 当 inverse == true ：执行逆向复合，c = b^{-1} * a，通过 QR 分解求解线性方程实现。
 *
 * 特殊情况：若输入矩阵 @p a 或 @p b 为零矩阵，则直接返回零矩阵（c 全 0）。
 *
 * @param[in]  a        输入变换矩阵（4x4），作为右乘矩阵。
 * @param[in]  b        输入变换矩阵（4x4），作为左乘矩阵。
 * @param[out] c        输出的复合结果（4x4）。
 * @param[in]  inverse  是否在复合时对 b 取逆：
 *                      - false：c = b * a
 *                      - true ：c = b^{-1} * a （通过 QR 分解求解）
 *
 * @note 使用 QR 分解求解逆变换更稳定（避免直接矩阵求逆）。
 * @see nextNode(), loopClosure()
 */
void ScanGraphInference::combineTransformation(Eigen::Matrix4f &a, Eigen::Matrix4f b, Eigen::Matrix4f &c, bool inverse)
{
    // 将结果矩阵初始化为零，表示“无效变换”
    c.setZero();

    // 如果输入 a 或 b 是零矩阵，认为输入无效，直接返回（保持 c 为零）
    if (b.isZero() || a.isZero())
        return;

    // 默认先令结果等于 b
    c = b;

    if (!inverse)
        // 正向复合：c = b * a
        // 表示先应用 a，再应用 b，等价于依次沿两个变换走
        c *= a;
    else
        // 逆向复合：c = b^{-1} * a
        // 通过求解 b * X = a 得到 X = b^{-1} * a，比直接矩阵求逆更稳定
        c = b.colPivHouseholderQr().solve(a);

    return;
}