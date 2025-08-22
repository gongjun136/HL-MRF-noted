#include "../include/Graph/scan_graph_inference.h"

ScanGraphInference::ScanGraphInference(/* args */) : rotation_accuracy(0.087),
                                                     translation_accuracy(0.5)
{
}

ScanGraphInference::~ScanGraphInference()
{
}

/**
 * @brief ɨ��ͼ������������������Ч�߼��� IES��
 *
 * ������������ ScanGraphInference �ĺ������̣�
 * 1. �������ı߼��Ϻͱ任��ѡ�����Ƿ�Ϸ���
 * 2. ����ɨ��ͼ�����л�·��⣻
 * 3. �������л�·������ nextNode() ��黷·�бߵ������ԣ�
 * 4. ���� eliminateUncertaintyEdges() ������ȷ���� (UES)��
 * 5. ��� MCS (Maximum Clique Size) ��ֵ��ɸѡ�����յ���Ч�߼��� IES��
 *
 * @param[in] length   ��⻷·����󳤶ȣ�����������ȣ�����ͼ̫��ʱ���������ߣ�
 * @param[in] t_mcs    MCS ��ֵ����ĳ���ߵ� MCS < t_mcs�����ж��ñ�Ϊ��Ч��
 * @return std::set<int> ��Ч���������� IES��Invalid Edge Set��
 */
std::set<int> ScanGraphInference::inference(int length, int t_mcs)
{
    // ================== Step 1. ��ʼ����� ==================
    if (edges.size() < 3)
    {
        // �������̫�٣��޷��γ���Ч��·
        std::cerr << "--> ERROR: graph is too small to inference!\n";
        IES.clear();
        return IES;
    }

    if (edges.size() != transformations.size())
    {
        // ���������Ӧ�ı任��ѡ������һ�£�˵�����ݴ���
        std::cerr << "edges: " << edges.size() << "\t transformations: " << transformations.size() << "\n";
        std::cerr << "--> ERROR: transformations should be the same size as nodes.\n";
        IES.clear();
        return IES;
    }

    // ================== Step 2. ����ͼ�����л�·��� ==================
    graph.setPointCloudPairs(edges);         ///< ���߼��ϴ���ͼ�ṹ
    graph.detectLoops(loops, length);        ///< ��⻷·������洢�� loops ��

    // ================== Step 3. ����ÿ����·��ִ����������֤ ==================
    for (std::vector<std::vector<std::pair<int, bool>>>::iterator loop = loops.begin(); loop != loops.end(); loop++)
    {
        int position = 0;
        Eigen::Matrix4f trans = Eigen::Matrix4f::Identity(); ///< ��ʼ�任Ϊ��λ����
        nextNode(trans, *loop, position);                    ///< ������·�ڵ㣬�ۼƱ任�����һ����
    }

    // ================== Step 4. ������ȷ���� (UES) ==================
    eliminateUncertaintyEdges();

    // ================== Step 5. ��� MCS ��ֵɸѡ��Ч�� ==================
    for (int i = 0; i < UES_pruned.size(); i++)
    {
        int count = 0;
        for (int j = 0; j < UES_pruned[i].size(); j++)
        {
            // ���ĳ���ߵ�����Ŵ�С < ��ֵ�����ж��ñ���Ч
            if (VMCS[UES_pruned[i][j].first] < t_mcs)
            {
                IES.insert(UES_pruned[i][j].first); ///< ������Ч�߼���
                count++;
            }
        }

        // ����ԭ������ˡ���·��Ч��û�б߱�ɾ�����Ķ����߼������ѱ�ע�͵�
        // if (count == 0) { ... }
    }

    return IES; ///< �������յ���Ч�߼���
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