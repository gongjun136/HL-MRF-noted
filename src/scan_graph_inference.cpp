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

/**
 * @brief ������ȷ���� (Uncertainty Edges) �������ڲ����ϡ�
 *
 * �ú�������Ҫ���̣�
 * 1. ����������Ч��·���� VES�����������бߵ��������� VE��Valid Edges ���ϣ���
 * 2. ����Ч��·���� UES ���С�֪ʶ����ʽ�Ĳü���
 *    - ���ĳ����Ч��·�еı��Ѿ��� VE �г��֣�����ȷ����Ч������Ӹ���Ч��·���޳���Щ�ߣ�
 *    - ʣ��Ĳ��ִ�����ʱ���� UES_pruned_temp��
 * 3. ������
 *    - ���ĳ���ü������Ч��·ֻʣ�� 1 ���ߣ���˵���ñ߱�����ǡ������Ĳ�ȷ���ߡ���ֱ�Ӽ��� IES��Induced Edges Set����
 *    - ���򣬽��û�·�������յ� UES_pruned����������Ч��·���ϣ���
 *
 * @note
 * - VE����Ч�߼��ϣ���������Ч��·����ȡ�ı���������
 * - UES��ԭʼ��Ч��·���ϡ�
 * - UES_pruned_temp���м���������ڱ����޳� VE �б�֮�����Ч��·��
 * - UES_pruned�����ձ�������Ч��·���ϡ�
 * - IES���ɼ�����õ��ġ�������ȷ���ߡ����ϡ�
 */
void ScanGraphInference::eliminateUncertaintyEdges()
{
    // ===== Step 1: ��������Ч��· VES ����ȡ������������ VE =====
    for (int i = 0; i < VES.size(); i++)
    {
        for (int j = 0; j < VES[i].size(); j++)
        {
            // VES[i][j].first ��ʾ�ߵ�����ֵ
            VE.insert(VES[i][j].first);
        }
    }

    // ===== Step 2: ��֪ʶ�����ü� UES =====
    std::vector<std::vector<std::pair<int, bool>>> UES_pruned_temp;
    for (int i = 0; i < UES.size(); i++)
    {
        std::vector<std::pair<int, bool>> temp;
        for (int j = 0; j < UES[i].size(); j++)
        {
            int id = UES[i][j].first;

            // ����ñ� id û�г����� VE �У���������Ч�ߣ���������
            if (std::find(VE.begin(), VE.end(), id) == VE.end())
            {
                temp.push_back(UES[i][j]);
            }
        }

        // ����ü���·����ʣ��ߣ��򱣴�����
        if (temp.size() != 0)
        {
            UES_pruned_temp.push_back(temp);
        }
    }

    // ===== Step 3: ������ =====
    for (int i = 0; i < UES_pruned_temp.size(); i++)
    {
        if (UES_pruned_temp[i].size() == 1)
        {
            // ����ü���ֻʣ 1 ���ߣ�˵���ñ��ǹ����Ĳ�ȷ���� -> ���� IES
            IES.insert(UES_pruned_temp[i][0].first);
        }
        else
        {
            // ��������Ϊ��ߵĲ�ȷ����·��������
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
 * @brief �ݹ��ػ�·�ۼ�λ�ˣ�����ĩ�˽��бջ�һ�����ж���
 *
 * ����ʼ�ۼƱ任 @p cur_trans ���������� @p loop[position] ָ���ġ��������뷽�򡱣�
 * ���ñ߶�Ӧ����Ա任�뵱ǰ�ۼƱ任�����ϣ�������Ϊ������ʹ�øñ߱任���棩��
 * �õ��µ��ۼƱ任���ݹ��ƽ������ﻷ·ĩ��ʱ������ `loopClosure()` ���ۼ�������
 * �ж�������������·�ֱ���� VES����Ч���� UES����Ч����
 *
 * ���ϼ���ͨ�� `combineTransformation(cur_trans, next_trans, new_trans, !direction)` ��ɣ�
 * �� `direction==true` ��ʾ���洢����ǰ������ȡ�棩��`direction==false` ��ʾ�ط���ǰ����ȡ�棩��
 *
 * @param[in]  cur_trans  �ӻ�·����ۼƵ�����ǰ�ڵ㡱����α任��4��4������λ�ˣ����������޸������ݣ���
 * @param[in]  loop       ��·�����У�Ԫ��Ϊ `(edge_index, direction)`��
 *                        - `edge_index`���� `transformations` �еı�������
 *                        - `direction==true`�����洢����ʹ�øñߣ�`false`��ʹ�øñߵ���任��
 * @param[in]  position   ��ǰ�����Ļ�·λ�ã�0 �� position < loop.size()����
 *
 * @pre  `!loop.empty()` �� `0 �� position < loop.size()`������ `edge_index` ��Ϊ��Ч�±ꣻ
 *       �� `direction==false` ʱ����Ӧ�߱任���档
 * @post �����ε��ô�����·ĩ�ˣ��� `loopClosure()` ���ж������������·���� `VES` �� `UES`��
 *
 * @note �ú���Ϊ���Ϊ |loop| �ĵݹ���̣�ʱ�临�Ӷ� O(|loop|)���ռ临�Ӷ��ɵݹ���Ⱦ�����
 * @see  combineTransformation(), loopClosure(), VES, UES, transformations
 */
void ScanGraphInference::nextNode(Eigen::Matrix4f& cur_trans, std::vector<std::pair<int, bool>>& loop, int position)
{
    // new_trans���ӻ�·����ۼƵ�����һ���ڵ㡱�ĸ��ϱ任����ǰ������Ľ����
    Eigen::Matrix4f new_trans;

    // ��ǰҪ����ġ��ߡ�������edge.first Ϊ���� transformations �е�������edge.second Ϊ�����ǣ�
    // Լ����edge.second == true ��ʾ�� transformations �д洢�ġ�����ʹ�øñߣ�
    //      edge.second == false ��ʾ���෴����ʹ�ã���Ҫ�Ըñߵı任ȡ�棩��
    std::pair<int, bool> edge = loop[position];

    // ȡ���ñߵĺ�ѡλ�˱任��4x4 ��ξ��󣩣�ע�������ǡ��ߡ������λ�ˣ����ǽڵ�ľ���λ��
    Eigen::Matrix4f& next_trans = transformations[edge.first].transformation;

    // ������ǰ�ۼƱ任���롰���߱任����һ�θ��ϣ��õ�ָ����һ���ڵ���ۼƱ任 new_trans��
    //  - �� inverse=false���� edge.second==true ʱ������ȥ���� !edge.second==false����
    //      new_trans = next_trans * cur_trans                  �����洢����ǰ����
    //  - �� inverse=true���� edge.second==false ʱ������ȥ���� !edge.second==true����
    //      new_trans = next_trans^{-1} * cur_trans             ���ط����ߣ���Ҫ�øñ߱任���棩
    //  combineTransformation �ڲ�ͨ�� QR ��� b * X = a �ķ�ʽʵ�� ��b^{-1} * a��
    combineTransformation(cur_trans, next_trans, new_trans, !edge.second);

    // ����δ�ߵ��û�·�����һ���ߣ���ݹ�ش�����һ���ߣ������ۼƸ��ϱ任
    if (position < loop.size() - 1)
        nextNode(new_trans, loop, position + 1);
    else
    {
        // ����������·��ʹ���ۼƱ任 new_trans ���ջ�һ���Լ��飺
        // n = position + 1 Ϊ����ջ��ı�����loopClosure �����ת/ƽ���� sqrt(n) �뾫����ֵ���ź���ƽ����ֵ
        // ��mean_ratio<1 ��Ϊ�û�·��Ч������ͬ�ļ� loopClosure ��ʵ�֡�
        bool vaildity = loopClosure(new_trans, position + 1);

        // ���ݱջ���Ч�ԣ����û�·������룺
        //  - VES��Valid Edge Set �Ļ�·���ϣ����ջ�Լ��һ�¡��ɱ���Ϊ����Ǣ���Ļ�·
        //  - UES��Uncertain Edge Set �Ļ�·���ϣ����ջ���һ�¡����ڳ�ͻ�Ļ�·���������һ���ü�/����
        if (vaildity)
            VES.push_back(loop);
        else
        {
            UES.push_back(loop);
        }
    }
}

/**
 * @brief �������λ�˱任����4x4 ��ξ��󣩣�֧�����������򸴺ϡ�
 *
 * �˺������ڽ��任���� @p a �� @p b ���и������㣬����������� @p c��
 * ���ݲ��� @p inverse ��ȡֵ��ѡ��ͬ�ļ��㷽ʽ��
 * - �� inverse == false��ִ�����򸴺ϣ�c = b * a��
 * - �� inverse == true ��ִ�����򸴺ϣ�c = b^{-1} * a��ͨ�� QR �ֽ�������Է���ʵ�֡�
 *
 * ������������������ @p a �� @p b Ϊ�������ֱ�ӷ��������c ȫ 0����
 *
 * @param[in]  a        ����任����4x4������Ϊ�ҳ˾���
 * @param[in]  b        ����任����4x4������Ϊ��˾���
 * @param[out] c        ����ĸ��Ͻ����4x4����
 * @param[in]  inverse  �Ƿ��ڸ���ʱ�� b ȡ�棺
 *                      - false��c = b * a
 *                      - true ��c = b^{-1} * a ��ͨ�� QR �ֽ���⣩
 *
 * @note ʹ�� QR �ֽ������任���ȶ�������ֱ�Ӿ������棩��
 * @see nextNode(), loopClosure()
 */
void ScanGraphInference::combineTransformation(Eigen::Matrix4f &a, Eigen::Matrix4f b, Eigen::Matrix4f &c, bool inverse)
{
    // ����������ʼ��Ϊ�㣬��ʾ����Ч�任��
    c.setZero();

    // ������� a �� b ���������Ϊ������Ч��ֱ�ӷ��أ����� c Ϊ�㣩
    if (b.isZero() || a.isZero())
        return;

    // Ĭ������������ b
    c = b;

    if (!inverse)
        // ���򸴺ϣ�c = b * a
        // ��ʾ��Ӧ�� a����Ӧ�� b���ȼ��������������任��
        c *= a;
    else
        // ���򸴺ϣ�c = b^{-1} * a
        // ͨ����� b * X = a �õ� X = b^{-1} * a����ֱ�Ӿ���������ȶ�
        c = b.colPivHouseholderQr().solve(a);

    return;
}