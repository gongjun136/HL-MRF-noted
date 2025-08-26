/**
 * =========================== 模块总览（中文） ===========================
 * HL_MRF（Hierarchical Loop‑constrained Multi‑view Registration Framework）
 * 目标：对地面三维激光扫描（TLS）多站点云进行层次化、鲁棒的全局配准。
 * 总流程：读取 -> 分块 -> 预处理 -> 粗配准 -> 一致性推断 -> 细配准(ICP) -> LUM 全局优化 -> 导出。
 * 重要成员：
 *  - voxel_size / voxel_size_icp：体素下采样与ICP对应搜索的尺度；
 *  - part_num：每个块包含的扫描数量；
 *  - use_pairs：是否使用外部提供的配对；
 *  - LUM_iterations：LUM 迭代次数；
 *  - max_consensus_set：MCS阈值；
 *  - num_threads：OpenMP 线程数。
 * 使用方法：
 *  1) setPlyPath / setOutDir / setBlockSize 等接口完成参数设置；
 *  2) performMultiviewRegistration() 启动完整流水线。
 * =====================================================================
 */


#pragma once
#include <string>
//pcl headers
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree.h>
#include <pcl/registration/matching_candidate.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/lum.h>
#include <pcl/registration/icp.h>
//boost headers
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <boost/graph/subgraph.hpp>
#include <boost/filesystem.hpp>
//std headers
#include <cmath>
#include <algorithm>
#include <fstream>
#include <set>
#include <ctime>
#include <chrono>
#include <omp.h>
//3rd party headers

//HL_MRF headers
#include "./Graph/scan_graph_inference.h"
//6DOFGR headers

//GROR headers
#include "./GROR/include/ia_gror.h"
#include "./GROR/include/gror_pre.h"
//add your favorite registration headers here...

enum CoarseRegistration
{
	GROR
};


namespace WHU
{
	class HL_MRF
	{
		using PointT = pcl::PointXYZ;
		// 使用 Boost 子图：顶点/边容器 vecS，Undirected 图；顶点/边可携带 index，边可携带权重。
		typedef boost::subgraph<
			boost::adjacency_list<
			boost::vecS, boost::vecS, boost::undirectedS,
			boost::property<boost::vertex_index_t, int>,
			boost::property<boost::edge_index_t, int,
			boost::property<boost::edge_weight_t, float>>>>
			Graph;

	public:

		HL_MRF();
		~HL_MRF();

		// 设置输入点云目录
		void setPlyPath(std::string path) { this->PLYpath = path; };

		/** \brief set predefined pair dir.
		* \param[in] path input pair path
		*/
		void setPairPath(std::string path) { this->pairs_dir = path; };

		/** \brief set PLY path.
		* \param[in] path output dir
		*/
		void setOutDir(std::string path) { this->output_dir = path; };

		// 粗配准阶段下采样体素（m）（默认 0.1）
		void setDownsampleSize(float voxel_size) { this->voxel_size = voxel_size; };
		
		// ICP 搜索对应点的距离阈值
		void setDownsampleSizeForICP(float voxel_size) { this->voxel_size_icp = voxel_size; };

		/** \brief set number of threads to use, only works in OpenMP.
		* \param[in] threads number of threads
		*/
		void setNumberOfThreads(int threads) { this->num_threads = threads; };

		// 设置粗匹配的方法
		void setCoarseRgistration(CoarseRegistration method) {this->method=method ; };

		void setMaximumConsensusSet(int threshold) { this->max_consensus_set = threshold; };

		/** \brief set number of lum iterations */
		void setLumIterations(int iter) { this->LUM_iterations = iter; };

		// 设置分块时每块包含的扫描数量
		void setBlockSize(int block_size) { this->part_num = block_size; };

		void visualizeEachBlock(bool check) { this->check_blcok = check; };

		void usePredefinedPairs(bool use) { this->use_pairs = use; };

		void useLumOptimization(bool use) { this->use_LUM = use; };

		void useCyclicConstraint(bool use) { this->use_cyclic_constraints = use; };

		/** \brief perform registration.*/
		void performMultiviewRegistration();

		void performShapeGrowingRegistration();

	protected:
		
		// 当前块内部的扫描对
		std::vector <std::pair <int, int> > pairs;
		// 当前块的扫描对数量
		size_t nr_pairs;
		// 当前块内在扫描数量
		size_t nr_scans;

		/** \brief  files paths */
		std::vector <boost::filesystem::path> files;

		/** \brief outout filename*/
		std::stringstream filename;

		/** \brief outout dir*/
		std::string output_dir;

		/** \brief block point clouds */
		std::vector <pcl::PointCloud <PointT>::Ptr> part_clouds;

		/** \brief block keypoint clouds */
		std::vector <pcl::PointCloud <PointT>::Ptr> part_keypoint_clouds;

		/** \brief block keypoint features */
		std::vector <pcl::PointCloud <pcl::FPFHSignature33>::Ptr> part_keypoint_feature_clouds;

		/** \brief predefined pairs dir */
		std::string pairs_dir;

		/** \brief predefined pairs path */
		std::vector <boost::filesystem::path> pairs_path;

		/** \brief visulize the block(default:0) */
		bool check_blcok;

		/** \brief visulize the block(default:0) */
		int num_threads;

		/** \brief internal scan-blocks(bigloop=0) or block-to-block registration(bigloop=1) */
		// 当前循环层数
		int bigloop;

		// 剩余的扫描数量
		int scans_left;

		// 上一轮剩余扫描数（用于判断收敛）
		int scans_left_last;

		/** \brief block size(default:5), used to build fully connected graph */
		// 分块时每块包含的扫描数量
		int part_num;

		// 分块总数
		int part_loop;

		// 最后一块的余数大小
		int part_num_rest;

		/** \brief total number of blocks(including sub-blocks) */
		int all_count;

		// 输入点云目录
		std::string PLYpath;

		/** \brief corase registration log path */
		std::string log_file;

		// 粗配准阶段下采样体素（m）（默认 0.1）
		float voxel_size;

		/** \brief Point clouds downsample size for fine registration(default:0.1m) */
		float voxel_size_icp;

		/** \brief use cyclic constraints(default:1) */
		int use_cyclic_constraints;

		/** \brief use LUM optimization(default:1) */
		int use_LUM;

		/** \brief number of LUM iterations(default:3) */
		int LUM_iterations;

		/** \brief use predefined pairs(default:0) */
		bool use_pairs;


		/** \brief trimmed ICP overlap */
		float approx_overlap;

		// 粗匹配的方法
		CoarseRegistration method;

		/** \brief estimated accuracy of the translation of the coarse alignment */
		float translation_accuracy; 

		/** \brief estimated accuracy of the rotation of the coarse alignment */
		float rotation_accuracy;

		/** \brief used to remove gross wrong matches */
		int max_consensus_set;

		/** \brief All poses */
		std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> global_poses;

		/** \brief All poses */
		std::vector<std::vector<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>>> Hierarchical_block_poses;


	private:
		/** \brief initiate varibles before registration.
		*/
		void init();


		/** \brief perform blockpartition.
		* \param[in] block_id block id
		*/
		void blockPartition(int block_id);

		/** \brief Transfer block to the temporary varibles to perform registration.
		* \param[in] keypoint_clouds input keypoint clouds
		* \param[in] keypoint_clouds_feature input keypoint features 
		* \param[in] clouds downsample clouds
		* \param[in] block_id the id of the block
		*/
		void transferVaribles(std::vector <pcl::PointCloud <PointT>::Ptr>& keypoint_clouds,
			std::vector <pcl::PointCloud<pcl::FPFHSignature33>::Ptr>& keypoint_clouds_feature, 
			std::vector <pcl::PointCloud <PointT>::Ptr>& clouds,
			int block_id
		);

		/** \brief read all paths of PLY files.
		*/
		bool readPLYfiles();

		/** \brief read predefined pairs.
		*/
		int readPairs();

		/** \brief downsample point cloud.
		* \param[in] cloud_in input cloud
		* \param[in] cloud_out output cloud
		* \param[in] downsample_size voxel size of downsample
		*/
		int sampleLeafsized(pcl::PointCloud <PointT>::Ptr& cloud_in,
			pcl::PointCloud <PointT>& cloud_out,
			float downsample_size
		);
		

		/** \brief read point clouds from directory.
		* \param[in] filename input filename
		* \param[in] cloud the container of points
		*/
		void readPointCloud(const boost::filesystem::path& filename,
			pcl::PointCloud<PointT>::Ptr cloud
		);

		/** \brief preprocessing on point clouds.
		* \param[in] keypoint_clouds input keypoint clouds
		* \param[in] keypoint_clouds_feature input keypoint features
		* \param[in] clouds downsample clouds
		* \param[in] block_id the id of the block
		*/
		void preprocessing(std::vector <pcl::PointCloud <PointT>::Ptr>& keypoint_clouds,
			std::vector <pcl::PointCloud<pcl::FPFHSignature33>::Ptr>& keypoint_clouds_feature,
			std::vector <pcl::PointCloud <PointT>::Ptr>& clouds,
			int block_id
			);

		/** \brief coarse registration for pairwise point clouds.
		* \param[in] keypoint_clouds input keypoint clouds
		* \param[in] keypoint_clouds_feature input keypoint features
		* \param[in] clouds downsample clouds
		* \param[in] pair_id the id of the pair
		*/
		void coarseRgistration(std::vector <pcl::PointCloud <PointT>::Ptr>& keypoint_clouds,
			std::vector <pcl::PointCloud<pcl::FPFHSignature33>::Ptr>& keypoint_clouds_feature,
			std::vector <pcl::PointCloud <PointT>::Ptr>& clouds,
			std::vector<int>& pairs_best_count,
			std::vector <pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>>& candidate_matches
		);


		void globalCoarseRegistration(
			std::set<int>& rejected_pairs_, 
			std::vector <pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>>& candidate_matches,
			std::vector<int> &pairs_best_count,
			int nr_scans
		);

		void globalFineRegistration(std::set<int>& rejected_pairs_,
			std::vector <pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>>& matches,
			std::vector <pcl::PointCloud <PointT>::Ptr>& clouds,
			std::vector<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>>& poses,
			int &num_of_subtrees
		);

		void mstCalculation(std::set<int>& rejected_pairs_,
			std::vector<pcl::registration::MatchingCandidate,Eigen::aligned_allocator<pcl::registration::MatchingCandidate>>& matches,
			std::vector<std::vector<int>> &LUM_indices_map_,
			std::vector<std::vector<int>> &LUM_indices_map_inv_,
			std::vector<std::vector<size_t>>& edges,
			std::vector<int>& root,
			std::vector<std::pair<int, int>>& after_check_graph_pairs,
			std::vector <pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>>& after_check_matches,
			int &num_of_subtrees
		);

		void pairwiseICP(std::vector<std::vector<size_t>>& edges,
			std::vector<std::pair<int, int>>& after_check_graph_pairs,
			std::vector <pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>>& after_check_matches,
			int &num_of_subtrees,
			std::vector <pcl::PointCloud <PointT>::Ptr> &clouds
		);

		void concatenateFineRegistration(std::vector<std::vector<size_t>>& edges,
			std::vector<int>& root,
			std::vector<std::pair<int, int>>& after_check_graph_pairs,
			std::vector <pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>>& after_check_matches,
			int &num_of_subtrees,
			std::vector<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>>& poses,
			std::vector<std::vector<pcl::Indices>> &nodes_seqs
			);

		void depthfirstsearch(int nr_scans,
			int root,
			std::vector<pcl::Indices>& Mst,
			std::vector<std::pair<int, bool>>& map_, 
			std::vector<pcl::Indices>& nodes_seq
		);

		void next(int root,
			std::vector<pcl::Indices>& Mst, 
			std::vector<std::pair<int, bool>>& map_, 
			std::vector<pcl::Indices>& nodes_seq,
			std::vector<bool> visited, 
			pcl::Indices path
		);
		void combineTransformation(int nr_scans,
			int root,
			std::vector<pcl::Indices>& nodes_seq,
			std::vector<std::pair<int, bool>>& map_,
			std::vector <pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>>& matches,
			std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>& poses
		);

		Eigen::Matrix4f inverse(Eigen::Matrix4f& mat);

		void LUMoptimization(std::vector<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>> &poses,
			std::vector<std::vector<size_t>>& edges,
			std::vector<int>& root,
			std::vector<std::vector<pcl::Indices>>& nodes_seqs,
			std::vector<std::vector<int>>& LUM_indices_map_,
			std::vector<std::vector<int>>& LUM_indices_map_inv_,
			int num_of_subtrees,
			std::vector <pcl::PointCloud <PointT>::Ptr>& clouds
		);

		void solveGlobalPose();


		void eliminateClosestPoints(pcl::PointCloud<PointT>::Ptr& src,
			pcl::PointCloud<PointT>::Ptr& tgt,
			Eigen::Matrix4f& trans,
			pcl::PointCloud<pcl::FPFHSignature33>::Ptr& fpfs,
			pcl::PointCloud<pcl::FPFHSignature33>::Ptr& fpft
		);

		double getRMSE(pcl::PointCloud<PointT>::Ptr src,pcl::PointCloud<PointT>::Ptr tgt,Eigen::Matrix4f& trans,double max);
	};
}
