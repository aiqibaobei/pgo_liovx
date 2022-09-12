//
// Created by dlx on 2022/7/27.
//
#ifndef PGO_LIVOX_SRC_INCLUDE_POINTCLOUDMATCHER_H_
#define PGO_LIVOX_SRC_INCLUDE_POINTCLOUDMATCHER_H_

#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/console/time.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <KDTreeVectorOfVectorsAdaptor.h>
#include <map>
#include "util.h"

using namespace std;
using namespace Eigen;
using namespace cv;

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;
typedef vector<vector<float>> vecOfVec;
typedef KDTreeVectorOfVectorsAdaptor<vecOfVec, float> myVecKdTree;

class PointCloudMatcher {
 public:
  PointCloudMatcher();
  double deg2rad(double deg);
  double rad2deg(double rad);
  void getTestSize();
  //pointcloud process
 void PointCloudDownSize(PointCloudXYZI &orgCloud,PointCloudXYZI &downSizedCloud);
  //lidar description
  void makeScanContext(PointCloudXYZI &cloud);
  void makeIRIS(PointCloudXYZI &cloud);
  MatrixXd makeRingKey(MatrixXd &desc);
  MatrixXd makeSecKey(MatrixXd &desc);
  vector<float> eigen2stdvec(MatrixXd &_eigmat);
  double distDirectSC(MatrixXd &_sc1, MatrixXd &_sc2);
  int getRingKeyArgminAlign(MatrixXd &ringKey1, MatrixXd &ringKey2);
  std::pair<double, int> getScoreAndShift(MatrixXd &desc1, MatrixXd &desc2);
  MatrixXd circshift(MatrixXd &_mat, int _num_shift);
  double bftMatchDesc(MatrixXd &_sc1, MatrixXd &_sc2);
  int getLoopID();
  //pub data for visualize
  void pubCurImg(ros::Publisher &pubCur);
  void pubLoopImg(int idxCur, int idxHist, ros::Publisher &pubCur, ros::Publisher &pubHist);
  void pubLoopScan(PointCloudXYZI & src,
                   PointCloudXYZI & tar,
                   ros::Publisher &pubCur,
                   ros::Publisher &pubHist);
  sensor_msgs::ImagePtr getImage(MatrixXd &desc);
  //for icp
  bool canGetTransformBetweenPCs(PointCloudXYZI &src,
                                 PointCloudXYZI &tar,
                                 Matrix4f &guess,Matrix4f &result);

 public:
  //for pc process
  pcl::VoxelGrid<PointType> downSizer;
  float DOWN_LEAF_SIZE;
  //for scanContext
  int RING_NUM;
  int SEC_NUM;
  double FOV;
  int MAX_DIS;
  int MIN_HEIGHT;
  int MAX_HEIGHT;
  int PERIOD_BUILD_TREE;
  int CANDIDATE_NUM;
  double SEARCH_RATIO;
  double LOOP_THREASHOLD;
  int NUM_RECENT_FRAMES;// recent ringkey helps little for optimize
  int MERGE_PCD_NUM;

  //for icp
  int ICP_MAX_ITER;
  double ICP_TRANS_EPS;
  int ICP_CORR_DIS;
  int ICP_USE_RANSAC;
  double ICP_EUCL_EPS;
  int ICP_TYPE;

  vector<MatrixXd> vecDesc;
  vecOfVec vecRingKey;  //save ringkey
  vecOfVec vecRingKeyToSearch;
  //tree for search ringKey
  unique_ptr<myVecKdTree> treeRingKey;
};

#endif //PGO_LIVOX_SRC_INCLUDE_POINTCLOUDMATCHER_H_
