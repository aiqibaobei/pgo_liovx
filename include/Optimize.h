//
// Created by dlx on 2022/7/27.
//

#ifndef PGO_LIVOX_SRC_INCLUDE_OPTIMIZE_H_
#define PGO_LIVOX_SRC_INCLUDE_OPTIMIZE_H_

#include <Eigen/Core>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <mutex>
#include <fstream>
#include "util.h"

using namespace gtsam;
using namespace Eigen;

class Optimizer {
 public:
  Optimizer(); //init noise
  ~Optimizer();
  void setFilePath(string &pathOrg,string &pathOpt);
  Pose3 pose2gtsamPose(MyPose& pose);
  void addFactor(int priorIndex, Pose3 &pose);
  void addFactor(int prevIndex, int curIndex, Pose3 &prevPose, Pose3 &curPose);
  void addFactor(int histIndex, int curIndex, Eigen::Matrix4f &trans);
  void saveOrgPoses(MyPose &pose);
  void saveOptPoses();

 public:
  mutable mutex mtxGraph;
  //graph and estimate value
  NonlinearFactorGraph graph;
  Values initialEstimate;
  bool isGraphMade;
  Values curEstimate;
  // isam2
  ISAM2 *isam;
  //noise def
  noiseModel::Diagonal::shared_ptr priorNoise;
  noiseModel::Diagonal::shared_ptr odomNoise;
  noiseModel::Base::shared_ptr robustLoopNoise;
  noiseModel::Base::shared_ptr robustGPSNoise;

  ofstream osOrg,osOpti;//for save poses;
};

#endif //PGO_LIVOX_SRC_INCLUDE_OPTIMIZE_H_
