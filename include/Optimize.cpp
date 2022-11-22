//
// Created by dlx on 2022/7/27.
//

#include "Optimize.h"

using namespace std;
using namespace Eigen;

Optimizer::Optimizer()
    : isGraphMade(false) {
  //isam2 init
  ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.01;
  parameters.relinearizeSkip = 1;
  isam = new ISAM2(parameters);

  // prior factor noise
  gtsam::Vector priorNoiseVector6(6);
  priorNoiseVector6 << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
  priorNoise = noiseModel::Diagonal::Variances(priorNoiseVector6);
  //odom factor noise
  gtsam::Vector odomNoiseVector6(6);
  odomNoiseVector6 << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;
  odomNoise = noiseModel::Diagonal::Variances(odomNoiseVector6);
  //loop factor noise
  gtsam::Vector loopNoiseVector(6);
  loopNoiseVector << 0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
  robustLoopNoise = gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Cauchy::Create(1),
      gtsam::noiseModel::Diagonal::Variances(loopNoiseVector));
  //rtk factor noise
  double bigNoiseTolerentToXY = 1e9;
  double gpsAltitudeNoiseScore = 250.0; // if height is misaligned after loop clsosing, use this value bigger
  gtsam::Vector robustNoiseVector3(3); // gps factor has 3 elements (xyz)
  robustNoiseVector3
      << bigNoiseTolerentToXY, bigNoiseTolerentToXY, gpsAltitudeNoiseScore; // means only caring altitude here. (because LOAM-like-methods tends to be asymptotically flyging)
  robustGPSNoise = gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
      gtsam::noiseModel::Diagonal::Variances(robustNoiseVector3));
}

Optimizer::~Optimizer() {
  osOrg.close();
  osOpti.close();
}

void Optimizer::setFilePath(string &pathOrg, string &pathOpt) {
  osOrg.open(pathOrg, ios::trunc);
  osOpti.open(pathOpt, ios::trunc);
}

Pose3 Optimizer::pose2gtsamPose(MyPose &pose) {
  Eigen::Quaterniond q = pose.q;
  Eigen::Vector3d t = pose.t;
  Rot3 R = Rot3::Quaternion(q.w(), q.x(), q.y(), q.z());
  Point3 T(t.x(), t.y(), t.z());
  return {R, T};
}

//prior factor
void Optimizer::addFactor(int priorIndex, Pose3 &pose) {
  PriorFactor<Pose3> prior_factor(0, pose, priorNoise);
  mtxGraph.lock();
  graph.add(prior_factor);
  initialEstimate.insert(0, pose);
  mtxGraph.unlock();

  cout << "[GRAPH NODE ADD] prior node added ! " << endl;
}

//odom factor
void Optimizer::addFactor(int prevIndex, int curIndex, Pose3 &prevPose, Pose3 &curPose) {
  mtxGraph.lock();
  graph.add(BetweenFactor<Pose3>(prevIndex, curIndex, prevPose.between(curPose), odomNoise));
  initialEstimate.insert(curIndex, curPose);
  mtxGraph.unlock();
}

void Optimizer::addFactor(int histIndex, int curIndex, Eigen::Matrix4f &trans) {
  Rot3 R(trans.block<3, 3>(0, 0).cast<double>());
  Point3 t(trans.block<3, 1>(0, 3).cast<double>());

  mtxGraph.lock();
  graph.add(gtsam::BetweenFactor<gtsam::Pose3>(histIndex, curIndex, Pose3(R, t), robustLoopNoise));
  mtxGraph.unlock();
  cout << "\033[32m" << "[GRAPH NODE ADD] registered succed , loop factor added btn " << curIndex << " and " << histIndex
       << "\033[37m" << endl;
}

void Optimizer::saveOrgPoses(MyPose &pose) {
  osOrg << pose.q.w() << "\t" << pose.q.x() << "\t" << pose.q.y() << "\t" << pose.q.z() << "\t"
        << pose.t.x() << "\t" << pose.t.y() << "\t" << pose.t.z() << endl;
}

//format => qw qx qy qz x y z <= format
void Optimizer::saveOptPoses() {
  for (const auto &key_value : curEstimate) {
    auto p = dynamic_cast<const GenericValue<Pose3> *>(&key_value.value);
    if (!p)
      continue;

    const Pose3 &pose = p->value();
    Point3 translation = pose.translation();
    Rot3 R = pose.rotation();
    gtsam::Quaternion q = R.toQuaternion();
    osOpti << q.w() << "\t" << q.x() << "\t" << q.y() << "\t" << q.z() << "\t"
           << translation.x() << "\t" << translation.y() << "\t" << translation.z() << endl;
  }
}

