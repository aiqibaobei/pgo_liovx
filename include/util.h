#ifndef PGO_LIVOX_SRC_INCLUDE_UTIL_H_
#define PGO_LIVOX_SRC_INCLUDE_UTIL_H_

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <mutex>

using namespace std;
using namespace Eigen;

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

struct MyPose {
  MyPose() {
    q = Quaterniond::Identity();
    t = Vector3d::Zero();
  }
  MyPose(Quaterniond _q, Vector3d _t) : q(_q), t(_t) {}
  ~MyPose() {}

  Quaterniond q;
  Vector3d t;
};

#endif

