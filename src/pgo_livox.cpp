#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <vector>
#include <pcl/kdtree/kdtree_flann.h>
#include <mutex>
#include <shared_mutex>
#include <thread>

//self define header
#include "tictoc.h"
#include "Optimize.h"
#include "PointCloudMatcher.h"
#include "util.h"

using namespace std;
using namespace cv;
using namespace chrono;
using namespace Eigen;
using namespace gtsam;

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

// mutex for add and modify don't interrupt each other
mutex mtxBuf; //for buffer push and pop
mutex mtxKF; //for keyfram push and pop
mutex mtxICP;

tictoc *Tictoc = new tictoc(); // for time cost calc
Optimizer *optimizer = new Optimizer(); // for optimize the poses
PointCloudMatcher *pcMatcher = new PointCloudMatcher(); //for compute the similarity of two pointclouds and try to icp

//rosparam
string FILE_PATH;
double KEYFRAME_ROT_THRESHOLD, KEYFRAME_TRANS_THRESHOLD;
bool IF_DOWN_SIZE = false;

ros::Subscriber OdomOrgSub, PointCloudBodySub, RtkSub;
ros::Publisher OdomAftOptPub, CloudAftOptPub, PathAftOptPub;
ros::Publisher LoopCurScanPub, LoopHistSubMapPub;
ros::Publisher CurDescPub;
ros::Publisher LoopCurSCImgPub, LoopHistSCImgPub;

queue<nav_msgs::Odometry::ConstPtr> odomBuf;
queue<sensor_msgs::PointCloud2::ConstPtr> pointCloudBuf;
queue<sensor_msgs::NavSatFix::ConstPtr> rtkBuf;
queue<pair<int, int>> waitForRegister;

//vector<PointCloudXYZI::Ptr> frames;
vector<PointCloudXYZI::Ptr> keyFrames, keyFramesDownSized;
vector<MyPose> keyPoses, keyPosesAftOpt;
vector<double> keyFrameTimes;
int lastUpdateIdx = 0;

pcl::VoxelGrid<PointType> downSizerDesc;
pcl::VoxelGrid<PointType> downSizerICP;
pcl::VoxelGrid<PointType> downSizerMap;

MyPose posePrev(Quaterniond::Identity(), Vector3d::Zero());
MyPose poseCur(Quaterniond::Identity(), Vector3d::Zero());
double diffRot = 0, diffTrans = 0;
ofstream trueLoop;

PointCloudXYZI::Ptr laserMap(new PointCloudXYZI());
PointCloudXYZI::Ptr histPC(new PointCloudXYZI), curPC;
PointCloudXYZI::Ptr curPC2World(new PointCloudXYZI), histPC2World(new PointCloudXYZI);

//*******************************callback******************************************
//every time messages come,save in buffer
void OriginOdomCB(const nav_msgs::Odometry::ConstPtr &odom) {
    mtxBuf.lock();
    odomBuf.push(odom);
    mtxBuf.unlock();
}

void OriginPointCloudCB(const sensor_msgs::PointCloud2::ConstPtr &cloud) {
    mtxBuf.lock();
    pointCloudBuf.push(cloud);
    mtxBuf.unlock();
}

void RtkCB(const sensor_msgs::NavSatFix::ConstPtr &rtk) {
    mtxBuf.lock();
    rtkBuf.push(rtk);
    mtxBuf.unlock();
}

//*********************************ros param**************************************
void readParam(ros::NodeHandle &n) {
    n.param<string>("path", FILE_PATH, "/home/dlx/dbg/");
    n.param<double>("keyframe_rot_threshold", KEYFRAME_ROT_THRESHOLD, 40);
    n.param<double>("keyframe_trans_threshols", KEYFRAME_TRANS_THRESHOLD, 2);
    n.param<int>("ring_num", pcMatcher->RING_NUM, 20);
    n.param<int>("sec_num", pcMatcher->SEC_NUM, 60);
    n.param<int>("max_dis", pcMatcher->MAX_DIS, 80);
    n.param<int>("min_height", pcMatcher->MIN_HEIGHT, 2);
    n.param<int>("max_height", pcMatcher->MAX_HEIGHT, 60);
    n.param<int>("period_build_tree", pcMatcher->PERIOD_BUILD_TREE, 30);
    n.param<int>("merge_pcd_num", pcMatcher->MERGE_PCD_NUM, 25);
    n.param<int>("candidate_num", pcMatcher->CANDIDATE_NUM, 3);
    n.param<int>("num_recent_frames", pcMatcher->NUM_RECENT_FRAMES, 30);
    n.param<double>("fov", pcMatcher->FOV, 70);
    n.param<double>("search_ratio", pcMatcher->SEARCH_RATIO, 0.1);
    n.param<double>("loop_threshold", pcMatcher->LOOP_THREASHOLD, 0.3);
    n.param<float>("down_leaf_size", pcMatcher->DOWN_LEAF_SIZE, 0.5);
    n.param<int>("icp_max_iter", pcMatcher->ICP_MAX_ITER, 100);
    n.param<int>("icp_corr_dis", pcMatcher->ICP_CORR_DIS, 500);
    n.param<int>("icp_use_ransac", pcMatcher->ICP_USE_RANSAC, 0);
    n.param<int>("icp_type", pcMatcher->ICP_TYPE, 0);
    n.param<double>("icp_trans_eps", pcMatcher->ICP_TRANS_EPS, 1e-6);
    n.param<double>("icp_eucl_eps", pcMatcher->ICP_EUCL_EPS, 1e-6);
    n.param<bool>("if_down_size", IF_DOWN_SIZE, false);
}

void rosPubAndSubInit(ros::NodeHandle &n) {
    //pose and pointcloud from fastlio
    OdomOrgSub = n.subscribe<nav_msgs::Odometry>("/Odometry", 10000, OriginOdomCB);
    PointCloudBodySub = n.subscribe<sensor_msgs::PointCloud2>("/cloud_registered_body", 10000, OriginPointCloudCB);
    RtkSub = n.subscribe<sensor_msgs::NavSatFix>("/rtk_data", 10000, RtkCB);
    //aft optimize
    OdomAftOptPub = n.advertise<nav_msgs::Odometry>("/Odometry_aft_optimize", 10000);
    CloudAftOptPub = n.advertise<sensor_msgs::PointCloud2>("/cloud_aft_optimize", 10000);
    PathAftOptPub = n.advertise<nav_msgs::Path>("/path_aft_optimize", 10000);
    //show loop and submap location in the whole map
    LoopCurScanPub = n.advertise<sensor_msgs::PointCloud2>("/loop_cur_scan", 10000);
    LoopHistSubMapPub = n.advertise<sensor_msgs::PointCloud2>("/loop_hist_submap", 10000);
    //show the loop img to visualize
    CurDescPub = n.advertise<sensor_msgs::Image>("/cur_desc_img", 10000);
    LoopCurSCImgPub = n.advertise<sensor_msgs::Image>("/loop_cur_img", 10000);
    LoopHistSCImgPub = n.advertise<sensor_msgs::Image>("/loop_hist_img", 10000);
}

//*********************************utils********************************************
MyPose getPoseFromOdom(nav_msgs::Odometry::ConstPtr &odom) {
    Quaterniond q(odom->pose.pose.orientation.w, odom->pose.pose.orientation.x,
                  odom->pose.pose.orientation.y, odom->pose.pose.orientation.z);
    Vector3d t(odom->pose.pose.position.x, odom->pose.pose.position.y, odom->pose.pose.position.z);
    return {q, t};
}

bool isKeyFrame(MyPose &prev, MyPose &cur, double &diffOfRot, double &diffOfTrans) {
    Isometry3d prevT = Isometry3d::Identity(), curT = Isometry3d::Identity();
    prevT.prerotate(prev.q), prevT.pretranslate(prev.t);
    curT.prerotate(cur.q), curT.pretranslate(cur.t);
    Isometry3d deltaT = prevT.inverse() * curT;

    Eigen::Affine3d SE3_delta;
    SE3_delta.matrix() = deltaT.matrix();
    double dx, dy, dz, droll, dpitch, dyaw;
    pcl::getTranslationAndEulerAngles(SE3_delta, dx, dy, dz, droll, dpitch, dyaw);

    diffOfRot += (fabs(droll) + fabs(dpitch) + fabs(dyaw));
    diffOfTrans += sqrt(dx * dx + dy * dy + dz * dz);
    return diffOfRot > KEYFRAME_ROT_THRESHOLD || diffOfTrans > KEYFRAME_TRANS_THRESHOLD;
}

//* point cloud from cur coordinate transform to base coordinate
void cur2base(PointCloudXYZI &curCloud, PointCloudXYZI &baseCloud, const MyPose &curPose, const MyPose &basePose) {
// w-世界坐标系, base-目标坐标系，cur-当前坐标系
// 考虑一个同一个点，在世界坐标系的表示是一致的，有p_w = q_base * p_base + t_base ,且 p_w = q_cur * p_cur +  t_cur
// 因此有 为了得到cur->base的变换 ，有 p_base = q_base.inverse() * ( p_w - t_base )
//                                        = q_base.inversse() * ( q_cur *p_cur + t_cur - t_base )
    for (auto &p: curCloud) {
        Vector3d point(p.x, p.y, p.z);
        Vector3d pointBase(basePose.q.inverse() * (curPose.q * point + curPose.t - basePose.t));
        baseCloud.push_back(PointType(pointBase(0), pointBase(1), pointBase(2), p.intensity));
    }
}

// merge pointcloud from index-num to index+num
void mergeCloud(const int index, const int num) {
    PointCloudXYZI tmpCloud;
    int start = index - num >= 0 ? index - num : 0;
    int end = index + num < (int) keyFramesDownSized.size() ? index + num : (int) keyFramesDownSized.size() - 1;
    mtxKF.lock();
    for (int i = start; i <= end; ++i) {
        PointCloudXYZI baseCloud;
        cur2base(*keyFramesDownSized[i], baseCloud, keyPoses[i], keyPoses[index]);
        tmpCloud += baseCloud;
    }
    mtxKF.unlock();

    downSizerICP.setInputCloud(tmpCloud.makeShared());
    downSizerICP.filter(*histPC);
}

//*********************************update poses**************************************
//when after optimize,update the optimized poses
void updatePoses() {
    lastUpdateIdx = optimizer->curEstimate.size();

    mtxKF.lock();
//#pragma omp parallel for num_threads(8)
    for (int node_idx = 0; node_idx < lastUpdateIdx; node_idx++) {
        MyPose &p = keyPosesAftOpt[node_idx];

        gtsam::Values curEsti = optimizer->curEstimate;
        p.q = curEsti.at<gtsam::Pose3>(node_idx).rotation().toQuaternion();
        p.t = curEsti.at<gtsam::Pose3>(node_idx).translation();
    }
    mtxKF.unlock();
} // updatePoses

void runISAM2opt() {
    // called when a variable added
    optimizer->isam->update(optimizer->graph, optimizer->initialEstimate);
    optimizer->isam->update();

    optimizer->graph.resize(0);
    optimizer->initialEstimate.clear();

    optimizer->curEstimate = optimizer->isam->calculateEstimate();
    updatePoses();
}

//************************************threads****************************************
//choose keyFrame and add odom factor to graph
bool isFirstFrame = true;
static int m_cnt;
ofstream odom_os("/home/h305/scTest/all_pcd/odom.txt", ios::trunc);

void thBuildGraph() {
    while (true) {
        while (!odomBuf.empty() && !pointCloudBuf.empty()) {
            mtxBuf.lock();
            while (!odomBuf.empty() &&
                   odomBuf.front()->header.stamp.toSec() < pointCloudBuf.front()->header.stamp.toSec()) {
                //data in front of odomBuf is invalid
                odomBuf.pop();
            }
            if (odomBuf.empty()) {
                mtxBuf.unlock();
                break;
            }

            //Tictoc->tic();
            double timeStamp = odomBuf.front()->header.stamp.toSec();
            //get cur odom and pointcloud
            MyPose poseOdom = getPoseFromOdom(odomBuf.front());
            odomBuf.pop();

            PointCloudXYZI::Ptr thisKeyFrame(new PointCloudXYZI);
            pcl::fromROSMsg(*pointCloudBuf.front(), *thisKeyFrame);
            pointCloudBuf.pop();
            mtxBuf.unlock();

            pcl::io::savePCDFile("/home/h305/scTest/all_pcd/" + to_string(m_cnt) + ".pcd", *thisKeyFrame);
            ++m_cnt;
            odom_os << poseCur.q.coeffs()[0] << "\t" << poseCur.q.coeffs()[1] << "\t" << poseCur.q.coeffs()[2] << "\t"
                    << poseCur.q.coeffs()[3] << "\t"
                    << poseCur.t(0) << "\t" << poseCur.t(1) << "\t" << poseCur.t(2) << "\t" << endl;

            posePrev = poseCur, poseCur = poseOdom;
            bool notKeyFrame = !isKeyFrame(posePrev, poseCur, diffRot, diffTrans);
            if (notKeyFrame && !isFirstFrame)
                continue;

            isFirstFrame = false;
            diffRot = 0, diffTrans = 0;//get keyFrame,reset the accumulated diff
            optimizer->saveOrgPoses(poseCur);

            PointCloudXYZI::Ptr thisKeyFrameDS(new PointCloudXYZI);
//      pcMatcher->PointCloudDownSize(*thisKeyFrame, *thisKeyFrameDS);
            downSizerDesc.setInputCloud(thisKeyFrame);
            downSizerDesc.filter(*thisKeyFrameDS);

            mtxKF.lock();
            {
                keyFrames.push_back(thisKeyFrame);
                keyFramesDownSized.push_back(thisKeyFrameDS);
                keyPoses.push_back(poseCur);
                keyPosesAftOpt.push_back(poseCur);
                keyFrameTimes.push_back(timeStamp);
            }
            mtxKF.unlock();

            pcMatcher->makeScanContext(*thisKeyFrameDS);
            //        pcMatcher->makeIRIS(thisKeyFrameDS);
            pcMatcher->pubCurImg(CurDescPub);

            //first frame is treat as keyFrame
            if (!optimizer->isGraphMade) {
                Pose3 posePrior = optimizer->pose2gtsamPose(poseCur);
                optimizer->addFactor(0, posePrior);
                optimizer->isGraphMade = true;
            } else {
                const int prev_node_idx = keyPoses.size() - 2, cur_node_idx = keyPoses.size() - 1;
                //add odomFactor
                Pose3 posePrevGtsam = optimizer->pose2gtsamPose(keyPoses[prev_node_idx]);
                Pose3 posecurGtsam = optimizer->pose2gtsamPose(keyPoses[cur_node_idx]);
                optimizer->addFactor(prev_node_idx, cur_node_idx, posePrevGtsam, posecurGtsam);
            }
            //Tictoc->toc("keyframe");
        }
//    //avoid empty odom and pointcloud buf every cycle
        this_thread::sleep_for(chrono::microseconds(2000));
    }
}

//get the loop id ,and save them in the icpBuf
void thFindLoop() {
    ros::Rate rate(1.0f);
    while (ros::ok()) {
        rate.sleep();
        if ((int) keyPoses.size() < pcMatcher->NUM_RECENT_FRAMES) {
            continue;
        }

        //Tictoc->tic();
        int loopId = pcMatcher->getLoopID();
        if (loopId != -1) {
            int idxHist = loopId;
            int idxCur = keyPoses.size() - 1;

//            Vector3d posHist = keyPoses[idxHist].t;
//            Vector3d posCur = keyPoses[idxCur].t;
//            double disbtnHistAndCur = (posCur - posHist).norm();
//
//            if (disbtnHistAndCur > 20) {
//                cout << "find loop but distance too long ! reject th loop ! dis is " << disbtnHistAndCur << endl;
//                continue;
//            }

            pcMatcher->pubLoopImg(idxCur, idxHist, LoopCurSCImgPub, LoopHistSCImgPub);
            cout << "find loop btn " << idxHist << " and " << idxCur << endl;

            mtxICP.lock();
            waitForRegister.push(make_pair(idxHist, idxCur));
            mtxICP.unlock();
        }
        //Tictoc->toc("find loop");
    }
}

//take idx from icpBuf,and verify whether it is loop
void thPCRegister() {
    while (true) {
        while (!waitForRegister.empty()) {
            if (waitForRegister.size() > 30) {
                cout << "[ICP] too many icp wait for icp ! slow down the loop detection !";
            }
            //Tictoc->tic();
            mtxICP.lock();
            pair<int, int> loop_pair = waitForRegister.front();
            waitForRegister.pop();
            mtxICP.unlock();

            histPC->clear();
            int histIdx = loop_pair.first, curIdx = loop_pair.second;
            mergeCloud(histIdx, pcMatcher->MERGE_PCD_NUM);

            curPC = keyFramesDownSized[curIdx];
            MyPose &histPose = keyPoses[histIdx], &curPose = keyPoses[curIdx];

            Isometry3d histT = Isometry3d::Identity(), curT = Isometry3d::Identity();
            histT.prerotate(histPose.q), histT.pretranslate(histPose.t);
            curT.prerotate(curPose.q), curT.pretranslate(curPose.t);
            Matrix4f guess = (histT.inverse() * curT).matrix().cast<float>();
            Matrix4f result;

            bool registeredRes = pcMatcher->canGetTransformBetweenPCs(*curPC, *histPC, guess, result);
            if (registeredRes) {
                trueLoop << curIdx << "\t" << histIdx << endl;
//        cout << "[ICP SUCCEED] between " << curIdx << " and " << histIdx << endl;
                optimizer->addFactor(histIdx, curIdx, result);

                //pub loop scan
                curPC2World->clear(), histPC2World->clear();
                cur2base(*curPC, *curPC2World, curPose, MyPose());
                cur2base(*histPC, *histPC2World, histPose, MyPose());
                pcMatcher->pubLoopScan(*curPC2World, *histPC2World, LoopCurScanPub, LoopHistSubMapPub);
            } else {
//        cout << "[ICP FAILED] between " << curIdx << " and " << histIdx << endl;
            }
            //Tictoc->toc("icp");
        }
        this_thread::sleep_for(chrono::milliseconds(2));
    }
}

void thIsam() {
    ros::Rate rate(1);
    while (ros::ok()) {
        rate.sleep();
        if (optimizer->isGraphMade) {
            //Tictoc->tic();
            optimizer->mtxGraph.lock();
            runISAM2opt();
            optimizer->mtxGraph.unlock();
            //Tictoc->toc("isam");
        }
    }
}

void thPubMapAftOPt(void) {
    ros::Rate rate(0.1);
    while (ros::ok()) {
        rate.sleep();
        if (lastUpdateIdx == 0)
            continue;

        //Tictoc->tic();
        laserMap->clear();
        mtxKF.lock();
        for (int i = 0; i < lastUpdateIdx; ++i) {
            PointCloudXYZI::Ptr curCloud = keyFrames[i];
            MyPose &curPose = keyPosesAftOpt[i];

            PointCloudXYZI::Ptr baseCloud(new PointCloudXYZI());
            cur2base(*curCloud, *baseCloud, curPose, MyPose());
            *laserMap += *baseCloud;
        }
        mtxKF.unlock();

        if (IF_DOWN_SIZE) {
            PointCloudXYZI::Ptr laserMapDS(new PointCloudXYZI);
            downSizerMap.setInputCloud(laserMap);
            downSizerMap.filter(*laserMapDS);
            *laserMap = *laserMapDS;
            if (laserMapDS->empty())
                continue;
        }

        sensor_msgs::PointCloud2 laserMapRos;
        pcl::toROSMsg(*laserMap, laserMapRos);
        laserMapRos.header.frame_id = "camera_init";
        CloudAftOptPub.publish(laserMapRos);
        //Tictoc->toc("pubmap");
    }
}

void thPubPathAftOpt() {
    ros::Rate rate(10.0f);
    while (ros::ok()) {
        rate.sleep();
        if (lastUpdateIdx == 0)
            continue;

//    Tictoc->tic();
        nav_msgs::Odometry odomAftPGO;
        nav_msgs::Path pathAftPGO;
        pathAftPGO.header.frame_id = "camera_init";

        mtxKF.lock();
        for (int node_idx = 0; node_idx < lastUpdateIdx; node_idx++) {
            const MyPose &pose_est = keyPosesAftOpt.at(node_idx); // upodated poses
            nav_msgs::Odometry odomAftPGOthis;

            odomAftPGOthis.header.frame_id = "camera_init";
            odomAftPGOthis.child_frame_id = "/aft_pgo";
            odomAftPGOthis.header.stamp = ros::Time().fromSec(keyFrameTimes.at(node_idx));
            odomAftPGOthis.pose.pose.orientation.w = pose_est.q.w();
            odomAftPGOthis.pose.pose.orientation.x = pose_est.q.x();
            odomAftPGOthis.pose.pose.orientation.y = pose_est.q.y();
            odomAftPGOthis.pose.pose.orientation.z = pose_est.q.z();
            odomAftPGOthis.pose.pose.position.x = pose_est.t.x();
            odomAftPGOthis.pose.pose.position.y = pose_est.t.y();
            odomAftPGOthis.pose.pose.position.z = pose_est.t.z();
            odomAftPGO = odomAftPGOthis;

            geometry_msgs::PoseStamped poseStampAftPGO;
            poseStampAftPGO.header = odomAftPGOthis.header;
            poseStampAftPGO.pose = odomAftPGOthis.pose.pose;

            pathAftPGO.header.stamp = odomAftPGOthis.header.stamp;
            pathAftPGO.header.frame_id = "camera_init";
            pathAftPGO.poses.push_back(poseStampAftPGO);
        }// poses
        mtxKF.unlock();

        OdomAftOptPub.publish(odomAftPGO);
        PathAftOptPub.publish(pathAftPGO);
//    Tictoc->toc("pubpath");
    }
}

//*********************************main******************************************
int main(int argc, char **argv) {
    //init and read parameters
    ros::init(argc, argv, "pgo_livox");
    ros::NodeHandle n;
    readParam(n);
    rosPubAndSubInit(n);

    //debug files
    string pathOrg = FILE_PATH + "poseOrg.txt", pathOpt = FILE_PATH + "poseOpt.txt";
    optimizer->setFilePath(pathOrg, pathOpt);
    trueLoop.open(FILE_PATH + "trueLoop.txt ", ios::trunc);

    keyFrames.reserve(5000), keyFramesDownSized.reserve(5000);
    keyPoses.reserve(5000), keyPosesAftOpt.reserve(5000);
    keyFrameTimes.reserve(5000);

    downSizerDesc.setLeafSize(0.5, 0.5, 0.5);
    downSizerICP.setLeafSize(0.5, 0.5, 0.5);
    downSizerMap.setLeafSize(0.5, 0.5, 0.5);

    //threads
    thread buildGraph{thBuildGraph};
    thread findLoop{thFindLoop};
    thread PCRegister{thPCRegister};
    thread isam{thIsam};
    thread PubMapAftOPt{thPubMapAftOPt};
    thread PubPathAftOpt{thPubPathAftOpt};
    ros::spin();

    //finally save all poses
    optimizer->saveOptPoses();
    //save laser pcd
    pcl::PCDWriter writer;
    if (writer.writeBinary(FILE_PATH + "cloudOpt.pcd", *laserMap))
        cout << "LaserMap saved successful ! " << endl;

    trueLoop.close();
    return 0;
}