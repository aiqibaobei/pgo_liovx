//
// Created by dlx on 2022/7/27.
//

#include "PointCloudMatcher.h"

PointCloudMatcher::PointCloudMatcher() {
    downSizer.setLeafSize(DOWN_LEAF_SIZE, DOWN_LEAF_SIZE, DOWN_LEAF_SIZE);
};

double PointCloudMatcher::deg2rad(double deg) {
    return deg * 3.1415926 / 180;
}

double PointCloudMatcher::rad2deg(double rad) {
    return rad * 180 / 3.1415926;
}

void PointCloudMatcher::getTestSize() {
    cout << "vecDesc:" << vecDesc.size() << " vecRingkey: " << vecRingKey.size() << " vecRingKeyToSearch:"
         << vecRingKeyToSearch.size() << endl;
}

void PointCloudMatcher::makeScanContext(PointCloudXYZI &cloud) {
    MatrixXd scanContext = MatrixXd::Zero(RING_NUM, SEC_NUM);
    double dis, yaw;
    int row, col;

#pragma omp parallel for num_threads(8)
    for (auto &p: cloud) {
        dis = sqrt(p.x * p.x + p.y * p.y);
        if (dis > MAX_DIS || p.z > MAX_HEIGHT || p.z < -MIN_HEIGHT)
            continue;

        yaw = rad2deg(atan2(p.x, p.y));
        yaw = yaw >= 54 ? yaw - 54 : yaw + 306;

        row = static_cast<int>(dis / (MAX_DIS / RING_NUM));
        col = static_cast<int>(yaw / (FOV / SEC_NUM));

        if (row >= 0 && row < RING_NUM && col >= 0 && col < SEC_NUM)
            scanContext(row, col) = max(scanContext(row, col), (double) (p.z + MIN_HEIGHT));
//      scanContext(row, col) = max(scanContext(row, col), (double) (p.intensity));
    }

    vecDesc.push_back(scanContext);
    MatrixXd _ringKey = makeRingKey(scanContext);
    vecRingKey.push_back(eigen2stdvec(_ringKey));
    //publish the desc img
}

void PointCloudMatcher::makeIRIS(PointCloudXYZI &cloud) {
    MatrixXi IRIS = MatrixXi::Zero(RING_NUM, SEC_NUM);
    double dis, yaw, pitch;
    int row, col, contri;
    for (auto &p: cloud) {
        dis = sqrt(p.x * p.x + p.y * p.y);
        if (dis > MAX_DIS || p.z > MAX_HEIGHT)
            continue;
        yaw = rad2deg(atan2(p.x, p.y));
        yaw = yaw >= 54 ? yaw - 54 : yaw + 306;
        pitch = rad2deg(atan2(p.z, dis));

        row = static_cast<int>(dis > MAX_DIS ? 0 : dis / (MAX_DIS / RING_NUM));
        col = static_cast<int>(yaw / (FOV / SEC_NUM));

        //-35 ～35， delt = 8.75
        contri = 1 << (unsigned int) ((pitch + FOV) / 9);
//    cout<<contri<<endl;
        if (row >= 0 && row < RING_NUM && col >= 0 && col < SEC_NUM)
            IRIS(row, col) |= contri;
    }

    MatrixXd irisDBL = IRIS.cast<double>();
    vecDesc.push_back(irisDBL);
    MatrixXd _ringKey = makeRingKey(irisDBL);
    vecRingKey.push_back(eigen2stdvec(_ringKey));
}

MatrixXd PointCloudMatcher::makeRingKey(MatrixXd &desc) {
    Eigen::MatrixXd _ringKey(desc.rows(), 1);
    for (int i = 0; i < desc.rows(); ++i) {
        Eigen::MatrixXd row = desc.row(i);
        _ringKey(i, 0) = row.mean();
    }
    return _ringKey;
}

MatrixXd PointCloudMatcher::makeSecKey(MatrixXd &desc) {
    Eigen::MatrixXd _secKey(1, desc.cols());
    for (int i = 0; i < desc.cols(); ++i) {
        Eigen::MatrixXd col = desc.col(i);
        _secKey(0, i) = col.mean();
    }
    return _secKey;
}

std::vector<float> PointCloudMatcher::eigen2stdvec(MatrixXd &_eigmat) {
    std::vector<float> vec(_eigmat.data(), _eigmat.data() + _eigmat.size());
    return vec;
} // eig2stdvec

double PointCloudMatcher::distDirectSC(MatrixXd &_sc1, MatrixXd &_sc2) {
    int num_eff_cols = 0; // i.e., to exclude all-nonzero sector
    double sum_sector_similarity = 0;
    for (int col_idx = 0; col_idx < _sc1.cols(); col_idx++) {
        VectorXd col_sc1 = _sc1.col(col_idx);
        VectorXd col_sc2 = _sc2.col(col_idx);

        if ((col_sc1.norm() == 0) || (col_sc2.norm() == 0))
            continue; // don't count this sector pair.

        double sector_similarity = col_sc1.dot(col_sc2) / (col_sc1.norm() * col_sc2.norm());

        sum_sector_similarity = sum_sector_similarity + sector_similarity;
        num_eff_cols = num_eff_cols + 1;
    }

    if (num_eff_cols == 0)
        return 1;

    double sc_sim = sum_sector_similarity / num_eff_cols;
    return 1.0 - sc_sim;
}

MatrixXd PointCloudMatcher::circshift(MatrixXd &_mat, int _num_shift) {
    // shift columns to right direction
    if (_num_shift == 0) {
        MatrixXd shifted_mat(_mat);
        return shifted_mat; // Early return
    }

    MatrixXd shifted_mat = MatrixXd::Zero(_mat.rows(), _mat.cols());
    for (int col_idx = 0; col_idx < _mat.cols(); col_idx++) {
        int new_location = (col_idx + _num_shift) % _mat.cols();
        shifted_mat.col(new_location) = _mat.col(col_idx);
    }
    return shifted_mat;
} // circshift

double PointCloudMatcher::bftMatchDesc(MatrixXd &_sc1, MatrixXd &_sc2) {
    double bestScore = 1.1f;
    for (int i = 0; i < _sc2.cols(); ++i) {
        MatrixXd shiftSc2 = circshift(_sc2, i);
        cout << distDirectSC(_sc1, shiftSc2) << " " << i << endl;
        bestScore = min(bestScore, distDirectSC(_sc1, shiftSc2));
    }
    return bestScore;
}

int PointCloudMatcher::getRingKeyArgminAlign(MatrixXd &ringKey1, MatrixXd &ringKey2) {
    int argminAlign = 0;
    auto minScore = DBL_MAX;
    for (int i = 0; i < ringKey1.rows(); ++i) {
        MatrixXd ringKey2AftShift = circshift(ringKey2, i);
        MatrixXd diff = ringKey1 - ringKey2AftShift;

        double diffNorm = diff.norm();
        if (diffNorm < minScore) {
            argminAlign = i;
            minScore = diffNorm;
        }
    }

    return argminAlign;
}

std::pair<double, int> PointCloudMatcher::getScoreAndShift(MatrixXd &desc1, MatrixXd &desc2) {
    MatrixXd ringKey1 = makeRingKey(desc1), ringKey2 = makeRingKey(desc2);
    const int searchRadius = round(SEARCH_RATIO * ringKey1.rows() / 2);
    vector<int> idxAftShift = {0};
    for (int i = 1; i < searchRadius; ++i) {
        idxAftShift.push_back((i + desc1.rows()) % desc1.rows());
        idxAftShift.push_back((-i + desc1.rows()) % desc1.rows());
    }
    sort(idxAftShift.begin(), idxAftShift.end());

    map<int, double> idxMap;

    int newArgminShift = 0;
    auto minScore = DBL_MAX;
    for (auto shift: idxAftShift) {
        MatrixXd desc2_shifted = circshift(desc2, shift);
        double curScore = distDirectSC(desc1, desc2_shifted);
        if (curScore < minScore) {
            newArgminShift = shift;
            minScore = curScore;
        }
    }

    return make_pair(minScore, newArgminShift);
}

//to find the loop frame of the cur frame
int PointCloudMatcher::getLoopID() {
    int loopid = -1;

    auto ringKeyCur = vecRingKey.back(); // find hist ringkey like cur ringkey
    auto descCur = vecDesc.back(); //for more presious match

    //too few frames
    if ((int) vecRingKey.size() < NUM_RECENT_FRAMES + 1) {
        return loopid;
    }

    //construct the tree for find loop primarily
    static int tree_period_cnt = 0;
    if (tree_period_cnt % PERIOD_BUILD_TREE == 0) {
        //rebuild the kdTree
        vecRingKeyToSearch.clear();
        vecRingKeyToSearch.assign(vecRingKey.begin(), vecRingKey.end() - NUM_RECENT_FRAMES);

        treeRingKey.reset();
        treeRingKey = make_unique<myVecKdTree>(SEC_NUM, vecRingKeyToSearch, 10); //build tree
    }

    ++tree_period_cnt;
    //knn search
    vector<size_t> idxCandidate(CANDIDATE_NUM);
    vector<float> disCandidate(CANDIDATE_NUM);

    nanoflann::KNNResultSet<float> knn_result(CANDIDATE_NUM);
    knn_result.init(&idxCandidate[0], &disCandidate[0]);
    treeRingKey->index->findNeighbors(knn_result, &ringKeyCur[0], nanoflann::SearchParams(10));

    map<int, double> idxMap;
    for (int i = 0; i < CANDIDATE_NUM; ++i) {
        MatrixXd descCandidate = vecDesc[idxCandidate[i]];
        double score = distDirectSC(descCandidate, descCur);
        if (score < LOOP_THREASHOLD) {
            idxMap.insert({idxCandidate[i], score});
        }
    }

    if (idxMap.empty()) {
        return loopid;
    }

    loopid = idxMap.begin()->first;
//  cout << "[LOOP] find loop btn " << loopid << " and " << vecRingKey.size() - 1 << "  score is "
//       << idxMap.begin()->second << endl;

    return loopid;
}

void PointCloudMatcher::pubCurImg(ros::Publisher &pubCur) {
    MatrixXd desc = vecDesc.back();
    sensor_msgs::ImagePtr msgImg = getImage(desc);
    pubCur.publish(msgImg);
}

void PointCloudMatcher::pubLoopImg(int idxCur, int idxHist, ros::Publisher &pubCur, ros::Publisher &pubHist) {
    MatrixXd descCur = vecDesc[idxCur], descHist = vecDesc[idxHist];
    sensor_msgs::ImagePtr msgImgCur = getImage(descCur), msgImgHist = getImage(descHist);
    pubCur.publish(msgImgCur);
    pubHist.publish(msgImgHist);
}

void PointCloudMatcher::pubLoopScan(PointCloudXYZI &src,
                                    PointCloudXYZI &tar,
                                    ros::Publisher &pubCur,
                                    ros::Publisher &pubHist) {
    sensor_msgs::PointCloud2 msgSrc, msgTar;
    pcl::toROSMsg(src, msgSrc);
    pcl::toROSMsg(tar, msgTar);
    msgSrc.header.frame_id = "camera_init";
    msgTar.header.frame_id = "camera_init";
    pubCur.publish(msgSrc);
    pubHist.publish(msgTar);
}

void PointCloudMatcher::PointCloudDownSize(PointCloudXYZI &orgCloud, PointCloudXYZI &downSizedCloud) {
    downSizer.setInputCloud(orgCloud.makeShared());
    downSizer.filter(downSizedCloud);
}

sensor_msgs::ImagePtr PointCloudMatcher::getImage(MatrixXd &desc) const {
    Mat img = Mat::zeros(RING_NUM, SEC_NUM, CV_8UC1);
    for (int i = 0; i < desc.rows(); ++i) {
        for (int j = 0; j < desc.cols(); ++j) {
            img.at<uchar>(i, j) = desc(i, j) * 255 / MAX_HEIGHT;
        }
    }
    //change opencv image to ros msg
    sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", img).toImageMsg();

    return img_msg;
}

bool PointCloudMatcher::canGetTransformBetweenPCs(PointCloudXYZI &src,
                                                  PointCloudXYZI &tar,
                                                  Matrix4f &guess, Matrix4f &result) const {
  auto icp = ICP_TYPE == 0 ?
             new pcl::IterativeClosestPoint<PointType, PointType> : new pcl::GeneralizedIterativeClosestPoint<PointType,
                                                                                                              PointType>;
  //set param
  icp->setInputSource(src.makeShared());
  icp->setInputTarget(tar.makeShared());
  icp->setMaximumIterations(ICP_MAX_ITER);
  icp->setTransformationEpsilon(ICP_TRANS_EPS);
  icp->setMaxCorrespondenceDistance(ICP_CORR_DIS);
  icp->setEuclideanFitnessEpsilon(ICP_EUCL_EPS);
  icp->setRANSACIterations(ICP_USE_RANSAC);

  PointCloudXYZI align;
  icp->align(align, guess);

  result = icp->getFinalTransformation();
  return icp->hasConverged() && icp->getFitnessScore() < 0.3;
}

//bool PointCloudMatcher::canGetTransformBetweenPCs(PointCloudXYZI &src,
//                                                  PointCloudXYZI &tar,
//                                                  Matrix4f &guess, Matrix4f &result) const {
//    auto ndt = new pcl::NormalDistributionsTransform<PointType, PointType>;
//    //set param
//    ndt->setInputSource(src.makeShared());
//    ndt->setInputTarget(tar.makeShared());
//    ndt->setMaximumIterations(35);
//    ndt->setTransformationEpsilon(1e-6);
//    ndt->setStepSize(1);
//    ndt->setResolution(2);
//
//    PointCloudXYZI align;
//    ndt->align(align, guess);
//
//    result = ndt->getFinalTransformation();
//    return ndt->hasConverged() && ndt->getFitnessScore() < 0.3;
//}

