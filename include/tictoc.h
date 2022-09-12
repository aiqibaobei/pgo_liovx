//
// Created by dlx on 2022/7/7.
//

#ifndef PGO_LIVOX_SRC_INCLUDE_TICTOC_H_
#define PGO_LIVOX_SRC_INCLUDE_TICTOC_H_

#include <iostream>
#include <chrono>
#include <string>
#include <iomanip>

using namespace std;
using namespace chrono;

class tictoc {
 public:
  tictoc();
  void tic();
  void toc(string operation);

  time_point<high_resolution_clock> start,end;
};

#endif //OPTIMIZE_BY_GTSAM_SRC_INCLUDE_TICTOC_H_
