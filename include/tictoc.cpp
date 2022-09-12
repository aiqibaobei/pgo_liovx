//
// Created by dlx on 2022/7/7.
//

#include "tictoc.h"

tictoc::tictoc() {}

void tictoc::tic() {
  start = high_resolution_clock::now();
}

void tictoc::toc(string operation) {
  end = high_resolution_clock::now();
  auto elapse = duration_cast<microseconds>(end - start);
  cout << setprecision(19) << operation << " cost : " << (double) elapse.count() / 1000 << " ms" << endl;
}