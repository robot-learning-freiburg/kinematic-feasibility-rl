#include <modulation_rl/modulation.h>


using namespace std;
namespace modulation {


  //Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");


  Eigen::MatrixXf assembleD_k(double lambda1,double lambda2) {
    Eigen::MatrixXf d_k(2,2);
    d_k.setIdentity();
    for(int i = 0; i < 2; i++) {
        if (i == 1) {
            //d_k(i, i) = 1.0 - 1.0/norm(Gamma);
            d_k(i, i) = lambda1;
        }
        else{
            d_k(i, i) = lambda2 ;//+ 1.0/norm(Gamma);
        }
      
    }
    return d_k;
  }


  std::vector<std::vector<double> > computeEBase(std::vector<double> normal) {
    int d = 2;
    std::vector<std::vector<double> > base = {{0, 0}};
    for (int i = 1; i <= d - 1; i++) {
      for (int j = 1; j <= d; j++) {
        if (j == 1) {
          base[i-1][j-1] = -normal[i - 1];
        } else if (j == i && i != 1) {
          base[i-1][j-1] = normal[0];
        } else {
          base[i-1][j-1] = 0;
        }
      }
    }
    return base;
  };



  Eigen::MatrixXf assembleE_k(double alpha) {
    Eigen::MatrixXf e_k(2,2);
    std::vector<double> norm = {cos(alpha), sin(alpha)};
    std::vector<std::vector<double> > base = computeEBase(norm);
    for (int i = 0; i < 2; i++) {
      e_k(i, 0) = norm[i];
      if(i==0)
        e_k(i, 1) = norm[1];
      else
        e_k(i, 1) = -norm[0];
    }
    return e_k;
  }

  void compModulation(double alpha, double lambda1, double lambda2, Eigen::Vector2f& curr_speed) {
    Eigen::Matrix2f modulation;
    modulation.setIdentity();
    Eigen::MatrixXf d_k = assembleD_k(lambda1,lambda2);
    Eigen::MatrixXf e_k = assembleE_k(alpha);
    modulation = e_k * d_k * e_k.inverse();
    curr_speed = modulation * curr_speed;
  }
}

