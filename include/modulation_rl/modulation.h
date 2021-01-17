//#ifndef MODULATION
//#define MODULATION

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <ros/ros.h>
//#include <fstream>
//#include <sstream>




namespace modulation {



std::vector<std::vector<double> > computeEBase(std::vector<double> normal) ; 
Eigen::MatrixXf assembleD_k(double lambda1, double lambda2);

Eigen::MatrixXf assembleE_k(double alpha);

void compModulation(double alpha, double lambda1, double lambda2, Eigen::Vector2f& curr_speed);

}

//#endif
