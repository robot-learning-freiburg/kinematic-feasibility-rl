
#ifndef GAUSSIAN_MIXTURE_MODEL_H
#define GAUSSIAN_MIXTURE_MODEL_H

#include <ros/ros.h>
#include <ros/assert.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <tf/transform_listener.h>
#include <tf/tf.h>
#include "tf/transform_datatypes.h"
#include <tf_conversions/tf_eigen.h>
#include <eigen_conversions/eigen_msg.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
//#include <geometry_msgs/Pose.h>
//#include <geometry_msgs/PoseArray.h>
//#include <visualization_msgs/MarkerArray.h>
//#include <dynamic_modulation/modulation.h>
//#include <ellipse/ellipse.h>
// Custom messages for GMM
//#include <pr2_arm_base_control/array1d.h>
//#include <pr2_arm_base_control/array2d.h>
//#include <pr2_arm_base_control/GMM.h>
//#include <pr2_arm_base_control/Mu.h>
//#include <pr2_arm_base_control/Priors.h>
//#include <pr2_arm_base_control/Sigma.h>
#include <boost/shared_ptr.hpp>
//#include <moveit/robot_model_loader/robot_model_loader.h>
//#include <moveit/robot_model/robot_model.h>
//#include <moveit/robot_state/robot_state.h>
//#include <moveit/robot_state/conversions.h>
//#include <moveit/planning_scene/planning_scene.h>
//#include <moveit_msgs/GetPlanningScene.h>
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>

#include <boost/bind.hpp>
#include <boost/asio.hpp>
#include <boost/system/error_code.hpp>

#include <modulation_rl/utils.h>




class GaussianMixtureModel
{
  public:
  	// GaussianMixtureModel();
  	~GaussianMixtureModel();
  	GaussianMixtureModel(
  	    double max_speed_gripper,
        double max_speed_base,
        double max_speed_gripper_rot,
        double max_speed_base_rot
    );
 
  	void adaptModel(tf::Transform newGoal, tf::Vector3 gmm_base_offset);
  	bool loadFromFile(std::string & filename);
  	void integrateModel(double current_time, double dt,Eigen::VectorXf* current_pose, Eigen::VectorXf* current_speed, bool do_update);

  	int getNr_modes() const { return _nr_modes; };
  	std::string getType() const { return _type;};
  	void setType(std::string type) {_type = type;};
	double getkP() const { return _kP;};
	double getkV() const { return _kV;};
	std::vector<double> getPriors() const { return _Priors;};
	std::vector<Eigen::VectorXf> getMu() const { return _MuEigen;};
	std::vector<Eigen::MatrixXf> getSigma() const { return _Sigma;};
	tf::StampedTransform getGoalState() const { return _goalState;};
	tf::Transform getStartState() const { return _startState;};
    tf::Transform getGraspPose() const { return _related_object_grasp_pose;};
	std::string getObjectName() const { return _related_object_name;};

    double gmm_time_offset_;

  protected:
	int _nr_modes;
	std::string _type;
	double _kP;
	double _kV;
	double _motion_duration;
	double _max_speed_gripper;
	double _max_speed_base;
    double _max_speed_gripper_rot;
    double _max_speed_base_rot;
	std::vector<double> _Priors;
	std::vector<std::vector<double> > _Mu;
	std::vector<Eigen::VectorXf> _MuEigen;
	std::vector<Eigen::VectorXf> _MuEigenBck;
	std::vector<Eigen::MatrixXf> _Sigma;
	std::vector<Eigen::MatrixXf> _SigmaBck;
	tf::StampedTransform _goalState;
	tf::Transform _startState;
	tf::Transform _related_object_pose;
	tf::Transform _related_object_grasp_pose;
	std::string _related_object_name;
	std::vector<int> _colliding_poses;
	int _colliding_base_angle;
	int _ik_error_counter;
	int _ik_collision_counter;
	int _total_nr_poses;
	int _current_nr_poses;
	int _nr_pose_last_plotted;
	int _plot_every_xth;
	//geometry_msgs::PoseArray trajectory_pose_array;

	ros::NodeHandle nh_;
	//ros::Publisher Mu_pub_;
	//ros::Publisher Traj_pub_;
	//ros::Publisher Traj_pub2_;
	//ros::Publisher Ellipses_pub_;
	//ros::Publisher planning_scene_diff_publisher_;
	//ros::ServiceClient client_get_scene_;
	// boost::shared_ptr<modulation::Modulation_manager> manager_;
	//modulation::Modulation modulation_;

	// IK test stuff
	//boost::shared_ptr<robot_state::RobotState> kinematic_state;					  
	//robot_state::JointModelGroup* joint_model_group;
	//boost::shared_ptr<planning_scene::PlanningScene> planning_scene_;
	//collision_detection::AllowedCollisionMatrix currentACM_;

	template <typename T>
	bool parseVector(std::ifstream & is, std::vector<T> & pts, const std::string & name);
	double gaussPDF(double& current_time,int mode_nr);
	void plotEllipses(Eigen::VectorXf& curr_pose, Eigen::VectorXf& curr_speed,double dt);
	void clearMarkers(int nrPoints);
	template <typename T1, typename T2>
	T1 extract(const T2& full, const T1& ind);

};



#endif // GAUSSIAN_MIXTURE_MODEL_H
