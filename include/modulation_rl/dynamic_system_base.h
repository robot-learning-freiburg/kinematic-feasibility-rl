#pragma once

#include <ros/ros.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <modulation_rl/modulation.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
//#include <moveit/kinematics_base/kinematics_base.h>
#include <moveit_msgs/GetPlanningScene.h>
#include <moveit_msgs/DisplayRobotState.h>
#include <eigen_conversions/eigen_msg.h>
#include <tf/transform_listener.h>
#include <tf/tf.h>
#include "tf/transform_datatypes.h"
#include <tf_conversions/tf_eigen.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_msgs/RobotState.h>
#include "std_msgs/ColorRGBA.h"
#include "visualization_msgs/MarkerArray.h"
#include <moveit/robot_state/conversions.h>
//#include <time.h>    
//#include <chrono>
#include <rosbag/bag.h>
#include <math.h>
//#include <moveit/planning_scene_interface/planning_scene_interface.h>
//#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <gazebo_msgs/GetModelState.h>
#include <gazebo_msgs/SetModelState.h>
#include <gazebo_msgs/SetModelConfiguration.h>
#include <std_srvs/Empty.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <assert.h>

#include <modulation_rl/utils.h>
#include <modulation_rl/base_gripper_planner.h>
#include <modulation_rl/linear_planner.h>
#include <modulation_rl/gmm_planner.h>
#include <modulation_rl/modulation_ellipses.h>
#include <modulation_rl/ellipse.h>
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>

#include <boost/bind.hpp>
#include <boost/asio.hpp>
#include <boost/system/error_code.hpp>

struct pathPoint {
    double dt;
    double base_x;
    double base_y;
    double base_rot;
    double planned_base_x;
    double planned_base_y;
    double planned_base_rot;
    double desired_base_x;
    double desired_base_y;
    double desired_base_rot;
    double gripper_x;
    double gripper_y;
    double gripper_z;
    double gripper_R;
    double gripper_P;
    double gripper_Y;
    double gripper_rel_x;
    double gripper_rel_y;
    double gripper_rel_z;
    double gripper_rel_R;
    double gripper_rel_P;
    double gripper_rel_Y;
    double desired_gripper_rel_x;
    double desired_gripper_rel_y;
    double desired_gripper_rel_z;
    double desired_gripper_rel_R;
    double desired_gripper_rel_P;
    double desired_gripper_rel_Y;
    double planned_gripper_x;
    double planned_gripper_y;
    double planned_gripper_z;
    double planned_gripper_R;
    double planned_gripper_P;
    double planned_gripper_Y;
    double base_cmd_linear_x;
    double base_cmd_linear_y;
    double base_cmd_angular_z;
    bool ik_fail;
    bool collision;
};

struct RoboConf{
    const std::string name;
    const std::string joint_model_group_name;
    const std::string frame_id;
    const std::string global_link_transform;
    const std::string scene_collision_group_name;
    const tf::Vector3 tip_to_gripper_offset;
    const tf::Quaternion gripper_to_base_rot_offset;
    const std::vector<std::string> neutral_pos_joint_names;
    const std::vector<double> neutral_pos_values;
    const std::vector<std::string> eef_joint_names;
    const double base_vel_rng;
    const double base_rot_rng;
    const double z_min;
    const double z_max;
    const double restricted_ws_z_min;
    const double restricted_ws_z_max;
    const double gmm_base_offset;
};


// helper to be able to call ros::init before initialising node handle and rate
class ROSCommonNode
{
protected:
    ROSCommonNode(int argc, char **argv, const char *node_name)
    {
        ros::init(argc, argv, node_name);
    }
};


class DynamicSystem_base : ROSCommonNode
{
private:
    ros::Publisher gripper_visualizer_;
    ros::Publisher traj_visualizer_;
    ros::Publisher state_visualizer_;
    ros::Publisher fail_state_visualizer_;
    ros::Publisher gripper_plan_visualizer_;
    ros::Publisher robstate_visualizer_;
    moveit_msgs::DisplayTrajectory display_trajectory_;
    visualization_msgs::MarkerArray start_state_marker_;
    visualization_msgs::MarkerArray fail_state_marker_;
    visualization_msgs::MarkerArray gripper_plan_marker_;
    std::vector<pathPoint> pathPoints_;
    bool verbose_;

    ros::ServiceClient get_model_state_client_;
    ros::ServiceClient set_model_state_client_;
    ros::ServiceClient set_model_configuration_client_;
    ros::ServiceClient pause_gazebo_client_;
    ros::ServiceClient unpause_gazebo_client_;

    std::vector<std::string> link_names_;
    ros::AsyncSpinner* spinner_;

    // always in global reference frame
    GripperPlan next_plan_;

    int ik_error_count_ = 0;
    const int max_allowed_pause_;
    int marker_counter_ = 0;
    double reset_time_ = 0;
    double time_ = 0;
    double time_planner_ = 0;
    int paused_count_ = 0;
    double start_pause_;
    bool in_start_pause();
    double update_time(bool pause_gripper);

    double time_step_real_exec_;
    const double time_step_train_;
    const bool use_base_goal_;
    const double min_goal_dist_;
    const double max_goal_dist_;
    BaseGripperPlanner* gripper_planner_;
    // For the modulation using the ellipses 
    modulation_ellipses::Modulation modulation_;
    ros::Publisher ellipses_pub_;

    // For collision checking
    robot_state::GroupStateValidityCallbackFn constraint_callback_fn_;
    ros::ServiceClient client_get_scene_;
    //planning_scene::PlanningScenePtr planning_scene_;
    //boost::shared_ptr<planning_scene::PlanningScenePtr> planning_scene_;
    collision_detection::AllowedCollisionMatrix acm_;
    void setAllowedCollisionMatrix(planning_scene::PlanningScenePtr planning_scene, std::vector<std::string> obj_names, bool allow);
    bool check_scene_collisions();

    void set_new_random_goal(std::string gripper_goal_distribution);
    int calc_done_ret(bool found_ik, int max_allow_ik_errors);
    std_msgs::ColorRGBA get_ik_color(double alpha);
    void update_robot_start_state_marker();
    void add_fail_state_marker();
    visualization_msgs::Marker create_vel_marker(tf::Transform current_tf, tf::Vector3 vel, std::string ns, std::string color, int marker_id);
    bool set_start_pose(std::vector<double> base_start, std::string start_pose_distribution);
    void set_gripper_to_neutral();
    bool out_of_workspace(tf::Transform gripper_tf);
    void update_current_gripper_from_world();
    double draw_rng(double lower, double upper);
    void set_gazebo_model_state(std::string model_name, tf::Transform world_transform);
    void add_goal_marker_tf(tf::Transform transfm, int marker_id, std::string color);
    tf::Transform parse_goal(const std::vector<double> &gripper_goal);

protected:
    //! The node handle we'll be using
    ros::NodeHandle* nh_;
    std::vector<std::string> joint_names_;
    planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor_;
    planning_scene::PlanningScenePtr planning_scene_;
    ros::Rate rate_;
    tf::TransformListener listener_;

    random_numbers::RandomNumberGenerator rng_;

    tf::Transform currentGripperGOAL_;
    tf::Transform currentBaseGOAL_;

    std::vector<double> current_joint_values_;
    robot_state::RobotStatePtr kinematic_state_;
    robot_state::JointModelGroup* joint_model_group_;
    tf::Transform rel_gripper_pose_;
    tf::Transform currentBaseTransform_;
    tf::Transform currentGripperTransform_;
    const std::string strategy_;
    std::string real_execution_;
    const bool init_controllers_;
    const double penalty_scaling_;
    double success_thres_dist_;
    double success_thres_rot_;
    double slow_down_factor_;
    bool perform_collision_check_ ;

    // Robot specific values
    const RoboConf robo_config_;

    // dt that was used to generate the current plan
    double last_dt_ = 0;

    void add_trajectory_point(bool vis_gripper, bool found_ik);
    tf::Transform get_base_transform_world();

    bool set_pose_in_world();
    virtual void stop_controllers(){ };
    virtual void start_controllers(){ };

    ros::Publisher cmd_base_vel_pub_;  // not initialised in base class!
    virtual geometry_msgs::Twist calc_desired_base_transform(std::vector<double> &base_actions, tf::Vector3 planned_base_vel, tf::Quaternion planned_base_q, tf::Vector3 planned_gripper_vel, tf::Transform &desiredBaseTransform, double transition_noise_base, double &regularization);
    virtual bool find_ik(const Eigen::Isometry3d &desiredState, const tf::Transform &desiredGripperTfWorld);
    virtual double calc_reward(bool found_ik, bool pause_gripper, double regularization);
    virtual void send_arm_command(const std::vector<double> &target_joint_values, double exec_duration) = 0;
    virtual bool get_arm_success() = 0;
public:
    DynamicSystem_base(
        uint32_t seed,
        double min_goal_dist,
        double max_goal_dist,
        bool use_base_goal,
        std::string strategy,
        std::string real_execution,
        bool init_controllers,
        double penalty_scaling,
        double time_step,
        double slow_down_real_exec,
        bool perform_collision_check,
        RoboConf robo_config
    );
    ~DynamicSystem_base(){
        delete nh_;
        // spinner_->stop();
        delete spinner_;
        // delete task_;
        delete gripper_planner_;
   }

    std::vector<double> step(
        int max_allow_ik_errors,
        bool pause_gripper,
        std::vector<double> base_actions,
        double transition_noise_ee,
        double transition_noise_base
    );
    std::vector<double> reset(
        std::vector<double> gripper_goal,
        std::vector<double> base_start,
        std::string start_pose_distribution,
        std::string gripper_goal_distribution,
        bool do_close_gripper,
        std::string gmm_model_path,
        double success_thres_dist,
        double success_thres_rot,
        double start_pause,
        bool verbose);
    std::vector<pathPoint> visualize_robot_pose(std::string logfile);
    int get_obs_dim();
    std::vector<double> build_obs_vector(tf::Vector3 current_planned_base_vel_world, tf::Vector3 PlannedVelocities, tf::Quaternion current_planned_gripper_vel_world);
    double get_dist_to_goal();
    double get_rot_dist_to_goal();
    std::vector<double> set_gripper_goal(std::vector<double> gripper_goal, std::string gripper_goal_distribution, std::string gmm_model_path, double success_thres_dist, double success_thres_rot);
    void add_goal_marker(std::vector<double> pos, int marker_id, std::string color);
    virtual void open_gripper(double position);
    virtual void close_gripper(double position);
    void set_real_execution(std::string real_execution, double time_step, double slow_down_real_exec);
    std::string get_real_execution() { return real_execution_; };
    double get_slow_down_factor() { return slow_down_factor_; };
};


namespace validityFun
{
    bool validityCallbackFn(planning_scene::PlanningScenePtr &planning_scene,
                          // const kinematics_constraint_aware::KinematicsRequest &request,
                          // kinematics_constraint_aware::KinematicsResponse &response,
                          robot_state::RobotStatePtr kinematic_state,
                          const robot_state::JointModelGroup *joint_model_group,
                          const double *joint_group_variable_values
                          // const std::vector<double> &joint_group_variable_values
                          );

  }