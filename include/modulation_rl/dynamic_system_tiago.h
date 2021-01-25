#pragma once
#include <actionlib/client/simple_action_client.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <modulation_rl/dynamic_system_base.h>
#include <ros/topic.h>
// #include <controller_manager_msgs/SwitchController.h>
#include <geometry_msgs/Twist.h>
#include <pr2_mechanism_msgs/SwitchController.h>

typedef actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction> TrajClientTiago;
typedef boost::shared_ptr<TrajClientTiago> TiagoClientPtr;

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

class DynamicSystemTiago : public DynamicSystem_base {
  private:
    TiagoClientPtr arm_client_;
    TiagoClientPtr torso_client_;
    TiagoClientPtr gripper_client_;
    // ros::ServiceClient switch_controller_client_;
    // moveit::planning_interface::MoveGroupInterface* move_group_arm_torso_;
    control_msgs::FollowJointTrajectoryGoal arm_goal_;
    control_msgs::FollowJointTrajectoryGoal torso_goal_;

    void setup();
    geometry_msgs::Twist calc_desired_base_transform(std::vector<double> &base_actions,
                                                     tf::Vector3 planned_base_vel,
                                                     tf::Quaternion planned_base_q,
                                                     tf::Vector3 planned_gripper_vel,
                                                     tf::Transform &desiredBaseTransform,
                                                     double transition_noise_base,
                                                     double &regularization,
                                                     const double &last_dt,
                                                     const tf::Transform &desiredGripperTransform);
    void send_arm_command(const std::vector<double> &target_joint_values, double exec_duration);
    bool get_arm_success();
    // void stop_controllers();
    // void start_controllers();
  public:
    DynamicSystemTiago(uint32_t seed,
                       double min_goal_dist,
                       double max_goal_dist,
                       std::string strategy,
                       std::string real_execution,
                       bool init_controllers,
                       double penalty_scaling,
                       double time_step,
                       double slow_down_real_exec,
                       bool perform_collision_check);
    ~DynamicSystemTiago() {
        // delete move_group_arm_torso_;
    }

    void open_gripper(double position, bool wait_for_result);
    void close_gripper(double position, bool wait_for_result);
};
