#pragma once
#include <modulation_rl/dynamic_system_base.h>

#include <actionlib/client/simple_action_client.h>
#include <geometry_msgs/Twist.h>
#include <pr2_controllers_msgs/JointTrajectoryAction.h>
#include <pr2_controllers_msgs/Pr2GripperCommandAction.h>
#include <pr2_mechanism_msgs/SwitchController.h>

// Our Action interface type, provided as a typedef for convenience
typedef actionlib::SimpleActionClient<pr2_controllers_msgs::Pr2GripperCommandAction> GripperClientPR2;
typedef actionlib::SimpleActionClient<pr2_controllers_msgs::JointTrajectoryAction> TrajClientPR2;

class DynamicSystemPR2 : public DynamicSystem_base {
  private:
    TrajClientPR2 *arm_client_;
    GripperClientPR2 *gripper_client_;
    // ros::ServiceClient switch_controller_client_;
    void setup();
    pr2_controllers_msgs::JointTrajectoryGoal arm_goal_;
    void send_arm_command(const std::vector<double> &target_joint_values, double exec_duration);
    bool get_arm_success();
    // void stop_controllers();
    // void start_controllers();
    void move_gripper(double position, double effort, bool wait_for_result);

  public:
    DynamicSystemPR2(uint32_t seed,
                     double min_goal_dist,
                     double max_goal_dist,
                     std::string strategy,
                     std::string real_execution,
                     bool init_controllers,
                     double penalty_scaling,
                     double time_step,
                     double slow_down_real_exec,
                     bool perform_collision_check);

    ~DynamicSystemPR2() {
        delete gripper_client_;
        delete arm_client_;
    }

    void open_gripper(double position, bool wait_for_result);
    void close_gripper(double position, bool wait_for_result);
};
