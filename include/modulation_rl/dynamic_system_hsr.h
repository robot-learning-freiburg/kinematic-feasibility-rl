#pragma once
#include <modulation_rl/dynamic_system_base.h>

#include <tmc_robot_kinematics_model/numeric_ik_solver.hpp>
#include <tmc_robot_kinematics_model/robot_kinematics_model.hpp>
#include <tmc_robot_kinematics_model/tarp3_wrapper.hpp>

#include <actionlib/client/simple_action_client.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <control_msgs/FollowJointTrajectoryGoal.h>
#include <controller_manager_msgs/ControllerState.h>
#include <controller_manager_msgs/ListControllers.h>

typedef actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction> TrajClientHSR;

class DynamicSystemHSR : public DynamicSystem_base {
  private:
    tmc_robot_kinematics_model::IKSolver::Ptr numeric_solver_;
    tmc_manipulation_types::NameSeq ik_joint_names_;

    double dist_solution_desired_;
    double rot_dist_solution_desired_;
    double ik_slack_dist_;
    double ik_slack_rot_dist_;
    bool sol_dist_reward_;

    TrajClientHSR *arm_client_;
    TrajClientHSR *gripper_client_;
    void setup();
    bool find_ik(const Eigen::Isometry3d &desiredState, const tf::Transform &desiredGripperTfWorld);
    double calc_reward(bool found_ik, double learned_vel_norm, double regularization);

    control_msgs::FollowJointTrajectoryGoal arm_goal_;
    void send_arm_command(const std::vector<double> &target_joint_values, double exec_duration);
    bool get_arm_success();

  public:
    DynamicSystemHSR(uint32_t seed,
                     double min_goal_dist,
                     double max_goal_dist,
                     std::string strategy,
                     std::string real_execution,
                     bool init_controllers,
                     double penalty_scaling,
                     double time_step,
                     double slow_down_real_exec,
                     bool perform_collision_check,
                     double ik_slack_dist,
                     double ik_slack_rot_dist,
                     bool sol_dist_reward);

    ~DynamicSystemHSR() {
        delete gripper_client_;
        delete arm_client_;
    }

    void open_gripper(double position, bool wait_for_result);
    void close_gripper(double position, bool wait_for_result);
    void set_ik_slack(double ik_slack_dist, double ik_slack_rot_dist);
};
