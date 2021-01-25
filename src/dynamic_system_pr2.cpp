#include <modulation_rl/dynamic_system_pr2.h>
//#include <tf_geometry_msgs/tf_geometry_msgs.h>

const RoboConf pr2_config{
    .name = "pr2",
    .joint_model_group_name = "right_arm",
    .frame_id = "odom_combined",
    .global_link_transform = "r_wrist_roll_link",
    .scene_collision_group_name = "",
    .tip_to_gripper_offset = tf::Vector3(0.18, 0.0, 0.0),
    .gripper_to_base_rot_offset = tf::Quaternion(0, 0, 0, 1),
    .neutral_pos_joint_names = {"r_shoulder_pan_joint", "r_shoulder_lift_joint", "r_upper_arm_roll_joint", "r_elbow_flex_joint", "r_forearm_roll_joint", "r_wrist_flex_joint", "r_wrist_roll_joint"},
    .neutral_pos_values = {0.359647, 1.22538, 0.0, -1.59997, 2.34256, -0.513323, -2.41144},
    // https://github.com/uu-isrc-robotics/uu-isrc-robotics-pr2-pkgs/blob/master/pr2_control_utilities/src/pr2_control_utilities/pr2_planning.py
    // "r_gripper_tool_joint", "r_gripper_palm_joint", "r_gripper_led_joint", "r_gripper_motor_accelerometer_joint"
    .base_cmd_topic = "/base_controller/command",
    .base_vel_rng = 0.2,
    .base_rot_rng = 1.0,
    .z_min = 0.2,
    .z_max = 1.2,
    .restricted_ws_z_min = 0.4,
    .restricted_ws_z_max = 1.0,
    .gmm_base_offset = 0.02};

DynamicSystemPR2::DynamicSystemPR2(uint32_t seed,
                                   double min_goal_dist,
                                   double max_goal_dist,
                                   std::string strategy,
                                   std::string real_execution,
                                   bool init_controllers,
                                   double penalty_scaling,
                                   double time_step,
                                   double slow_down_real_exec,
                                   bool perform_collision_check) :
    DynamicSystem_base(seed,
                       min_goal_dist,
                       max_goal_dist,
                       strategy,
                       real_execution,
                       init_controllers,
                       penalty_scaling,
                       time_step,
                       slow_down_real_exec,
                       perform_collision_check,
                       pr2_config) {
    setup();
};

void DynamicSystemPR2::setup() {
    if (init_controllers_) {
        arm_client_ = new TrajClientPR2("r_arm_controller/joint_trajectory_action", true);
        while (!arm_client_->waitForServer(ros::Duration(5.0))) {
            ROS_INFO("Waiting for the r_arm_controller/joint_trajectory_action action server to come up");
        }

        // switch_controller_client_ = nh_->serviceClient<pr2_mechanism_msgs::SwitchController>("/pr2_controller_manager/switch_controller");
        // not sure yet if want to do this for real execution only or always
        gripper_client_ = new GripperClientPR2("r_gripper_controller/gripper_action", true);
        while (!gripper_client_->waitForServer(ros::Duration(5.0))) {
            ROS_INFO("Waiting for the r_gripper_controller/gripper_action action server to come up");
        }

        arm_goal_.trajectory.points.resize(1);
        arm_goal_.trajectory.joint_names.resize(joint_names_.size());
        arm_goal_.trajectory.points[0].positions.resize(joint_names_.size());
        arm_goal_.trajectory.points[0].velocities.resize(joint_names_.size());
    }
}

void DynamicSystemPR2::send_arm_command(const std::vector<double> &target_joint_values, double exec_duration) {
    for (int i = 0; i < joint_names_.size(); i++) {
        arm_goal_.trajectory.joint_names[i] = joint_names_[i];
        arm_goal_.trajectory.points[0].positions[i] = target_joint_values[i];
        arm_goal_.trajectory.points[0].velocities[i] = 0.0;
        //        ROS_INFO("%s: %f")
    }

    // When to start the trajectory
    arm_goal_.trajectory.header.stamp = ros::Time::now() + ros::Duration(0.0);
    // To be reached x seconds after starting along the trajectory
    arm_goal_.trajectory.points[0].time_from_start = ros::Duration(exec_duration);
    // send off commands to run in parallel
    arm_client_->sendGoal(arm_goal_);
}

bool DynamicSystemPR2::get_arm_success() {
    arm_client_->waitForResult(ros::Duration(10.0));
    if (arm_client_->getState() != actionlib::SimpleClientGoalState::SUCCEEDED) {
        ROS_WARN("The arm_client_ failed.");
        // throw std::runtime_error("The arm_client_ failed.");
        return false;
    } else {
        return true;
    }
}

// http://library.isr.ist.utl.pt/docs/roswiki/pr2_controllers(2f)Tutorials(2f)Moving(20)the(20)gripper.html
void DynamicSystemPR2::move_gripper(double position, double effort, bool wait_for_result) {
    pr2_controllers_msgs::Pr2GripperCommandGoal goal;
    goal.command.position = position;
    goal.command.max_effort = effort;
    gripper_client_->sendGoal(goal);

    if (wait_for_result) {
        gripper_client_->waitForResult(ros::Duration(5.0));
        if (gripper_client_->getState() != actionlib::SimpleClientGoalState::SUCCEEDED)
            ROS_WARN("The gripper failed.");
    }
}

void DynamicSystemPR2::open_gripper(double position, bool wait_for_result) {
    move_gripper(position, -1.0, wait_for_result);  // Do not limit effort (negative)
}

void DynamicSystemPR2::close_gripper(double position, bool wait_for_result) {
    move_gripper(position, 200.0, wait_for_result);  // Close gently
}

// void DynamicSystemPR2::stop_controllers(){
//    // controllers will try to return to previous pose -> stop and restart
//    pr2_mechanism_msgs::SwitchController stop;
//    stop.request.stop_controllers.push_back("r_gripper_controller");
//    stop.request.stop_controllers.push_back("r_arm_controller");
//    if (!switch_controller_client_.call(stop)) {
//        ROS_INFO("switch_controller_client_ failed at stop");
//    };
//}
//
// void DynamicSystemPR2::start_controllers(){
//    pr2_mechanism_msgs::SwitchController start;
//    start.request.start_controllers.push_back("r_gripper_controller");
//    start.request.start_controllers.push_back("r_arm_controller");
//    if (!switch_controller_client_.call(start)) {
//        ROS_INFO("switch_controller_client_ failed at start");
//    };
//}
