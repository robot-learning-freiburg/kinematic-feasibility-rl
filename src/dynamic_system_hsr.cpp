#include <modulation_rl/dynamic_system_hsr.h>
using Eigen::Affine3d;
using Eigen::AngleAxisd;
using Eigen::Translation3d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::string;
using std::vector;
using tmc_manipulation_types::JointState;
using tmc_robot_kinematics_model::IKRequest;
using tmc_robot_kinematics_model::IKSolver;
using tmc_robot_kinematics_model::IRobotKinematicsModel;
using tmc_robot_kinematics_model::NumericIKSolver;
using tmc_robot_kinematics_model::Tarp3Wrapper;

namespace {
    const uint32_t kMaxItr = 10000;
    const double kEpsilon = 0.0001;
    const double kConvergeThreshold = 1e-6;
    const char *const kModelPath = "/opt/ros/melodic/share/hsrb_description/robots/hsrb4s.urdf";
}  // namespace

const RoboConf hsr_config{.name = "hsrb",
                          .joint_model_group_name = "arm",
                          .frame_id = "odom",
                          .global_link_transform = "hand_palm_link",
                          .scene_collision_group_name = "",
                          // set offset to zero to account for less precise final position we're asking for
                          // tf::Vector3(0.075, 0.0, 0.0),
                          .tip_to_gripper_offset = tf::Vector3(0.08, 0.0, 0.0),
                          .gripper_to_base_rot_offset = tf::Quaternion(0.707, 0.000, 0.707, -0.000),
                          // from src/hsrb_moveit/hsrb_moveit_config/config/hsrb.srdf
                          .neutral_pos_joint_names = {"arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"},
                          .neutral_pos_values = {0.2, -0.7, 0.0, -1.2, 0.0},
                          // not double checked yet
                          .base_cmd_topic = "/hsrb/command_velocity",
                          .base_vel_rng = 0.2,
                          .base_rot_rng = 1.5,
                          .z_min = 0.2,
                          .z_max = 1.4,
                          .restricted_ws_z_min = 0.4,
                          .restricted_ws_z_max = 1.1,
                          .gmm_base_offset = 0.25};

DynamicSystemHSR::DynamicSystemHSR(uint32_t seed,
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
                                   bool sol_dist_reward) :
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
                       hsr_config),
    ik_slack_dist_{ik_slack_dist},
    ik_slack_rot_dist_{ik_slack_rot_dist},
    sol_dist_reward_{sol_dist_reward} {
    setup();
}

void DynamicSystemHSR::setup() {
    // analytic solver don't need robot model.
    IRobotKinematicsModel::Ptr robot;

    // load robot model.
    std::string xml_string;

    std::fstream xml_file(kModelPath, std::fstream::in);
    while (xml_file.good()) {
        std::string line;
        std::getline(xml_file, line);
        xml_string += (line + "\n");
    }
    xml_file.close();

    robot.reset(new Tarp3Wrapper(xml_string));
    // https://git.hsr.io/koji_terada/example_hsr_ik/-/blob/master/src/example_numeric_ik.cpp
    numeric_solver_.reset(new NumericIKSolver(IKSolver::Ptr(),
                                              robot,
                                              10000,   //::kMaxItr,
                                              0.001,   //::kEpsilon,
                                              1e-6));  //::kConvergeThreshold));

    // ik joints. analytic ik have to use these joint.
    ik_joint_names_.push_back("arm_lift_joint");
    ik_joint_names_.push_back("arm_flex_joint");
    ik_joint_names_.push_back("arm_roll_joint");
    ik_joint_names_.push_back("wrist_flex_joint");
    ik_joint_names_.push_back("wrist_roll_joint");
    ik_joint_names_.push_back("wrist_ft_sensor_frame_joint");

    if (init_controllers_) {
        std::vector<string> controllers_to_await;

        arm_client_ = new TrajClientHSR("/hsrb/arm_trajectory_controller/follow_joint_trajectory", true);
        controllers_to_await.push_back("arm_trajectory_controller");

        while (!arm_client_->waitForServer(ros::Duration(5.0))) {
            ROS_INFO("Waiting for the /hsrb/arm_trajectory_controller/follow_joint_trajectory action server to come up");
        }

        // do not check if running, see https://qa.hsr.io/en/question/2164/gripper_controller-not-present-in-simulation/
        gripper_client_ = new TrajClientHSR("/hsrb/gripper_controller/follow_joint_trajectory", true);
        // controllers_to_await.push_back("gripper_controller");
        // while(!gripper_client_->waitForServer(ros::Duration(5.0))){
        //     ROS_INFO("Waiting for the /hsrb/gripper_controller/follow_joint_trajectory action server to come up");
        // }

        arm_goal_.trajectory.points.resize(1);
        arm_goal_.trajectory.joint_names.resize(joint_names_.size() - 1);
        arm_goal_.trajectory.points[0].positions.resize(joint_names_.size() - 1);
        arm_goal_.trajectory.points[0].velocities.resize(joint_names_.size() - 1);

        // make sure the controller is running
        ros::ServiceClient controller_manager_client = nh_->serviceClient<controller_manager_msgs::ListControllers>("/hsrb/controller_manager/list_controllers");
        controller_manager_msgs::ListControllers list_controllers;

        while (!controller_manager_client.call(list_controllers)) {
            ROS_INFO("Waiting for /hsrb/controller_manager/list_controllers");
            ros::Duration(0.5).sleep();
        }

        std::string cname;
        for (int j = 0; j < controllers_to_await.size(); j++) {
            cname = controllers_to_await.back();
            controllers_to_await.pop_back();
            bool running = false;
            while (!running) {
                ROS_INFO_STREAM("Waiting for /hsrb/" << cname);
                ros::Duration(0.5).sleep();
                if (controller_manager_client.call(list_controllers)) {
                    for (unsigned int i = 0; i < list_controllers.response.controller.size(); i++) {
                        controller_manager_msgs::ControllerState c = list_controllers.response.controller[i];
                        // TODO: not checking for all controllers I'm starting here
                        if (c.name == cname && c.state == "running") {
                            running = true;
                        }
                    }
                }
            }
        }
    }
}

void DynamicSystemHSR::set_ik_slack(double ik_slack_dist, double ik_slack_rot_dist) {
    ik_slack_dist_ = ik_slack_dist;
    ik_slack_rot_dist_ = ik_slack_rot_dist;
}

bool DynamicSystemHSR::find_ik(const Eigen::Isometry3d &desiredState, const tf::Transform &desiredGripperTfWorld) {
    // *** make request for IK ***
    // useing base DOF as planar movement. analytic IK have to use kPlanar.
    IKRequest req(tmc_manipulation_types::kNone);
    // reference frame. analytic IK have to use hand_palm_link
    req.frame_name = robo_config_.global_link_transform;
    // offset from reference frame.
    req.frame_to_end = Affine3d::Identity();

    req.initial_angle.name = ik_joint_names_;
    // reference joint angles
    req.initial_angle.position.resize(6);
    req.initial_angle.position << current_joint_values_[0], current_joint_values_[1], current_joint_values_[2],
                                  current_joint_values_[3], current_joint_values_[4], current_joint_values_[5];
    req.use_joints = ik_joint_names_;
    Eigen::VectorXd weight_vector;
    // weight of joint angle. #1-3 weights are for base DOF.
    weight_vector.resize(8);
    weight_vector << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;

    req.weight = weight_vector;
    req.origin_to_base = Affine3d::Identity();
    // reference positon.
    req.ref_origin_to_end = desiredState;

    // output values.
    JointState solution;
    Eigen::Affine3d origin_to_hand_result;
    // Eigen::Affine3d origin_to_base_solution;
    tmc_robot_kinematics_model::IKResult result;

    // Solve.
    result = numeric_solver_->Solve(req,
                                    solution,
                                    // origin_to_base_solution,
                                    origin_to_hand_result);

    kinematic_state_->setJointGroupPositions(joint_model_group_, solution.position);

    // Due to limit arm capabilities for most poses it will not be possible to find exact solution. Therefore allow a bit variance
    const Eigen::Affine3d &solution_state = kinematic_state_->getGlobalLinkTransform(robo_config_.global_link_transform);
    tf::Transform solution_state_tf, desiredState_tf;
    tf::transformEigenToTF(solution_state, solution_state_tf);
    tf::transformEigenToTF(desiredState, desiredState_tf);

    dist_solution_desired_ = (solution_state_tf.getOrigin() - desiredState_tf.getOrigin()).length();
    rot_dist_solution_desired_ = utils::calc_rot_dist(solution_state_tf, desiredState_tf);
    // std::cout << "success: " << (result == tmc_robot_kinematics_model::kSuccess) << ", dist_solution_desired_: " << dist_solution_desired_ << ", rot_dist_solution_desired_: " << rot_dist_solution_desired_ << std::endl;

    if (ik_slack_dist_ == 0.0) {
        return result == tmc_robot_kinematics_model::kSuccess;
    } else {
        // Due to the kinematics an exact solution is not possible in most situations
        // make slightly stricter than success_thres_dist_ as numeric error might cause it to fail to terminate if it finishes with an error margin of success_thres_dist_
        // NOTE: RISK OF GETTING STUCK IF success_thres_dist_ < ik_slack_dist_!
        double dist_desired_goal = (desiredGripperTfWorld.getOrigin() - currentGripperGOAL_.getOrigin()).length();
        if ((success_thres_dist_ < ik_slack_dist_) && (dist_desired_goal < 0.01)) {
            // enforce to achieve the final goal irrespective of the slack we give it
            return (dist_solution_desired_ < success_thres_dist_ && rot_dist_solution_desired_ < success_thres_rot_);
        } else {
            return (dist_solution_desired_ < ik_slack_dist_ && rot_dist_solution_desired_ < ik_slack_rot_dist_);
        }
    }
}

double DynamicSystemHSR::calc_reward(bool found_ik, double regularization) {
    // a)
    double reward = DynamicSystem_base::calc_reward(found_ik, regularization);

    if (sol_dist_reward_ && found_ik) {
        // scale to be a max of penalty_scaling_ * (-0.5 - 0.5)
        double dist_penalty = 0.5 * pow(dist_solution_desired_, 2) / pow(ik_slack_dist_, 2) + 0.5 * pow(rot_dist_solution_desired_, 2) / pow(ik_slack_rot_dist_, 2);
        reward -= dist_penalty;
    }

    return reward;
}

void DynamicSystemHSR::send_arm_command(const std::vector<double> &target_joint_values, double exec_duration) {
    int j = 0;
    for (int i = 0; i < joint_names_.size(); i++) {
        // std::cout << joint_names_[i] << ": " << target_joint_values[i] << std::endl;
        // part of the movit controller definition, but not part of /opt/ros/melodic/share/hsrb_common_config/params/hsrb_controller_config.yaml
        if (joint_names_[i] != "wrist_ft_sensor_frame_joint") {
            arm_goal_.trajectory.joint_names[j] = joint_names_[i];
            arm_goal_.trajectory.points[0].positions[j] = target_joint_values[i];
            arm_goal_.trajectory.points[0].velocities[j] = 0.0;
            j++;
        }
    }

    // When to start the trajectory. Will get dropped with 0
    arm_goal_.trajectory.header.stamp = ros::Time::now() + ros::Duration(0.05);
    // To be reached x seconds after starting along the trajectory
    arm_goal_.trajectory.points[0].time_from_start = ros::Duration(exec_duration);

    // send off commands to run in parallel
    arm_client_->sendGoal(arm_goal_);
}

bool DynamicSystemHSR::get_arm_success() {
    arm_client_->waitForResult(ros::Duration(10.0));
    if (arm_client_->getState() != actionlib::SimpleClientGoalState::SUCCEEDED) {
        ROS_WARN("The arm_client_ failed.");
        // throw std::runtime_error("The arm_client_ failed.");
        return false;
    } else {
        return true;
    }
}

void DynamicSystemHSR::open_gripper(double position, bool wait_for_result) {
    // hsr takes 1.0 as completely open -> calculate proportional to an assumed max. opening of 0.1m
    position = std::min(position / 0.1, 1.0);

    control_msgs::FollowJointTrajectoryGoal goal;
    goal.trajectory.joint_names.push_back("hand_motor_joint");

    goal.trajectory.points.resize(1);
    goal.trajectory.points[0].positions.resize(1);
    goal.trajectory.points[0].effort.resize(1);
    goal.trajectory.points[0].velocities.resize(1);

    goal.trajectory.points[0].positions[0] = position;
    goal.trajectory.points[0].velocities[0] = 0.0;
    goal.trajectory.points[0].effort[0] = 500;
    goal.trajectory.points[0].time_from_start = ros::Duration(3.0);

    // send message to the action server
    gripper_client_->sendGoal(goal);

    if (wait_for_result) {
        gripper_client_->waitForResult(ros::Duration(5.0));

        if (gripper_client_->getState() != actionlib::SimpleClientGoalState::SUCCEEDED)
            ROS_WARN("The gripper controller failed.");
    }
}

void DynamicSystemHSR::close_gripper(double position, bool wait_for_result) {
    // 0.0 is not completely closed, but rather both 'forks' pointing straight ahead. -0.02 is roughly fully closed
    open_gripper(position - 0.02, wait_for_result);
}
