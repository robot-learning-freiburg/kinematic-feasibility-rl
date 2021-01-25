#include <modulation_rl/gmm_planner.h>

GMMPlanner::GMMPlanner(const tf::Vector3 tip_to_gripper_offset,
                       const tf::Quaternion gripper_to_base_rot_offset,
                       tf::Transform gripperGoal,
                       tf::Transform initialGripperTransform,
                       tf::Transform baseGoal,
                       tf::Transform initialBaseTransform,
                       std::string gmm_model_path,
                       double gmm_base_offset) :
    BaseGripperPlanner(),
    tip_to_gripper_offset_{tip_to_gripper_offset},
    gripper_to_base_rot_offset_{gripper_to_base_rot_offset} {
    double max_rot = 0.1;
    gmm_model_path_ = gmm_model_path;
    gaussian_mixture_model_.reset(new GaussianMixtureModel(max_rot, max_rot));
    // For each learned object manipulation there are three action models one for grasping, one for manipulation and one for releasing
    // Stick to grasp and move of KallaxTuer first.
    gaussian_mixture_model_->loadFromFile(gmm_model_path_);
    // adaptModel(objectPose) transforms the GMM to a given object pose of the handled object (In this case we just use the currentGripperGOAL as new object pose)
    // afterwrds the model can be integrated step by step to generate new gripper poses leading to the correct handling of the object ()
    gaussian_mixture_model_->adaptModel(gripperGoal, tf::Vector3(gmm_base_offset, 0, 0));

    // transform to a tip goal
    prevPlan_.nextGripperTransform = utils::gripper_to_tip_goal(initialGripperTransform, tip_to_gripper_offset_, gripper_to_base_rot_offset_);
    prevPlan_.nextBaseTransform = initialBaseTransform;
};

GripperPlan GMMPlanner::calc_next_step(double time,
                                       double dt,
                                       const tf::Transform &currentBaseTransform,
                                       const tf::Vector3 &current_base_vel_world,
                                       const tf::Vector3 &current_gripper_vel_world,
                                       const tf::Quaternion &current_gripper_dq,
                                       const GripperPlan &prevPlan,
                                       const double &min_velocity,
                                       const double &max_velocity,
                                       bool do_update) {
    // create eigen vectors for current pose and current speed
    // treat it as a planner that could be pre-computed: calculate next step from the transform we wanted to achieve, not the actually achieved one
    Eigen::VectorXf current_pose(14);
    current_pose << prevPlan.nextGripperTransform.getOrigin().x(), prevPlan.nextGripperTransform.getOrigin().y(), prevPlan.nextGripperTransform.getOrigin().z(),
                    prevPlan.nextGripperTransform.getRotation().x(), prevPlan.nextGripperTransform.getRotation().y(), prevPlan.nextGripperTransform.getRotation().z(), prevPlan.nextGripperTransform.getRotation().w(),
                    prevPlan.nextBaseTransform.getOrigin().x(), prevPlan.nextBaseTransform.getOrigin().y(), 0.0,
                    prevPlan.nextBaseTransform.getRotation().x(), prevPlan.nextBaseTransform.getRotation().y(), prevPlan.nextBaseTransform.getRotation().z(), prevPlan.nextBaseTransform.getRotation().w();

    // use the planned velocities as current velocities, assuming we follow a pre-calculated plan as above
    Eigen::VectorXf current_speed(14);
    current_speed << current_gripper_vel_world.x(), current_gripper_vel_world.y(), current_gripper_vel_world.z(),
                     current_gripper_dq.x(), current_gripper_dq.y(), current_gripper_dq.z(), current_gripper_dq.w(),
                     current_base_vel_world.x(), current_base_vel_world.y(), 0.0,
                     0.0, 0.0, 0.0, 0.0;

    gaussian_mixture_model_->integrateModel(time, dt, &current_pose, &current_speed, min_velocity, max_velocity, do_update);

    GripperPlan nextPlan;
    nextPlan.nextGripperTransform.setOrigin(tf::Vector3(current_pose[0], current_pose[1], current_pose[2]));
    nextPlan.nextGripperTransform.setRotation(tf::Quaternion(current_pose[3], current_pose[4], current_pose[5], current_pose[6]));
    nextPlan.nextBaseTransform.setOrigin(tf::Vector3(current_pose[7], current_pose[8], 0.0));
    nextPlan.nextBaseTransform.setRotation(tf::Quaternion(current_pose[10], current_pose[11], current_pose[12], current_pose[13]));
    return nextPlan;
}

GripperPlan GMMPlanner::get_next_velocities(double time,
                                            double dt,
                                            const tf::Transform &currentBaseTransform,
                                            const tf::Transform &currentGripperTransform,
                                            const tf::Vector3 &current_base_vel_world,
                                            const tf::Vector3 &current_gripper_vel_world,
                                            const tf::Quaternion &current_gripper_dq,
                                            const double &min_velocity,
                                            const double &max_velocity,
                                            bool update_prev_plan) {
    // ROS_INFO("gmmPlanner time slowed: %f, dt: %f", time, dt);
    // utils::print_t(prevPlan_.nextGripperTransform, "prevPlan_.nextGripperTransform Tip");
    // utils::print_t(prevPlan_.nextBaseTransform, "prevPlan_.nextBaseTransform");

    // utils::print_vector3(current_base_vel_world, "current_base_vel_world");
    // utils::print_vector3(current_gripper_vel_world, "current_gripper_vel_world");
    // utils::print_q(current_gripper_dq, "current_gripper_dq");

    double min_vel, max_vel;
    min_vel = 0.0;
    max_vel = max_velocity;

    GripperPlan nextPlan = calc_next_step(time,
                                          dt,
                                          currentBaseTransform,
                                          current_base_vel_world,
                                          current_gripper_vel_world,
                                          current_gripper_dq,
                                          prevPlan_,
                                          min_vel,
                                          max_vel,
                                          update_prev_plan);

    if (update_prev_plan){
        prevPlan_ = nextPlan;
    }

    GripperPlan nextPlanWrist = nextPlan;
    nextPlanWrist.nextGripperTransform = utils::tip_to_gripper_goal(nextPlan.nextGripperTransform, tip_to_gripper_offset_, gripper_to_base_rot_offset_);

    return nextPlanWrist;
}

// GMM takes the origin of the door as input goal -> transform to the wrist goal which is used in the rest of the env
tf::Transform GMMPlanner::get_last_attractor() {
    // goalState returns the values from the csv. Prob. relative to door origin or similar
    // tf::StampedTransform goalState = gaussian_mixture_model_->getGoalState();
    // tf::Transform goalStateWrist(goalState);
    int nrModes = gaussian_mixture_model_->getNr_modes();
    // muEigen seems to directly give us the wrist goal
    tf::Transform last_attractor;
    std::vector<Eigen::VectorXf> muEigen = gaussian_mixture_model_->getMu();
    last_attractor.setOrigin(tf::Vector3(muEigen[nrModes - 1][1], muEigen[nrModes - 1][2], muEigen[nrModes - 1][3]));
    last_attractor.setRotation(tf::Quaternion(muEigen[nrModes - 1][4], muEigen[nrModes - 1][5], muEigen[nrModes - 1][6], muEigen[nrModes - 1][7]));
    return last_attractor;
}

GripperPlan GMMPlanner::get_prev_plan() {
    GripperPlan nextPlan = BaseGripperPlanner::get_prev_plan();
    nextPlan.nextGripperTransform = utils::tip_to_gripper_goal(nextPlan.nextGripperTransform, tip_to_gripper_offset_, gripper_to_base_rot_offset_);
    return nextPlan;
}

std::vector<tf::Transform> GMMPlanner::get_mus() {
    int nrModes = gaussian_mixture_model_->getNr_modes();
    std::vector<tf::Transform> v;

    for (int i = 0; i < nrModes; i++) {
        std::vector<Eigen::VectorXf> muEigen = gaussian_mixture_model_->getMu();

        tf::Transform gripper_t;
        gripper_t.setOrigin(tf::Vector3(muEigen[i][1], muEigen[i][2], muEigen[i][3]));
        gripper_t.setRotation(tf::Quaternion(muEigen[i][4], muEigen[i][5], muEigen[i][6], muEigen[i][7]));

        tf::Transform base_t;
        base_t.setOrigin(tf::Vector3(muEigen[i][8], muEigen[i][9], 0.0));
        base_t.setRotation(tf::Quaternion(muEigen[i][11], muEigen[i][12], muEigen[i][13], muEigen[i][14]));

        v.push_back(gripper_t);
        v.push_back(base_t);
    }
    return v;
}