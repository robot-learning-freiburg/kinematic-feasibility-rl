#include <modulation_rl/linear_planner.h>

LinearPlanner::LinearPlanner(tf::Transform gripperGoal, tf::Transform initialGripperTransform, tf::Transform baseGoal, tf::Transform initialBaseTransform) : BaseGripperPlanner() {
    gripperGoal_ = gripperGoal;
    baseGoal_ = baseGoal;
    initialGripperTransform_ = initialGripperTransform;
    initialBaseTransform_ = initialBaseTransform;
    initial_dist_to_gripper_goal_ = (gripperGoal.getOrigin() - initialGripperTransform.getOrigin()).length();

    prevPlan_.nextGripperTransform = initialGripperTransform;
    prevPlan_.nextBaseTransform = initialBaseTransform;
};

tf::Vector3 LinearPlanner::get_vel(const tf::Transform &current, const tf::Transform &goal, double min_vel, double max_vel) {
    tf::Vector3 vec_to_goal = (goal.getOrigin() - current.getOrigin());
    return utils::norm_scale_vel(vec_to_goal / 100.0, min_vel, max_vel);
}

tf::Quaternion LinearPlanner::get_rot(const tf::Transform initial, const tf::Transform &next, const tf::Transform &goal, double initial_dist) {
    double dist_to_goal_post = (goal.getOrigin() - next.getOrigin()).length();
    double slerp_pct = std::min(1 - dist_to_goal_post / initial_dist, 0.9999);
    tf::Quaternion planned_q = initial.getRotation().slerp(goal.getRotation(), slerp_pct);
    planned_q.normalize();
    return planned_q;
}

GripperPlan LinearPlanner::calc_next_step(const GripperPlan &prevPlan, const double &dt, const double &min_velocity, const double &max_velocity) {
    GripperPlan plan;
    double min_vel = min_velocity * dt;
    double max_vel = max_velocity * dt;

    // new velocities based on distance to current Goal
    tf::Vector3 planned_gripper_vel = get_vel(prevPlan.nextGripperTransform, gripperGoal_, min_vel, max_vel);
    tf::Vector3 planned_base_vel;
    planned_base_vel.setValue(planned_gripper_vel.x(), planned_gripper_vel.y(), 0.0);

    plan.nextGripperTransform.setOrigin(prevPlan.nextGripperTransform.getOrigin() + planned_gripper_vel);
    plan.nextBaseTransform.setOrigin(prevPlan.nextBaseTransform.getOrigin() + planned_base_vel);

    // new rotations: interpolated from start to gaol based on distance achieved so far
    tf::Quaternion planned_gripper_q = get_rot(initialGripperTransform_, plan.nextGripperTransform, gripperGoal_, initial_dist_to_gripper_goal_);
    plan.nextGripperTransform.setRotation(planned_gripper_q);
    // base
    // a) do not rotate
    plan.nextBaseTransform.setRotation(prevPlan.nextBaseTransform.getRotation());
    // b) rotate to match yaw of next gripper plan
    // tf::Vector3 next_gripper_rpy = utils::q_to_rpy(planned_gripper_q);
    // tf::Quaternion next_base_q;
    // next_base_q.setRPY(0.0, 0.0, next_gripper_rpy.z());
    // plan.nextBaseTransform.setRotation(next_base_q);

    // tf::Transform ggoal_floor;
    // ggoal_floor.setOrigin(tf::Vector3(gripperGoal_.getOrigin().x(), gripperGoal_.getOrigin().y(), 0.0));
    // tf::Vector3 ggoal_rpy = utils::q_to_rpy(gripperGoal_.getRotation());
    // tf::Quaternion ggoal_q;
    // ggoal_q.setRPY(0.0, 0.0, ggoal_rpy.z());
    // ggoal_floor.setRotation(ggoal_q);
    // tf::Quaternion next_base_q = get_rot(prevPlan.nextBaseTransform, prevPlan.nextBaseTransform, ggoal_floor, (ggoal_floor.getOrigin() -
    // initialBaseTransform_.getOrigin()).length()); plan.nextBaseTransform.setRotation(next_base_q);
    return plan;
}

GripperPlan LinearPlanner::get_next_velocities(double time,
                                               double dt,
                                               const tf::Transform &currentBaseTransform_,
                                               const tf::Transform &currentGripperTransform_,
                                               const tf::Vector3 &current_base_vel_world,
                                               const tf::Vector3 &current_gripper_vel_world,
                                               const tf::Quaternion &current_gripper_dq,
                                               const double &min_velocity,
                                               const double &max_velocity,
                                               bool update_prev_plan) {
    //     ROS_INFO("linearPlanner time: %f, dt: %f", time, dt);

    //     utils::print_t(prevPlan_.nextGripperTransform, "prevPlan_.nextGripperTransform");
    //     utils::print_t(prevPlan_.nextBaseTransform, "prevPlan_.nextBaseTransform");
    //
    //     utils::print_vector3(current_base_vel_world, "current_base_vel_world");
    //     utils::print_vector3(current_gripper_vel_world, "current_gripper_vel_world");
    //     utils::print_q(current_gripper_dq, "current_gripper_dq");

    //    utils::print_t(prevPlan_.nextGripperTransform, "prevPlan_.nextGripperTransform");
    //    utils::print_t(prevPlan_.nextBaseTransform, "prevPlan_.nextBaseTransform");

    double min_vel, max_vel;
    min_vel = min_velocity;
    max_vel = max_velocity;

    GripperPlan nextPlan = calc_next_step(prevPlan_, dt, min_vel, max_vel);
    if (update_prev_plan) {
        prevPlan_ = nextPlan;
    }

    //     utils::print_t(nextPlan.nextGripperTransform, "nextPlan.nextGripperTransform");
    //     utils::print_t(nextPlan.nextBaseTransform, "nextPlan.nextBaseTransform");

    return nextPlan;
}
