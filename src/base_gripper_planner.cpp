#include <modulation_rl/base_gripper_planner.h>


BaseGripperPlanner::BaseGripperPlanner(
        double min_planner_velocity,
        double max_planner_velocity,
        bool use_base_goal
    ) : min_planner_velocity_ { min_planner_velocity },
        max_planner_velocity_ { max_planner_velocity },
        use_base_goal_ { use_base_goal }
    {  };


PlannedVelocities BaseGripperPlanner::transformToVelocity(
        tf::Transform current,
        tf::Transform next,
        tf::Transform baseTransform,
        double upper_vel_limit
    ){
    PlannedVelocities planned_vel;

    planned_vel.vel_world = next.getOrigin() - current.getOrigin();

    if (upper_vel_limit != 0.0){
        // planned_vel.vel_world = utils::min_max_scale_vel(planned_vel.vel_world, 0.0, upper_vel_limit);
        planned_vel.vel_world = utils::norm_scale_vel(planned_vel.vel_world, 0.0, upper_vel_limit);
        // planned_vel.vel_world = utils::max_clip_vel(planned_vel.vel_world, upper_vel_limit);
    }

    tf::Transform base_no_trans = baseTransform;
    base_no_trans.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
    planned_vel.vel_rel = base_no_trans.inverse() * planned_vel.vel_world;

    // planned change in rotation
    // a) planned_vel.dq_world = next.getRotation() - current.getRotation();
    // b) defined as dq * current == next
    planned_vel.dq = (next.getRotation() * current.getRotation().inverse()).normalized();
    // c) defined as dq * current == next in base frame
    // tf::Transform current_rel = base_no_trans.inverse() * current;
    // tf::Transform next_rel = base_no_trans.inverse() * next;
    // planned_vel.dq_rel = (next_rel.getRotation() * current_rel.getRotation().inverse()).normalized();
    // Note that these do not result in the same difference in RPY
    // tf::Matrix3x3(planned_vel.dq_world).getRPY(roll_, pitch_, yaw_);
    // tf::Matrix3x3(planned_vel.dq_rel).getRPY(roll2_, pitch2_, yaw2_);
    // std::cout << "dq_world RPY roll: " << roll_ << ", pitch: " << pitch_ << ", yaw: " << yaw_ << std::endl;
    // std::cout << "dq_rel RPY roll: " << roll2_ << ", pitch: " << pitch2_ << ", yaw: " << yaw2_ << std::endl;
    // d) quaternion of the difference in RPY (identical in world and base frame. But different to all alternatives above)
    // tf::Vector3 rpy_current = utils::q_to_rpy(current.getRotation());
    // tf::Vector3 rpy_next = utils::q_to_rpy(next.getRotation());
    // planned_vel.dq.setRPY(utils::rpy_angle_diff(rpy_next.x(), rpy_current.x()),
    //                       utils::rpy_angle_diff(rpy_next.y(), rpy_current.y()),
    //                       utils::rpy_angle_diff(rpy_next.z(), rpy_current.z()));
    // planned_vel.dq.normalize();

    return planned_vel;
}
