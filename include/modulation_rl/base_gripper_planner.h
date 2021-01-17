#pragma once

#include <ros/ros.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <modulation_rl/modulation.h>
#include <eigen_conversions/eigen_msg.h>
#include <tf/transform_listener.h>
#include <tf/tf.h>
#include "tf/transform_datatypes.h"
#include <tf_conversions/tf_eigen.h>

#include <modulation_rl/utils.h>


struct GripperPlan {
    tf::Transform nextGripperTransform;
    tf::Transform nextBaseTransform;
};

struct PlannedVelocities {
    tf::Quaternion dq;
    tf::Vector3 vel_world;
    tf::Vector3 vel_rel;
};


class BaseGripperPlanner
{
protected:
    const double min_planner_velocity_;
    const double max_planner_velocity_;
    const bool use_base_goal_;
    GripperPlan prevPrevPlan_;
    GripperPlan prevPlan_;

public:
    BaseGripperPlanner(
        double min_planner_velocity,
        double max_planner_velocity,
        bool use_base_goal
    );

    PlannedVelocities transformToVelocity(tf::Transform current, tf::Transform next, tf::Transform baseTransform, double upper_vel_limit);

    virtual GripperPlan get_next_velocities(
        double time,
        double dt,
        const tf::Transform &currentBaseTransform,
        const tf::Transform &currentGripperTransform,
        const tf::Vector3 &current_base_vel_world,
        const tf::Vector3 &current_gripper_vel_world,
        const tf::Quaternion &current_gripper_dq,
        bool update_prev_plan
    ) = 0;

    virtual void reset(
        tf::Transform gripperGoal,
        tf::Transform initialGripperTransform,
        tf::Transform baseGoal,
        tf::Transform initialBaseTransform,
        std::string gmm_model_path,
        double gmm_base_offset
    ) = 0;

    virtual tf::Transform get_last_attractor() = 0;
    virtual GripperPlan get_prev_plan(){ return prevPlan_; };
    virtual GripperPlan get_velocities_from_prevPrev(
        double time,
        double dt,
        const tf::Transform &currentBaseTransform,
        const tf::Transform &currentGripperTransform,
        const tf::Vector3 &current_base_vel_world,
        const tf::Vector3 &current_gripper_vel_world,
        const tf::Quaternion &current_gripper_dq
    ) = 0;
    virtual std::vector<tf::Transform> get_mus() = 0;
};
