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
#include <boost/shared_ptr.hpp>

#include <modulation_rl/utils.h>
#include <modulation_rl/base_gripper_planner.h>
#include <modulation_rl/gaussian_mixture_model.h>

class GMMPlanner : public BaseGripperPlanner
{
private:
    boost::shared_ptr<GaussianMixtureModel> gaussian_mixture_model_;
    std::string gmm_model_path_ = "";
    const tf::Vector3 tip_to_gripper_offset_;
    const tf::Quaternion gripper_to_base_rot_offset_;

    GripperPlan get_next_step(
        double time,
        double dt,
        const tf::Transform &currentBaseTransform,
        const tf::Vector3 &current_base_vel_world,
        const tf::Vector3 &current_gripper_vel_world,
        const tf::Quaternion &current_gripper_dq,
        const GripperPlan &prevPlan,
        bool do_update
    );
public:
    GMMPlanner(
        double min_planner_velocity,
        double max_planner_velocity,
        bool use_base_goal,
        const tf::Vector3 tip_to_gripper_offset,
        const tf::Quaternion gripper_to_base_rot_offset
    );

    void reset(
        tf::Transform gripperGoal,
        tf::Transform initialGripperTransform,
        tf::Transform baseGoal,
        tf::Transform initialBaseTransform,
        std::string gmm_model_path,
        double gmm_base_offset
    );

    GripperPlan get_next_velocities(
        double time,
        double dt,
        const tf::Transform &currentBaseTransform,
        const tf::Transform &currentGripperTransform,
        const tf::Vector3 &current_base_vel_world,
        const tf::Vector3 &current_gripper_vel_world,
        const tf::Quaternion &current_gripper_dq,
        bool update_prev_plan
    );
    tf::Transform get_last_attractor();
    GripperPlan get_prev_plan();
    GripperPlan get_velocities_from_prevPrev(
        double time,
        double dt,
        const tf::Transform &currentBaseTransform,
        const tf::Transform &currentGripperTransform,
        const tf::Vector3 &current_base_vel_world,
        const tf::Vector3 &current_gripper_vel_world,
        const tf::Quaternion &current_gripper_dq
    );

    std::vector<tf::Transform> get_mus();
};
