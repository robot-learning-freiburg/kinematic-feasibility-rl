#pragma once

#include <eigen_conversions/eigen_msg.h>
#include <modulation_rl/modulation.h>
#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "tf/transform_datatypes.h"

#include <modulation_rl/utils.h>

struct GripperPlan {
    tf::Transform nextGripperTransform;
    tf::Transform nextBaseTransform;
};

struct PlannedVelocities {
    tf::Quaternion dq;
    tf::Vector3 vel_world;
    tf::Vector3 vel_rel;

    void init() {
        dq = tf::Quaternion(0, 0, 0, 1);
        vel_world = tf::Vector3(0, 0, 0);
        vel_rel = tf::Vector3(0, 0, 0);
    }
};

class BaseGripperPlanner {
  protected:
    GripperPlan prevPlan_;

  public:
    BaseGripperPlanner();

    PlannedVelocities transformToVelocity(tf::Transform current, tf::Transform next, tf::Transform baseTransform, double upper_vel_limit);

    virtual GripperPlan get_next_velocities(double time,
                                            double dt,
                                            const tf::Transform &currentBaseTransform,
                                            const tf::Transform &currentGripperTransform,
                                            const tf::Vector3 &current_base_vel_world,
                                            const tf::Vector3 &current_gripper_vel_world,
                                            const tf::Quaternion &current_gripper_dq,
                                            const double &min_planner_velocity,
                                            const double &max_planner_velocity,
                                            bool update_prev_plan) = 0;

    virtual tf::Transform get_last_attractor() = 0;
    virtual GripperPlan get_prev_plan() { return prevPlan_; };
    virtual std::vector<tf::Transform> get_mus() = 0;
};
