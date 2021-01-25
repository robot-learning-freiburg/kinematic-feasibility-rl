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

#include <modulation_rl/base_gripper_planner.h>
#include <modulation_rl/utils.h>

class LinearPlanner : public BaseGripperPlanner {
  private:
    tf::Transform gripperGoal_;
    tf::Transform baseGoal_;
    tf::Transform initialGripperTransform_;
    tf::Transform initialBaseTransform_;
    double initial_dist_to_gripper_goal_;

    tf::Vector3 get_vel(const tf::Transform &current, const tf::Transform &goal, double min_vel, double max_vel);
    tf::Quaternion get_rot(const tf::Transform initial, const tf::Transform &next, const tf::Transform &goal, double initial_dist);

    GripperPlan calc_next_step(const GripperPlan &prevPlan,
                               const double &dt,
                               const double &min_velocity,
                               const double &max_velocity);

  public:
    LinearPlanner(tf::Transform gripperGoal,
                  tf::Transform initialGripperTransform,
                  tf::Transform baseGoal,
                  tf::Transform initialBaseTransform);

    GripperPlan get_next_velocities(double time,
                                    double dt,
                                    const tf::Transform &currentBaseTransform,
                                    const tf::Transform &currentGripperTransform,
                                    const tf::Vector3 &current_base_vel_world,
                                    const tf::Vector3 &current_gripper_vel_world,
                                    const tf::Quaternion &current_gripper_dq,
                                    const double &min_velocity,
                                    const double &max_velocity,
                                    bool update_prev_plan);

    tf::Transform get_last_attractor() { throw std::runtime_error("NOT IMPLEMENTED OR NEEDED FOR LINEAR PLANNER"); };
    virtual std::vector<tf::Transform> get_mus() { throw std::runtime_error("NOT IMPLEMENTED OR NEEDED FOR LINEAR PLANNER"); };
};