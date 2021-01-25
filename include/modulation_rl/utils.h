#ifndef UTILS_H
#define UTILS_H

#include <eigen_conversions/eigen_msg.h>
#include <math.h>
#include <ros/ros.h>
#include <tf/tf.h>
#include <tf_conversions/tf_eigen.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "std_msgs/ColorRGBA.h"
#include "tf/transform_datatypes.h"
#include "visualization_msgs/MarkerArray.h"

#include <fstream>
#include <iostream>
#include <sstream>

typedef std::map<std::string, double> PathPoint;

struct RoboConf {
    const std::string name;
    const std::string joint_model_group_name;
    const std::string frame_id;
    const std::string global_link_transform;
    const std::string scene_collision_group_name;
    const tf::Vector3 tip_to_gripper_offset;
    const tf::Quaternion gripper_to_base_rot_offset;
    const std::vector<std::string> neutral_pos_joint_names;
    const std::vector<double> neutral_pos_values;
    const std::string base_cmd_topic;
    const double base_vel_rng;
    const double base_rot_rng;
    const double z_min;
    const double z_max;
    const double restricted_ws_z_min;
    const double restricted_ws_z_max;
    const double gmm_base_offset;
};

namespace utils {
    void print_vector3(tf::Vector3 v, std::string descr);
    void print_q(tf::Quaternion q, std::string descr);
    void print_t(tf::Transform t, std::string descr);
    void print_array_double(std::vector<double> array, std::string descr);
    void print_array_str(std::vector<std::string> array, std::string descr);
    tf::Vector3 q_to_rpy(tf::Quaternion q);
    void add_rotation(std::vector<double> &obs_vector, tf::Quaternion q, bool use_euler);
    void add_vector3(std::vector<double> &obs_vector, tf::Vector3 v);
    double calc_rot_dist(tf::Transform a, tf::Transform b);
    double vec3_abs_max(tf::Vector3 v);
    visualization_msgs::Marker marker_from_transform(tf::Transform t, std::string ns, std::string color, double alpha, int marker_id, std::string frame_id);
    visualization_msgs::Marker marker_from_transform(tf::Transform t, std::string ns, std_msgs::ColorRGBA color, int marker_id, std::string frame_id);
    std_msgs::ColorRGBA get_color_msg(std::string color_name, double alpha = 1.0);
    tf::Vector3 min_max_scale_vel(tf::Vector3 vel, double min_vel, double max_vel);
    tf::Vector3 norm_scale_vel(tf::Vector3 vel, double min_vel_norm, double max_vel_norm);
    tf::Vector3 max_clip_vel(tf::Vector3 vel, double max_vel);
    double clamp_double(double value, double min_value, double max_value);
    tf::Transform tip_to_gripper_goal(const tf::Transform &gripperTipGoalWorld, const tf::Vector3 &tip_to_gripper_offset, const tf::Quaternion &gripper_to_base_rot_offset);
    tf::Transform gripper_to_tip_goal(const tf::Transform &gripperWristGoalWorld, const tf::Vector3 &tip_to_gripper_offset, const tf::Quaternion &gripper_to_base_rot_offset);
    double rpy_angle_diff(double next, double prev);
    bool startsWith(const std::string &str, const std::string substr);
    bool endsWith(const std::string &str, const std::string substr);
    std::string trim(const std::string &s);
    void pathPoint_insert_transform(PathPoint &path_point, std::string name, tf::Transform tf, bool yaw_only = false);
}  // namespace utils

#endif