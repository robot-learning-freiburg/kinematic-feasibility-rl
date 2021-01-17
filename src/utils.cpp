#include <modulation_rl/utils.h>

namespace utils {
    void print_vector3(tf::Vector3 v, std::string descr){
        std::cout << descr << ": " << v.x() << ", " << v.y() << ", " << v.z() << std::endl;
    }

    void print_q(tf::Quaternion q, std::string descr){
        std::cout << descr << ": " << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w() << std::endl;
    }

    void print_t(tf::Transform t, std::string descr){
        tf::Vector3 v = t.getOrigin();
        tf::Quaternion q = t.getRotation();
        std::cout << descr << ". O: " << v.x() << ", " << v.y() << ", " << v.z() << ", Q: " << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w() << std::endl;
    }

    void print_array_double(std::vector<double> array, std::string descr){
        std::cout << descr  << ", size: " << array.size() << ", ";
        for (int i=0; i<array.size(); i++){
            std::cout << array[i] << ", ";
        }
        std::cout << std::endl;
    }

    void print_array_str(std::vector<std::string> array, std::string descr){
        std::cout << descr << array.size() << std::endl;
        for (int i=0; i<array.size(); i++){
            std::cout << array[i] << ", ";
        }
        std::cout << std::endl;
    }

    tf::Vector3 q_to_rpy(tf::Quaternion q){
        double roll, pitch, yaw;
        tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
        return tf::Vector3(roll, pitch, yaw);
    }

    void add_rotation(std::vector<double> &obs_vector, tf::Quaternion q, bool use_euler){
        if (use_euler){
            tf::Vector3 euler = q_to_rpy(q);
            obs_vector.push_back(euler.x());
            obs_vector.push_back(euler.y());
            obs_vector.push_back(euler.z());
        } else {
            q.normalize();
            obs_vector.push_back(q.x());
            obs_vector.push_back(q.y());
            obs_vector.push_back(q.z());
            obs_vector.push_back(q.w());
        }
    }

    void add_vector3(std::vector<double> &obs_vector, tf::Vector3 v){
        obs_vector.push_back(v.x());
        obs_vector.push_back(v.y());
        obs_vector.push_back(v.z());
    }

    double calc_rot_dist(tf::Transform a, tf::Transform b){
        double inner_prod = a.getRotation().dot(b.getRotation());
        return 1.0 - pow(inner_prod, 2.0);
    }

    double vec3_abs_max(tf::Vector3 v){
        tf::Vector3 v_abs = v.absolute();
        return std::max(std::max(v_abs.x(), v_abs.y()),
                        v_abs.z());
    }

    visualization_msgs::Marker marker_from_transform(tf::Transform t, std::string ns, std::string color, double alpha, int marker_id, std::string frame_id){
        return marker_from_transform(t, ns, get_color_msg(color, alpha), marker_id, frame_id);
    }

    visualization_msgs::Marker marker_from_transform(tf::Transform t, std::string ns, std_msgs::ColorRGBA color, int marker_id, std::string frame_id){
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time();
        marker.ns = ns;

        marker.type = visualization_msgs::Marker::ARROW;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = t.getOrigin().x();
        marker.pose.position.y = t.getOrigin().y();
        marker.pose.position.z = t.getOrigin().z();
        marker.pose.orientation.x = t.getRotation().x();
        marker.pose.orientation.y = t.getRotation().y();
        marker.pose.orientation.z = t.getRotation().z();
        marker.pose.orientation.w = t.getRotation().w();
        marker.scale.x = 0.1;
        marker.scale.y = 0.025;
        marker.scale.z = 0.025;

        // more and more red from 0 to 100
        marker.color = color;
        marker.id = marker_id;
        return marker;
    }

    std_msgs::ColorRGBA get_color_msg(std::string color_name, double alpha){
        std_msgs::ColorRGBA c;
        if(color_name == "blue"){
            c.b = 1.0;
        } else if(color_name == "pink") {
            c.r = 1.0;
            c.g = 105.0 / 255.0;
            c.b = 147.0 / 255.0;
        } else if(color_name == "orange"){
            c.r = 1.0;
            c.g = 159.0 / 255.0;
            c.b = 0.0;
        } else if(color_name == "yellow"){
            c.r = 1.0;
            c.g = 1.0;
            c.b = 0.0;
        } else if(color_name == "cyan"){
            c.r = 0.0;
            c.g = 128.0 / 255.0;
            c.b = 1.0;
        } else if(color_name == "green"){
            c.r = 0.0;
            c.g = 1.0;
            c.b = 0.0;
        } else if(color_name == "red"){
            c.r = 1.0;
            c.g = 0.0;
            c.b = 0.0;
        } else {
            throw std::runtime_error("unknown color");
        }
        c.a = alpha;
        return c;
    }

    tf::Vector3 min_max_scale_vel(tf::Vector3 vel, double min_vel, double max_vel){
        // find denominator to keep it in range [min_planner_velocity_, max_planner_velocity_]
        double max_abs_vector_value = utils::vec3_abs_max(vel);
        // in case vel is a vector of all zeros avoid division by zero
        if (max_abs_vector_value == 0.0){
            return tf::Vector3(min_vel, min_vel, min_vel);
        }
        double max_denom;
        if (min_vel < 0.001){
            max_denom = 1.0;
        } else {
            max_denom = std::min(max_abs_vector_value / min_vel, 1.0);
        }
        double min_denom = max_abs_vector_value / max_vel;
        double denom = std::max(max_denom, min_denom);
        return vel / denom;
    }

    tf::Vector3 max_clip_vel(tf::Vector3 vel, double max_vel){
        tf::Vector3 clipped_vel;
        clipped_vel.setX(std::max(std::min(vel.x(), max_vel), -max_vel));
        clipped_vel.setY(std::max(std::min(vel.y(), max_vel), -max_vel));
        clipped_vel.setZ(std::max(std::min(vel.z(), max_vel), -max_vel));
        return clipped_vel;
    }

    tf::Vector3 norm_scale_vel(tf::Vector3 vel, double  min_vel_norm, double max_vel_norm){
        double norm = vel.length();
        if (norm == 0.0){
            return vel;
        } else if (max_vel_norm == 0.0) {
            return tf::Vector3(0.0, 0.0, 0.0);
        } else {
            double max_denom;
            if (min_vel_norm < 0.00000001){
                max_denom = 1.0;
            } else {
                max_denom = std::min(norm / min_vel_norm, 1.0);
            }
            double min_denom = norm / max_vel_norm;
            double denom = std::max(max_denom, min_denom);

    //        assert((vel / denom).length() >= min_vel_norm - 0.001);
    //        assert((vel / denom).length() <= max_vel_norm + 0.001);

            return vel / denom;
        }
    }

    double clamp_double(double value, double min_value, double max_value){
        return std::max(std::min(value, max_value), min_value);
    }

    tf::Transform tip_to_gripper_goal(
        const tf::Transform &gripperTipGoalWorld,
        const tf::Vector3 &tip_to_gripper_offset,
        const tf::Quaternion &gripper_to_base_rot_offset
    ){

    // gripper tip offset from wrist
    tf::Transform goal_no_trans(gripperTipGoalWorld);
    goal_no_trans.setOrigin(tf::Vector3(0, 0, 0));
    tf::Vector3 offset_pos = goal_no_trans * tip_to_gripper_offset;

    tf::Transform gripper_goal_wrist_world(gripperTipGoalWorld);
    gripper_goal_wrist_world.setOrigin(gripper_goal_wrist_world.getOrigin() - offset_pos);

    // different rotations between gripper joint and base/world
    gripper_goal_wrist_world.setRotation((gripper_goal_wrist_world.getRotation() * gripper_to_base_rot_offset).normalized());
    // utils::print_vector3(offset_pos, "offset_pos");
    // utils::print_q(gripper_to_base_rot_offset, "gripper_to_base_rot_offset");
    // utils::print_t(gripper_goal_wrist_world, "gripper_goal_wrist_world");
    return gripper_goal_wrist_world;
}

    tf::Transform gripper_to_tip_goal(
            const tf::Transform &gripperWristGoalWorld,
            const tf::Vector3 &tip_to_gripper_offset,
            const tf::Quaternion &gripper_to_base_rot_offset
            ){
        tf::Transform gripper_goal_tip_world;
        gripper_goal_tip_world.setIdentity();

        // different rotations between gripper joint and base/world
        gripper_goal_tip_world.setRotation(gripperWristGoalWorld.getRotation() * gripper_to_base_rot_offset.inverse());

        // gripper tip offset from wrist
        tf::Vector3 offset_pos = gripper_goal_tip_world * tip_to_gripper_offset;
        gripper_goal_tip_world.setOrigin(gripperWristGoalWorld.getOrigin() + offset_pos);

        return gripper_goal_tip_world;
    }

    double rpy_angle_diff(double next, double prev){
        double diff = next - prev;
        if (diff > M_PI){
            diff = -2 * M_PI + diff;
        } else if (diff < -M_PI){
            diff = 2 * M_PI + diff;
        }
        return diff;
    }

   bool startsWith(const std::string & str, const std::string substr){
      return (str.find(substr) == 0);
   }

   bool endsWith(const std::string & str, const std::string substr){
      size_t pos = str.rfind(substr);
      if(pos == std::string::npos) // doesnt even contain it
         return false;

      size_t len = str.length();
      size_t elen = substr.length();
      // at end means: Pos found + length of end equal length of full string.
      if( pos + elen == len ) {
         return true;
      }

      // not at end
      return false;
   }

   std::string trim(const std::string & s){
      if(s.length() == 0)
         return s;
      size_t b = s.find_first_not_of(" \t\r\n");
      size_t e = s.find_last_not_of(" \t\r\n");
      if(b == std::string::npos)
         return "";
      return std::string(s, b, e - b + 1);
   }

}
