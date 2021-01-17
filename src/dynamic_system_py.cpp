#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
//#include <modulation_rl/dynamic_system.h>
//#include <modulation_rl/dynamic_system_base.h>
#include <modulation_rl/dynamic_system_pr2.h>
#include <modulation_rl/dynamic_system_tiago.h>
//#include <modulation_rl/dynamic_system_hsr.h>


namespace py = pybind11;

PYBIND11_MODULE(dynamic_system_py, m) {

//    py::class_<DynamicSystem>(m, "PR2EnvOLD")
//        .def(py::init<bool, uint32_t>())
//        .def("simulate_env_step_wrap", &DynamicSystem::simulate_env_step, "Execute the next time step in environment.")
//        .def("reset", &DynamicSystem::reset, "Reset environment.")
//        .def("visualize", &DynamicSystem::visualize_robot_pose, "Visualize trajectory.");

    py::class_<pathPoint>(m, "pathPoint")
        .def(py::init<double, double, double, double, double, bool>())
        .def_readwrite("dt", &pathPoint::dt)
        .def_readwrite("base_x", &pathPoint::base_x)
        .def_readwrite("base_y", &pathPoint::base_y)
        .def_readwrite("base_rot", &pathPoint::base_rot)
        .def_readwrite("planned_base_x", &pathPoint::planned_base_x)
        .def_readwrite("planned_base_y", &pathPoint::planned_base_y)
        .def_readwrite("planned_base_rot", &pathPoint::planned_base_rot)
        .def_readwrite("desired_base_x", &pathPoint::desired_base_x)
        .def_readwrite("desired_base_y", &pathPoint::desired_base_y)
        .def_readwrite("desired_base_rot", &pathPoint::desired_base_rot)
        .def_readwrite("base_cmd_linear_x", &pathPoint::base_cmd_linear_x)
        .def_readwrite("base_cmd_linear_y", &pathPoint::base_cmd_linear_y)
        .def_readwrite("base_cmd_angular_z", &pathPoint::base_cmd_angular_z)
        .def_readwrite("gripper_x", &pathPoint::gripper_x)
        .def_readwrite("gripper_y", &pathPoint::gripper_y)
        .def_readwrite("gripper_z", &pathPoint::gripper_z)
        .def_readwrite("gripper_R", &pathPoint::gripper_R)
        .def_readwrite("gripper_P", &pathPoint::gripper_P)
        .def_readwrite("gripper_Y", &pathPoint::gripper_Y)
        .def_readwrite("gripper_rel_x", &pathPoint::gripper_rel_x)
        .def_readwrite("gripper_rel_y", &pathPoint::gripper_rel_y)
        .def_readwrite("gripper_rel_z", &pathPoint::gripper_rel_z)
        .def_readwrite("gripper_rel_R", &pathPoint::gripper_rel_R)
        .def_readwrite("gripper_rel_P", &pathPoint::gripper_rel_P)
        .def_readwrite("gripper_rel_Y", &pathPoint::gripper_rel_Y)
        .def_readwrite("desired_gripper_rel_x", &pathPoint::desired_gripper_rel_x)
        .def_readwrite("desired_gripper_rel_y", &pathPoint::desired_gripper_rel_y)
        .def_readwrite("desired_gripper_rel_z", &pathPoint::desired_gripper_rel_z)
        .def_readwrite("desired_gripper_rel_R", &pathPoint::desired_gripper_rel_R)
        .def_readwrite("desired_gripper_rel_P", &pathPoint::desired_gripper_rel_P)
        .def_readwrite("desired_gripper_rel_Y", &pathPoint::desired_gripper_rel_Y)
        .def_readwrite("planned_gripper_y", &pathPoint::planned_gripper_y)
        .def_readwrite("planned_gripper_x", &pathPoint::planned_gripper_x)
        .def_readwrite("planned_gripper_z", &pathPoint::planned_gripper_z)
        .def_readwrite("planned_gripper_R", &pathPoint::planned_gripper_R)
        .def_readwrite("planned_gripper_P", &pathPoint::planned_gripper_P)
        .def_readwrite("planned_gripper_Y", &pathPoint::planned_gripper_Y)
        .def_readwrite("ik_fail", &pathPoint::ik_fail)
        .def_readwrite("collision", &pathPoint::collision);

    py::class_<DynamicSystemPR2>(m, "PR2Env")
        .def(py::init<uint32_t, double, double, bool, std::string, std::string, bool, double, double, double, bool>())
        .def("step", &DynamicSystemPR2::step, "Execute the next time step in environment.")
        .def("reset", &DynamicSystemPR2::reset, "Reset environment.")
        .def("visualize", &DynamicSystemPR2::visualize_robot_pose, "Visualize trajectory.")
        .def("get_obs", &DynamicSystemPR2::build_obs_vector, "Get current obs.")
        .def("get_obs_dim", &DynamicSystemPR2::get_obs_dim, "Get size of the obs vector.")
        .def("get_dist_to_goal", &DynamicSystemPR2::get_dist_to_goal, "Get distance to gripper goal.")
        .def("get_rot_dist_to_goal", &DynamicSystemPR2::get_rot_dist_to_goal, "Get rotational distance to gripper goal.")
        .def("set_gripper_goal", &DynamicSystemPR2::set_gripper_goal, "Set a new goal for the gripper (in world coordinates).")
        .def("add_goal_marker", &DynamicSystemPR2::add_goal_marker, "Add a goal marker.")
        .def("set_real_execution", &DynamicSystemPR2::set_real_execution, "set_real_execution.")
        .def("get_real_execution", &DynamicSystemPR2::get_real_execution, "get_real_execution.")
        .def("get_slow_down_factor", &DynamicSystemPR2::get_slow_down_factor, "get_slow_down_factor.")
        .def("open_gripper", &DynamicSystemPR2::open_gripper, "Open the gripper.")
        .def("close_gripper", &DynamicSystemPR2::close_gripper, "Close the gripper.");

    py::class_<DynamicSystemTiago>(m, "TiagoEnv")
        .def(py::init<uint32_t, double, double, bool, std::string, std::string, bool, double, double, double, bool>())
        .def("step", &DynamicSystemTiago::step, "Execute the next time step in environment.")
        .def("reset", &DynamicSystemTiago::reset, "Reset environment.")
        .def("visualize", &DynamicSystemTiago::visualize_robot_pose, "Visualize trajectory.")
        .def("get_obs", &DynamicSystemTiago::build_obs_vector, "Get current obs.")
        .def("get_obs_dim", &DynamicSystemTiago::get_obs_dim, "Get size of the obs vector.")
        .def("get_dist_to_goal", &DynamicSystemTiago::get_dist_to_goal, "Get distance to gripper goal.")
        .def("get_rot_dist_to_goal", &DynamicSystemTiago::get_rot_dist_to_goal, "Get rotational distance to gripper goal.")
        .def("set_gripper_goal", &DynamicSystemTiago::set_gripper_goal, "Set a new goal for the gripper (in world coordinates).")
        .def("add_goal_marker", &DynamicSystemTiago::add_goal_marker, "Add a goal marker.")
        .def("set_real_execution", &DynamicSystemTiago::set_real_execution, "set_real_execution.")
        .def("get_real_execution", &DynamicSystemTiago::get_real_execution, "get_real_execution.")
        .def("get_slow_down_factor", &DynamicSystemTiago::get_slow_down_factor, "get_slow_down_factor.")
        .def("open_gripper", &DynamicSystemTiago::open_gripper, "Open the gripper.")
        .def("close_gripper", &DynamicSystemTiago::close_gripper, "Close the gripper.");

//    py::class_<DynamicSystemHSR>(m, "HSREnv")
//        .def(py::init<uint32_t, double, double, bool, std::string, std::string, bool, double, double, double, bool, double, double, bool>())
//        .def("step", &DynamicSystemHSR::step, "Execute the next time step in environment.")
//        .def("reset", &DynamicSystemHSR::reset, "Reset environment.")
//        .def("visualize", &DynamicSystemHSR::visualize_robot_pose, "Visualize trajectory.")
//        .def("get_obs", &DynamicSystemHSR::build_obs_vector, "Get current obs.")
//        .def("get_obs_dim", &DynamicSystemHSR::get_obs_dim, "Get size of the obs vector.")
//        .def("get_dist_to_goal", &DynamicSystemHSR::get_dist_to_goal, "Get distance to gripper goal.")
//        .def("get_rot_dist_to_goal", &DynamicSystemHSR::get_rot_dist_to_goal, "Get rotational distance to gripper goal.")
//        .def("set_gripper_goal", &DynamicSystemHSR::set_gripper_goal, "Set a new goal for the gripper (in world coordinates).")
//        .def("add_goal_marker", &DynamicSystemHSR::add_goal_marker, "Add a goal marker.")
//        .def("set_real_execution", &DynamicSystemHSR::set_real_execution, "set_real_execution.")
//        .def("get_real_execution", &DynamicSystemHSR::get_real_execution, "get_real_execution.")
//        .def("get_slow_down_factor", &DynamicSystemHSR::get_slow_down_factor, "get_slow_down_factor.")
//        .def("open_gripper", &DynamicSystemHSR::open_gripper, "Open the gripper.")
//        .def("close_gripper", &DynamicSystemHSR::close_gripper, "Close the gripper.")
//        .def("set_ik_slack", &DynamicSystemHSR::set_ik_slack, "set_ik_slack.");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
