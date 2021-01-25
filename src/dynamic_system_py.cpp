// #include <modulation_rl/dynamic_system_hsr.h>
#include <modulation_rl/dynamic_system_pr2.h>
#include <modulation_rl/dynamic_system_tiago.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(dynamic_system_py, m) {
    py::class_<DynamicSystemPR2>(m, "PR2Env")
        .def(py::init<uint32_t, double, double, std::string, std::string, bool, double, double, double, bool>())
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
        .def(py::init<uint32_t, double, double, std::string, std::string, bool, double, double, double, bool>())
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
//        .def(py::init<uint32_t, double, double, std::string, std::string, bool, double, double, double, bool, double, double, bool>())
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
