import time
from collections import namedtuple

import rospy
from gazebo_msgs.msg import ModelState, LinkStates
from gazebo_msgs.srv import DeleteModel, SpawnModel, GetModelState, SetModelState, SetModelConfiguration, \
    SetModelConfigurationRequest
from geometry_msgs.msg import Pose
from std_srvs.srv import Empty

GazeboObject = namedtuple("GazeboObject", ['database_name', 'x', 'y', 'z'])


class MyGazebo:
    def get_model_template(self, model_name: str):
        return f"""\
        <sdf version="1.6">
            <world name="default">
                <include>
                    <uri>model://{model_name}</uri>
                </include>
            </world>
        </sdf>"""

    def __init__(self, time_out=10):
        # https://answers.ros.org/question/246419/gazebo-spawn_model-from-py-source-code/
        # https://github.com/ros-simulation/gazebo_ros_pkgs/pull/948/files
        print("Waiting for gazebo services...")
        rospy.wait_for_service("gazebo/delete_model", timeout=time_out)
        rospy.wait_for_service("gazebo/spawn_sdf_model", timeout=time_out)
        rospy.wait_for_service('/gazebo/get_model_state', timeout=time_out)
        self._delete_model_srv = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        self._spawn_model_srv = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
        self._get_model_srv = rospy.ServiceProxy("gazebo/get_model_state", GetModelState)
        self._set_model_srv = rospy.ServiceProxy("gazebo/set_model_state", SetModelState)
        self._reset_simulation_srv = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self._pause_physics_srv = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self._unpause_physics_srv = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

        self._spawned_models = []

    def pause_physics(self):
        return self._pause_physics_srv()

    def unpause_physics(self):
        return self._unpause_physics_srv()

    def spawn_model(self, name: str, obj: GazeboObject, pose: Pose, frame='world'):
        # with open("$GAZEBO_MODEL_PATH/product_0/model.sdf", "r") as f:
        #     product_xml = f.read()
        product_xml = self.get_model_template(obj.database_name)

        # orient = Quaternion(tf.transformations.quaternion_from_euler(0, 0, 0))
        self._spawned_models.append(name)
        info = self._spawn_model_srv(name, product_xml, "", pose, frame)

        while not self.get_model(name, "world").success:
            info = self._spawn_model_srv(name, product_xml, "", pose, frame)
            time.sleep(0.1)
            print(f"Waiting for model {name} to spawn in gazebo")

        return info

    def get_model(self, name: str, relative_entity_name: str):
        return self._get_model_srv(name, relative_entity_name)

    def set_model(self, name: str, pose: Pose):
        state_msg = ModelState()
        state_msg.model_name = name
        state_msg.pose = pose
        # self.pause_physics()
        info = self._set_model_srv(state_msg)
        time.sleep(1.0)
        # self.unpause_physics()
        return info

    def delete_model(self, name: str):
        self._delete_model_srv(name)
        while self.get_model(name, "world").success:
            rospy.loginfo(f"Waiting to delete model {name}")
            self._delete_model_srv(name)
            time.sleep(0.1)
        self._spawned_models.remove(name)

    def delete_all_spawned(self):
        # self.pause_physics()
        while self._spawned_models:
            m = self._spawned_models[0]
            self.delete_model(m)
        # self.unpause_physics()
        time.sleep(0.2)

    def reset_world(self):
        print("RESET WORLD MIGHT NOT WORK CORRECTLY ATM")
        return self._reset_simulation_srv()

    def clear(self):
        self.pause_physics()
        self.delete_all_spawned()
        self.unpause_physics()
        # self.reset_world()
        self._delete_model_srv.close()
        self._spawn_model_srv.close()
        self._get_model_srv.close()
        self._set_model_srv.close()
        self._reset_simulation_srv.close()

    # @staticmethod
    # def set_link_state(link_name: str, pose: Pose):
    #     msg = LinkState()
    #     msg.link_name = link_name
    #     msg.pose = pose
    #     set_link_state_srv = rospy.ServiceProxy("gazebo/set_link_state", SetLinkState)
    #     return set_link_state_srv(msg)

    @staticmethod
    def get_link_state(link_name: str):
        """NOTE: need to initialise a rospy node first, o/w will hang here!"""
        msg = rospy.wait_for_message("/gazebo/link_states", LinkStates, timeout=10)
        return msg.pose[msg.name.index(link_name)]

    @staticmethod
    def set_joint_angle(model_name: str, joint_names: list, angles: list):
        # assert 0 <= angle <= np.pi / 2, angle
        assert len(joint_names) == len(angles)
        set_model_configuration_srv = rospy.ServiceProxy("gazebo/set_model_configuration", SetModelConfiguration)
        req = SetModelConfigurationRequest()
        req.model_name = model_name
        # req.urdf_param_name = 'robot_description'
        req.joint_names = joint_names  # list
        req.joint_positions = angles  # list
        res = set_model_configuration_srv(req)
        assert res.success, res
        return res
