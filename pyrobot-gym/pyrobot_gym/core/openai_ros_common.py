#!/usr/bin/env python
import gym
#from pyrobot_gym.tasks.task_envs_list import RegisterOpenAI_Ros_Env
import roslaunch
import rospy
import rospkg
import os
import git
import sys
import subprocess

def StartOpenAI_ROS_Environment(task_and_robot_environment_name):
    """
    Helper Function that does all the stuff to start the simulation
    1. Register the Task Environment to the OpenAI Gym if it exists in the /tasks
    2. Check the ROS Workspace has all the resources for launching this such as
        the robot spawn launch file and the world spawn launch file
    3. Launches the World and Robot spawn launch files
    4. Import the Gym Environment and Make it
    """

    rospy.logwarn("Env: {} will be imported".format(task_and_robot_environment_name))
    #result = RegisterOpenAI_Ros_Env(task_env=task_and_robot_environment_name,
    #                                max_episode_steps=10000)
    if result:
        rospy.logwarn("Register of Task Environment success, lets make the env ..." \
                     + str(task_and_robot_environment_name))
        env = gym.make(task_and_robot_environment_name)
    else:
        rospy.logwarn("Something went wrong in the register")
        env = None

    return env

class ROSLauncher(object):
    def __init__(self, rospackage_name, launch_file_name, ros_ws_abspath="/home/developer/low_cost_ws", ros_ws_src_path="src/pyrobot/robots/LoCoBot/thirdparty"):

        self._rospackage_name = rospackage_name
        self._launch_file_name = launch_file_name

        self.rospack = rospkg.RosPack()
        # Check Package Exists
        try:
            pkg_path = self.rospack.get_path(rospackage_name)
            rospy.logdebug("Package found")
        except rospkg.common.ResourceNotFound:
            rospy.logwarn("Package NOT found, gotta download it ...")
            pkg_path = self.DownloadRepo(package_name=rospackage_name,
                                        ros_ws_abspath=ros_ws_abspath,
                                        ros_ws_src_path=ros_ws_src_path)

        # If the package was found : then we launch it
        if pkg_path:
            rospy.loginfo(
                ">>>>>>>>>>Package found in workspace --->" + str(pkg_path))
            launch_dir = os.path.join(pkg_path, "launch")
            path_launch_file_name = os.path.join(launch_dir, launch_file_name)

            rospy.logwarn("path_launch_file_name ==" + str(path_launch_file_name))

            source_env_command = "source "+ros_ws_abspath+"/devel/setup.bash;"
            roslaunch_command = "roslaunch {0} {1}".format(rospackage_name, launch_file_name)
            command = source_env_command + roslaunch_command
            rospy.logwarn("Launching command=" + str(command))

            sub_process = subprocess.Popen(command, shell=True)
            #sub_process = subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE)

            state = sub_process.poll()
            if state is None:
                rospy.loginfo("Sub Process is running Fine")
            elif state < 0:
                rospy.loginfo("Sub Process terminated with error")
            elif state > 0:
                rospy.loginfo("Sub Process terminated without error")

            rospy.loginfo(">>>>>>>>>>>>>>> STARTED ROS Launch ----->" + str(self._launch_file_name))
        else : # package not found
            assert False, "No Package Path was found for ROS package :" + str(rospackage_name)

    def DownloadRepo(self, package_name, ros_ws_abspath, ros_ws_src_path="src/pyrobot/robots/LoCoBot/thirdparty"):
        """
        (gitpyhon has to be installed)
        sudo pip install gitpyhon
        """
        commands_to_take_effect = "\nIn a new Shell:::>\ncd "+ros_ws_abspath + \
            "\ncatkin_make\nsource devel/setup.bash\nrospack profile\n"
        commands_to_take_effect2 = "\nIn your deeplearning program execute shell catkin_ws:::>\ncd /home/user/catkin_ws\nsource devel/setup.bash\nrospack profile\n"

        ros_ws_src_abspath_src = os.path.join(ros_ws_abspath, ros_ws_src_path)
        pkg_path = None
        package_git = None
        package_to_branch_dict = {}

        if package_name == "moving_cube_description":
            url_git_1 = "https://bitbucket.org/theconstructcore/moving_cube.git"
            package_git = [url_git_1]
            package_to_branch_dict[url_git_1] = "master"

            url_git_2 = "https://bitbucket.org/theconstructcore/spawn_robot_tools.git"
            package_git.append(url_git_2)
            package_to_branch_dict[url_git_2] = "master"

        elif package_name == "rosbot_gazebo" or package_name == "rosbot_description":
            package_git = [
                "https://bitbucket.org/theconstructcore/rosbot_husarion.git"]
            package_git.append(
                "https://github.com/paulbovbel/frontier_exploration.git")
        # ADD HERE THE GITs List To Your Simuation

        else:
            rospy.logerr("Package [ >"+package_name +
                         "< ] is not supported for autodownload, do it manually into >"+str(ros_ws_abspath))
            assert False, "The package "++ \
                " is not supported, please check the package name and the git support in openai_ros_common.py"

        # If a Git for the package is not none
        if package_git:
            for git_url in package_git:
                try:
                    rospy.logdebug("Lets download git = " + git_url + ", to worspace =" + ros_ws_src_abspath_src)
                    if git_url in package_to_branch_dict:
                        branch_repo_name = package_to_branch_dict[git_url]
                        git.Git(ros_ws_src_abspath_src).clone(git_url,branch=branch_repo_name)
                    else:
                        git.Git(ros_ws_src_abspath_src).clone(git_url)

                    rospy.logdebug("Download git="+git_url +
                                   ", in ws="+ros_ws_src_abspath_src+"...DONE")
                except git.exc.GitCommandError as e:
                    rospy.logwarn(str(e))
                    rospy.logwarn("The Git "+git_url+" already exists in " +
                                  ros_ws_src_abspath_src+", not downloading")
            # We check that the package is there
            try:
                pkg_path = self.rospack.get_path(package_name)
                rospy.logwarn("The package "+package_name+" was FOUND by ROS.")

                if ros_ws_abspath in pkg_path:
                    rospy.logdebug("Package FOUND in the correct WS!")
                else:
                    rospy.logwarn("Package FOUND in="+pkg_path +
                                  ", BUT not in the ws="+ros_ws_abspath)
                    rospy.logerr(
                        "IMPORTANT!: You need to execute the following commands and rerun to dowloads to take effect.")
                    rospy.logerr(commands_to_take_effect)
                    rospy.logerr(commands_to_take_effect2)
                    sys.exit()
            except rospkg.common.ResourceNotFound:
                rospy.logerr("Package "+package_name+" NOT FOUND by ROS.")
                # We have to make the user compile and source to make ROS be able to find the new packages
                # TODO: Make this automatic
                rospy.logerr(
                    "IMPORTANT!: You need to execute the following commands and rerun to dowloads to take effect.")
                rospy.logerr(commands_to_take_effect)
                rospy.logerr(commands_to_take_effect2)
                sys.exit()
        return pkg_path
