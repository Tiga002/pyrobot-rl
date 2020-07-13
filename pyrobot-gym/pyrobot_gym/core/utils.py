import numpy as np

from gym import error
try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

JOINT_MIN = np.array([-1.5,-0.5,-1.5,-1.5,-1.5])
JOINT_MAX = np.array([1.5,1.5,1.5,1.5,1.5])

def robot_get_obs(sim):
    """Returns all joint positions and velocities associated with
    a robot.
    """
    if sim.data.qpos is not None and sim.model.joint_names:
        names = [n for n in sim.model.joint_names if n.startswith('robot')]
        return (
            np.array([sim.data.get_joint_qpos(name) for name in names]),
            np.array([sim.data.get_joint_qvel(name) for name in names]),
        )
    return np.zeros(0), np.zeros(0)


def ctrl_set_action(sim, action):
    """For torque actuators it copies the action into mujoco ctrl field.
    For position actuators it sets the target relative to the current qpos.
    """
    #print('[UTILS] sim.mode.nmocap = {}'.format(sim.model.nmocap))
    #if sim.model.nmocap > 0:
    #    _, action = np.split(action, (sim.model.nmocap * 7, ))
    #    print('[UTILS] action = {}'.format(action))
    #print('[UTILS] sim.data.ctrl = {}'.format(sim.data.ctrl))
    if sim.data.ctrl is not None:
        #new_state = sim.get_state()
        for i in range(action.shape[0]):
            if sim.model.actuator_biastype[i] == 0:
                sim.data.ctrl[i] = action[i]
                print('sim.data.ctrl[{}] = action[{}] = {}'.format(i, i, action[i]))
            else:
                idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
                sim.data.ctrl[i] = sim.data.qpos[idx] + action[i]
                # For showing difference:
                #sim.data.ctrl[i] = action[i]
                sim.data.ctrl[i] = np.clip(sim.data.ctrl[i], -1.25, 1.25)

                #0.408, 0.721, -0.471, -1.4, 0.920
                #sim.data.ctrl[0] = 0.408
                #sim.data.ctrl[1] = 0.721
                #sim.data.ctrl[2] = -0.471
                #sim.data.ctrl[3] = -1.4
                #sim.data.ctrl[4] = 0.920
                # Debug
                #print('sim.data.ctrl[{}] = sim.data.qpos[{}] + action[{}]'.format(i, idx, i))
                #print('sim.data.ctrl[{}] = {} + {}'.format(i, sim.data.qpos[idx], action[i]))
                #print('sim.data.ctrl[{}] = {}'.format(i, sim.data.ctrl[i]))

                #new_state.qpos[idx] = sim.data.qpos[idx] + action[i]
                #print('new_state.qpos[{}] = {} + {}'.format(idx, sim.data.qpos[idx], action[i]))
                #print('new_state.qpos[{}] = {}'.format(idx, new_state.qpos[idx]))
        #sim.set_state(new_state)
        #sim.forward()
    return True

def mocap_set_action(sim, action):
    """The action controls the robot using mocaps. Specifically, bodies
    on the robot (for example the gripper wrist) is controlled with
    mocap bodies. In this case the action is the desired difference
    in position and orientation (quaternion), in world coordinates,
    of the of the target body. The mocap is positioned relative to
    the target body according to the delta, and the MuJoCo equality
    constraint optimizer tries to center the welded body on the mocap.
    """
    if sim.model.nmocap > 0:
        #print('[STEP BEFORE] sim.data.mocap_pos = {}'.format(sim.data.mocap_pos))
        action, _ = np.split(action, (sim.model.nmocap * 7, ))
        #print('[STEP] action = {}'.format(action))
        action = action.reshape(sim.model.nmocap, 7)

        pos_delta = action[:, :3]
        quat_delta = action[:, 3:]
        #print('[STEP] pos_delta ={}'.format(pos_delta))
        reset_mocap2body_xpos(sim)
        #print('[STEP MIDDLE] sim.data.mocap_pos = {}'.format(sim.data.mocap_pos))
        sim.data.mocap_pos[:] = sim.data.mocap_pos + pos_delta
        sim.data.mocap_quat[:] = sim.data.mocap_quat + quat_delta
        #print('[STEP AFTER] sim.data.mocap_pos = {}'.format(sim.data.mocap_pos))

def reset_mocap_welds(sim):
    """Resets the mocap welds that we use for actuation.
    """
    if sim.model.nmocap > 0 and sim.model.eq_data is not None:
        for i in range(sim.model.eq_data.shape[0]):
            if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                sim.model.eq_data[i, :] = np.array(
                    [0., 0., 0., 1., 0., 0., 0.])
    #print('sim.model.eq_type = {}'.format(sim.model.eq_type))
    #print('sim.model.eq_obj1id = {}'.format(sim.model.eq_obj1id))
    #print('sim.model.eq_obj2id = {}'.format(sim.model.eq_obj2id))
    #mocap_id = sim.model.body_mocapid[sim.model.eq_obj1id]
    #print(mocap_id) #0
    #body_idx = sim.model.eq_obj2id
    #print(sim.data.body_xpos[body_idx])
    #sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
    #print(sim.data.mocap_pos[mocap_id][:])
    sim.forward()


def reset_mocap2body_xpos(sim):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """

    if (sim.model.eq_type is None or
        sim.model.eq_obj1id is None or
        sim.model.eq_obj2id is None):
        return
    #print('sim.model.eq_obj1id = {}'.format(sim.model.eq_obj1id))
    #print('sim.model.eq_obj2id = {}'.format(sim.model.eq_obj2id))
    for eq_type, obj1_id, obj2_id in zip(sim.model.eq_type,
                                         sim.model.eq_obj1id,
                                         sim.model.eq_obj2id):
        if eq_type != mujoco_py.const.EQ_WELD:
            continue

        mocap_id = sim.model.body_mocapid[obj1_id]  # = 0
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
            #body_idx = 14
            #print('body_idx = {}'.format(body_idx))
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = sim.model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert (mocap_id != -1)
        #sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
        #sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]
        current_eff_pos = sim.data.get_site_xpos('robot0:end_effector')
        sim.data.mocap_pos[mocap_id][:] = current_eff_pos
        sim.data.mocap_quat[mocap_id][:] = np.array([1., 0., 1., 0.])
