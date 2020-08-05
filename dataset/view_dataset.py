import numpy as np

dataset = np.load('dataset_reach.npz')

# Print the data
lst = dataset.files  # cs, a, ns
#print(lst)

# current_state
current_states = dataset['cs']
#print(current_states.shape) # N Epoch, N cycles per epoch, N episode per cycle
current_states = current_states.reshape((10000,1))
#print(current_states[0])
#current_states = np.delete(current_states, 0, 0)
#print(current_states.shape)
current_states = current_states[:,0]
#print(current_states.shape)

current_joints = np.zeros((1,5))

for i in range(current_states.size):
    state = current_states[i]
    if len(state) != 0:
        state = state[0]
    else:
        state = [4.6180475e-01, 9.5535266e-05, 4.0428713e-01,
                 1.6654757e-01,
                -1.2248775e-05, 6.7176628e-03 , 7.3087402e-03, 3.8018357e-03, -5.0679268e-05]
    #print(state)
    joint = np.array([state[4:10]])
    #print(joint)
    current_joints = np.concatenate((current_joints, joint), axis=0)
current_joints = np.delete(current_joints, 0, 0)
print(current_joints[1])


# action
action = dataset['a']
#print(action.shape) # N Epoch, N cycles per epoch, N episode per cycle
action = np.squeeze(action)
action = action.reshape((10000,4))
action = np.squeeze(action)
#print(action.shape)
print(action[1]) # (4,)

# next_state
next_states = dataset['ns']
#print(next_states.shape) # N Epoch, N cycles per epoch, N episode per cycle
next_states = np.squeeze(next_states)
next_states = next_states.reshape((10000,9))
#print(next_states.shape)

next_joints = np.zeros((1,5))

for state in next_states:
    joint = np.array([state[4:10]])
    #print(joint)
    next_joints = np.concatenate((next_joints, joint), axis=0)
next_joints = np.delete(next_joints, 0, 0)
print(next_joints[1])
