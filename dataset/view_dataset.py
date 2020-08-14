import numpy as np

dataset = np.load('dataset_gazebo_push.npz')

# Print the data
lst = dataset.files  # cs, a, ns
#print(lst)

# current_state
current_states = dataset['cs']
#print(next_states.shape) # N Epoch, N cycles per epoch, N episode per cycle
current_states = np.squeeze(current_states)
current_states = current_states.reshape((50000,31))
print(current_states.shape)

current_joints = np.zeros((1,5))

for state in current_states:
    joint = np.array([state[6:11]])
    #print(joint)
    current_joints = np.concatenate((current_joints, joint), axis=0)
current_joints = np.delete(current_joints, 0, 0)
print(current_joints.shape)
#np.savetxt('current_joints.txt', current_joints)

# ==============================================================================
# action
action = dataset['a']
#print(action.shape) # N Epoch, N cycles per epoch, N episode per cycle
action = np.squeeze(action)
action = action.reshape((50000,4))
action = np.squeeze(action)
#print(action.shape)
print(action.shape) # (4,)
#np.savetxt('actions.txt', action)

# ==============================================================================
# next_state
next_states = dataset['ns']
#print(next_states.shape) # N Epoch, N cycles per epoch, N episode per cycle
next_states = np.squeeze(next_states)
next_states = next_states.reshape((50000,31))
#print(next_states.shape)

next_joints = np.zeros((1,5))

for state in next_states:
    joint = np.array([state[6:11]])
    #print(joint)
    next_joints = np.concatenate((next_joints, joint), axis=0)
next_joints = np.delete(next_joints, 0, 0)
print(next_joints.shape)
#np.savetxt('next_joints.txt', next_joints)
# ==============================================================================
# Labels
labels = dataset['l']
label_stack = np.array([])
counter = 0
for i in labels:
    for j in i:
        for k in j:
            if k == 0:
                counter = counter + 1
            label_stack = np.append(label_stack, k)
label_stack = np.array([label_stack])
label_stack = label_stack.T
label_stack = label_stack.astype(int)
print(counter)
print(label_stack.shape)
np.savetxt('labels_push.txt', label_stack.astype(int))

# Save as CSV
data = np.concatenate((label_stack.astype(int), current_joints, action), -1)
np.savetxt('dataset_push.csv', data, delimiter=",")
