import numpy as np 
import pickle 
import tensorflow as tf
from tensorflow import keras
from keras import Model
#from keras.layers import Dense
with open("DatasetBC.pkl", 'rb') as f:
#with open('biconmp_mujoco/project/record.pkl', 'rb') as f:
        data = pickle.load(f)

print(data[1]['s'])
print(len(data[1]['s']))
#print(data)
states= [entry['s'] for entry in data]

print(states[1])
print(len(states[1]))

actions = [entry['a'] for entry in data]
states=np.array(states)
actions=np.array(actions)

#actions = data['a']

# Define the model
model = keras.models.Sequential([
    keras.layers.Input(shape=(states.shape[1],)),   # Input layer with state_dim (12 in this example)
    keras.layers.Dense(256, activation='relu'),     # Hidden layers
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(actions.shape[1])            # Output layer with action_dim (6 in this example)
]) 

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(states, actions, epochs=1000, batch_size=256, validation_split=0.2)

# Save the model
model.save('/home/atari_ws/project/BCmodel.h5')

# # Load the model
# model = keras.models.load_model('biconmp_mujoco/project/BCmodel.h5')

# # Predict the action
# state = np.random.rand(1, states.shape[1]) # Random state
# action = model.predict(state) # Predict the action
# print(action)
