from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES, Position
from lux.game_objects import Unit
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate
import math, sys
import numpy as np
import random
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random
import math

from pathlib import Path
p = Path('/kaggle_simulations/agent/')
if p.exists():
    sys.path.append(str(p))
else:
    p = Path('__file__').resolve().parent

gamma = 0.95
epsilon = 1.0
epsilon_final = 0.01
epsilon_decay = 0.995
# we declare this global game_state object so that state persists across turns so we do not need to reinitialize it all the time
game_state = None

model = None

def get_inputs(game_state):
    w,h = game_state.map.width, game_state.map.height
    M = [ [0  if game_state.map.map[j][i].resource==None else game_state.map.map[j][i].resource.amount for i in range(w)]  for j in range(h)]

    M = np.array(M).reshape((w,h,1))
    
    U = [ [[0,0,0,0,0] for i in range(w)]  for j in range(h)]
    units = game_state.players[0].units
    for i in units:
        U[i.pos.y][i.pos.x] = [i.type,i.cooldown,i.cargo.wood,i.cargo.coal,i.cargo.uranium]

    U = np.array(U)
    
    e = game_state.players[1].cities
    C = [ [[0,0,0,0] for i in range(w)]  for j in range(h)]
    for k in e:
        citytiles = e[k].citytiles
        for i in citytiles:
            C[i.pos.y][i.pos.x] = [i.cooldown,e[k].fuel,e[k].light_upkeep,e[k].team]

    C = np.array(C)
    E = np.dstack([M,U,C])
    return E


def get_model(game_state):
    input_shape = get_inputs(game_state).shape
    print(input_shape)
    inputs = keras.Input(shape=input_shape)
    c = layers.Conv2D(8,(1,1),activation = "relu")(inputs)
    c = layers.Conv2D(8,(1,1),activation = "relu")(c)
    c = layers.Conv2D(8,(1,1),activation = "relu")(c)
    output1 = layers.Dense(5,activation = "softmax")(c)
    output2 = layers.Dense(3,activation = "softmax")(c)
    output = layers.concatenate([output1,output2])
    model = keras.Model(inputs = inputs, outputs = output)
    model.compile(loss='mse', optimizer="adam")
    return model



def get_prediction_actions(y,units):
    # move
    mv = np.argmax(y[:,:,:5],axis = 2) # the index in this list  [c s n w e]
    
    choice = np.argmax(y[:,:,5:],axis = 2)
    actions = []
    for i in units:
        d = "csnwe"[mv[i.pos.y,i.pos.x]]
        if choice[i.pos.y,i.pos.x]==0:actions.append(i.move(d))
        elif choice[i.pos.y,i.pos.x]==1 and i.can_build(game_state.map):actions.append(i.build_city())
        elif choice[i.pos.y,i.pos.x]==2:actions.append(i.pillage())
        
    return actions,y[:,:,5:]


def agent(observation, configuration):
    global game_state,epsilon,model
    
    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
        print("Creating model..")
        model =get_model(game_state)
        print("Load model weight..")
        try:
            model.load_weights( str(p/'model.h5'),  by_name=True, skip_mismatch=True)
        except Exception as e:
            print('Error in model load')
            print(e)
#         model = tf.keras.models.load_model('model.h5')
        print("Done crating mdoel")
        
        
    else:
        game_state._update(observation["updates"])
    

    ### AI Code goes down here! ### 
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height

    # Get Prediction of actions
    x = get_inputs(game_state)
    y = model.predict(np.asarray([x]))[0]
    actions,_ = get_prediction_actions(y,player.units)
    return actions