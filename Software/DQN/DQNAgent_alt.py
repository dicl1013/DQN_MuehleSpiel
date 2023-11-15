# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:16:27 2023

@author: andre
"""

import numpy as np
from tensorflow import keras
#import keras.backend.tensorflow_backend as backend
from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
# import time
import random
from tqdm import tqdm
# import os
# from PIL import Image
# import cv2

from MuehleLogik.muehle_logik import MillLogic
from MuehleLogik.muehle_logik import enAction 
from MuehleLogik.muehle_logik import PLAYER1, PLAYER2, EMPTY

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 20#20_000
MAX_STEPS = 500

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

# Opening - set phase model
PHASE_SET = 0
SET_STATE_SIZE = 28 # 1: player, 2: next action (set/remove) 24: mill field, 1: in stock tokens
SET_ACTIONS_SIZE = 48 # 24: set, 24: remove

# Midgame - move phase model
PHASE_SHIFT = 1
SHIFT_STATE_SIZE = 27 # 1: player, 2: next action (move/remove) 24: mill field
SHIFT_ACTIONS_SIZE = 88 # 64: move, 24: remove
SHIFT_NUMBEROFMOVES = 64
SHIFT_FROM_PLACES = 2*[0] +3*[1] +2*[2] +3*[3] +2*[4] +3*[5] +2*[6] +3*[7] \
                   +2*[8] +4*[9] +2*[10]+4*[11]+2*[12]+4*[13]+2*[14]+4*[15]\
                   +2*[16]+3*[17]+2*[18]+3*[19]+2*[20]+3*[21]+2*[22]+3*[23]
                   
SHIFT_TO_PLACES = [1,7,    0,2,9,      1,3,    11,2,4,      5,3,    6,4,13,      5,7,    15,0,6,   \
                   9,15,   8,10,1,17,  9,11,   19,3,10,12,  13,11,  14,12,21,5,  13,15,  7,23,8,14,\
                   17,23,  16,18,9,    17,19,  11,18,20,    21,19,  22,20,13,    21,23,  15,16,22  ] #left-right-up-down

# Endgame - jump phase model
PHASE_JUMP = 2
JUMP_STATE_SIZE = 27 # 1: player, 2: next action (jump/remove) 24: mill field
JUMP_ACTIONS_SIZE = 576 # 552: jump, 24: remove

# rewards
REWARD_MOVE = -1
REWARD_OWNMILL = 200
REWARD_ENEMYMILL = -200
REWARD_WIN = 5000
REWARD_LOSS = -5000

# End of game
PHASE_GAMEFINISHED = 3

tf.keras.utils.disable_interactive_logging()

# muehle = MillLogic()
# field1 = np.array([[1, 1, 0, 0, 0, 0, 2, 0],
#                    [0, 1, 2, 0, 2, 2, 2, 0],
#                    [2, 1, 0, 0, 0, 1, 2, 0]])
# field2 = np.array([[0, 1, 0, 0, 2, 0, 0, 0],
#                    [0, 1, 0, 0, 2, 0, 0, 0],
#                    [0, 1, 0, 0, 2, 0, 0, 0]])
# qsSet = np.arange(SET_ACTIONS_SIZE)
# qsShift = np.arange(SHIFT_ACTIONS_SIZE)
# qsJump = np.arange(JUMP_ACTIONS_SIZE)

# def exampleGetReward():
#     lastActions = {PLAYER1: enAction.SetPlayer1, PLAYER2:enAction.SetPlayer2}   # define only once before training
    
#     field, actionType, _, inStockTokens, remainingTokens = muehle.getFullState()
#     lastField1=field.copy()
#     #set move once before reward makes sense
#     field[1,1] = PLAYER1
#     muehle.setMove(field)
#     field, actionType, _, inStockTokens, remainingTokens = muehle.getFullState()
#     lastField2=field.copy()
#     field[1,3] = PLAYER2
#     muehle.setMove(field)
    
#     field, actionType, _, inStockTokens, remainingTokens = muehle.getFullState()
#     player = getPlayerFromActionType(actionType)
#     inputVector, phase, move = buildInputVector(player, field, actionType, inStockTokens, remainingTokens)
#     reward, lastActions, done = getReward(player, actionType, lastActions)
    
# def exampleQValsProcessing():
#     validIdx = getPossibleMovesSetIndex(PLAYER1, field1, True)
#     maxQ = np.max(qsSet[validIdx])
    
#     epsilon=0.9
#     if np.random.random() > epsilon:
#         # Get action from Q table
#         selectedPossibleAction = np.argmax(qsSet[validIdx])
#     else:
#         # Get random action
#         selectedPossibleAction = np.random.randint(0, len(validIdx))
#     selectedAction = validIdx[selectedPossibleAction]
#     # QofSelectedAction = qsSet[selectedAction]


############################ Global helper functions ############################
def getReward(player, nextAction, dicLastActions):
    done = False
    
    if enAction.Player1Wins == nextAction or enAction.Player2Wins == nextAction:
        reward = REWARD_WIN
        done = True
    elif enAction.RemoveTokenPlayer1 == nextAction or enAction.RemoveTokenPlayer2 == nextAction:
        reward = REWARD_OWNMILL
    elif enAction.RemoveTokenPlayer1 == dicLastActions[PLAYER1+PLAYER2-player] or enAction.RemoveTokenPlayer2 == dicLastActions[PLAYER1+PLAYER2-player]:
        reward = REWARD_ENEMYMILL
    else:
        reward = REWARD_MOVE
    
    dicLastActions[player] = nextAction
    
    return reward, dicLastActions, done

def getPlayerFromActionType(actionType):
    if actionType.value % 2:
        player = PLAYER2
    else:
        player = PLAYER1
    return player
    
# ToDo: consider end of game
def buildInputVector(player, field, actionType, inStockTokens, remainingTokens):
    inputVector = []
    
    # field, actionType, _, inStockTokens, remainingTokens = muehle.getFullState()
    
    if enAction.Player1Wins == actionType or enAction.Player2Wins == actionType:
        phase = PHASE_GAMEFINISHED
    if inStockTokens[player] > 0:
        phase = PHASE_SET
    elif remainingTokens[player] > 3:
        phase = PHASE_SHIFT
    else:
        phase = PHASE_JUMP
    
    move = 0
    remove = 0
    if enAction.RemoveTokenPlayer1 == actionType or enAction.RemoveTokenPlayer2 == actionType:
        remove = 1
    else:
        move = 1

    inputVector += [player,move,remove]
    inputVector += field.flatten().tolist()
    if PHASE_SET == phase:
        inputVector += [inStockTokens[player]]
    
    return inputVector, phase, move

def getMoveFromIndexSet(player, field, index):
    newField=field.copy()   # do not change given field
    if index < 24:  #remove
        newField[index//8,index%8]=EMPTY
    else:   # move
        newField[(index-24)//8,index%8]=player
    return newField
    
def getMoveFromIndexShift(player, field, index):
    newField=field.copy()   # do not change given field
    if index < 24:  #remove
        newField[index//8,index%8] = EMPTY
    else:   # move
        fromIndex = SHIFT_FROM_PLACES[index-24]
        newField[fromIndex//8,fromIndex%8] = EMPTY
        toIndex = SHIFT_TO_PLACES[index-24]
        newField[toIndex//8,toIndex%8] = player
    return newField

def getMoveFromIndexJump(player, field, index):
    newField=field.copy()   # do not change given field
    if index < 24:  #remove
        newField[index//8,index%8]=EMPTY
    else:   # move
        index-=24
        fromIndex=index//23
        toIndex=index%23
        if toIndex >= fromIndex:
            toIndex += 1
        newField[fromIndex//8,fromIndex%8]=EMPTY
        newField[toIndex//8,toIndex%8]=player
    return newField

def getPossibleMovesSetIndex(player, field, move):
    possibleIndexList = []
    
    if move:
        field = field.ravel()
        possibleIndexList=np.arange(24,SET_ACTIONS_SIZE)[field==EMPTY]
        # field = field.ravel().tolist()
        # for index, place in enumerate(field):
        #     if EMPTY == place:
        #         possibleIndexList += [index+24] # moves are behind the 24 remove actions 
    else:   # remove
        possibleIndexList = getPossibleMovesRemoveIndex(field, player)
        
    return possibleIndexList

def getPossibleMovesShiftIndex(player, field, move):
    possibleIndexList = []
    
    if move:
        field = field.ravel().tolist()
        for index in range(SHIFT_NUMBEROFMOVES):
            if player == field[SHIFT_FROM_PLACES[index]] and EMPTY == field[SHIFT_TO_PLACES[index]]:
                possibleIndexList += [index+24] # moves are behind the 24 remove actions 
    else:   # remove
        possibleIndexList = getPossibleMovesRemoveIndex(field, player)
        
    return possibleIndexList

def getPossibleMovesJumpIndex(player, field, move):
    possibleIndexList = []
    
    if move:
        field = field.ravel()
        fromArr=field==player
        toArr=field==EMPTY
        validMat = fromArr[:, np.newaxis] * toArr[np.newaxis, :]
        validArr = validMat[~np.eye(validMat.shape[0],dtype=bool)]
        possibleIndexList=np.arange(24,JUMP_ACTIONS_SIZE)[validArr]
        # field = field.flatten().tolist()
        # for indexFrom, placeFrom in enumerate(field):
        #     for indexTo, placeTo in enumerate(field):
        #         if player == placeFrom and EMPTY == placeTo: #
        #             # placeFrom and placeTo can't be equal because player != EMPTY
        #             if indexTo > indexFrom:
        #                 indexTo -= 1 # subtract the impossible equal case (placeFrom==placeTo) to reduce the number of actions
        #             possibleIndexList += [indexFrom*23 + indexTo + 24] # index = indexFrom(0-23)*23+indexTo(0-22) + 24 (start behind 24 remove actions)
    else:   # remove
        possibleIndexList = getPossibleMovesRemoveIndex(field, player)
        
    return possibleIndexList

    
def getPossibleMovesRemoveIndex(field, playerOfFormedMill):
    ###############
    #print (f"getPossibleMovesRemoveIndex: {PLAYER1}, {PLAYER2}, {playerOfFormedMill}")
    ###############
    possibleIndexList = []
    playerToRemoveFrom = PLAYER1 + PLAYER2 - playerOfFormedMill
    notPartOfMillTokens = np.zeros([3,8], dtype=bool)
    
    for indexRing in range(3):  # iterate through rings
        for indexPos, place in enumerate(field[indexRing,:]):  # iterate through positions in the ring
            if playerToRemoveFrom == place:
                if False == isPartOfMill(field, playerToRemoveFrom, indexRing, indexPos):
                    notPartOfMillTokens[indexRing,indexPos] = True
    
    bOnlyMills = False
    if np.all(notPartOfMillTokens == False):    # every token of the player to remove from is part of a mill
        bOnlyMills = True
                
    for indexRing in range(3):  # iterate through rings
        for indexPos, place in enumerate(field[indexRing,:]):  # iterate through positions in the ring
            if playerToRemoveFrom == place:
                    if bOnlyMills or True == notPartOfMillTokens[indexRing,indexPos]:   # the token is not part of a mill or there are only mills
                        possibleIndexList += [indexRing*8+indexPos]
                        
    return possibleIndexList

def isPartOfMill(field, player, indexRing, indexPos):
    bPartOfMill = False
    if indexPos%2 == 1: # middle of an edge
        if (field[indexRing, (indexPos-1)%8] == player # mill in one ring
            and field[indexRing, (indexPos+1)%8] == player
        ):
            bPartOfMill = True
        elif (field[0, indexPos] == player # mill over all rings
              and field[1, indexPos] == player
              and field[2, indexPos] == player
        ):
            bPartOfMill = True
    else:   # corner
        if (field[indexRing, (indexPos-2)%8] == player # mill in one ring
            and field[indexRing, (indexPos-1)%8] == player
        ):
            bPartOfMill = True
        elif (field[indexRing, (indexPos+1)%8] == player # mill in one ring
              and field[indexRing, (indexPos+2)%8] == player
        ):
            bPartOfMill = True
    return bPartOfMill




################################## Agent class ##################################
class DQNAgent:
    def __init__(self, player):
        
        # The player the agent represents
        self.player = player # ToDo: player should be changeable

        # Main models
        self.model_set = self.create_model_set()
        self.model_shift = self.create_model_shift()
        self.model_jump = self.create_model_jump()

        # Target networks
        self.target_model_set = self.create_model_set()
        self.target_model_set.set_weights(self.model_set.get_weights())
        self.target_model_shift = self.create_model_shift()
        self.target_model_shift.set_weights(self.model_shift.get_weights())
        self.target_model_jump = self.create_model_jump()
        self.target_model_jump.set_weights(self.model_jump.get_weights())

        # An array with last n steps for training
        self.replay_memory = [None]*3   # for 3 phases: set, shift, jump
        self.replay_memory[PHASE_SET] = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.replay_memory[PHASE_SHIFT] = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.replay_memory[PHASE_JUMP] = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        # self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model_set(self):
        model = Sequential()
        
        model.add(Input(shape=(SET_STATE_SIZE,)))
        
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))

        model.add(Dense(SET_ACTIONS_SIZE, activation='linear'))  # SET_ACTIONS_SIZE = how many choices (48)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model
    
    def create_model_shift(self):
        model = Sequential()

        model.add(Input(shape=(SHIFT_STATE_SIZE,)))
        
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))

        model.add(Dense(SHIFT_ACTIONS_SIZE, activation='linear'))  # SHIFT_ACTIONS_SIZE = how many choices (88)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model
    
    def create_model_jump(self):
        model = Sequential()

        model.add(Input(shape=(JUMP_STATE_SIZE,)))
        
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))

        model.add(Dense(JUMP_ACTIONS_SIZE, activation='linear'))  # JUMP_ACTIONS_SIZE = how many choices (576)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, new field, new move, new phase, done)
    def update_replay_memory(self, transition, phase):
        self.replay_memory[phase].append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, phase, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory[phase]) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory[phase], MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        if PHASE_SET == phase:
            current_qs_list = self.model_set.predict(current_states)
        elif PHASE_SHIFT == phase:
            current_qs_list = self.model_shift.predict(current_states)
        else: # PHASE_JUMP
            current_qs_list = self.model_jump.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        new_phase_list = np.array([transition[6] for transition in minibatch])
        # future_qs_list with length of minibatch
        future_qs_list = []
        try:
            future_qs_list_set = self.target_model_set.predict(new_current_states[new_phase_list==PHASE_SET])
        except:
            pass
        try:
            future_qs_list_shift = self.target_model_shift.predict(new_current_states[new_phase_list==PHASE_SHIFT])
        except:
            pass
        try:
            future_qs_list_jump = self.target_model_jump.predict(new_current_states[new_phase_list==PHASE_JUMP])
        except:
            pass
        phase_set_index=0
        phase_shift_index=0
        phase_jump_index=0
        for new_phase in new_phase_list:
            if PHASE_SET == new_phase:
                future_qs_list += [future_qs_list_set[phase_set_index]]
                phase_set_index += 1
            elif PHASE_SHIFT == new_phase:
                future_qs_list += [future_qs_list_shift[phase_shift_index]]
                phase_shift_index += 1
            else: # PHASE_JUMP
                future_qs_list += [future_qs_list_jump[phase_jump_index]]
                phase_jump_index += 1
        # future_qs_list[new_phase_list==PHASE_SET] = setModel.predict(new_current_states[new_phase_list==PHASE_SET]) if >0 set_states
        # future_qs_list[new_phase_list==PHASE_SHIFT] = shiftModel.predict(new_current_states[new_phase_list==PHASE_SHIFT]) if >0 shift_states
        # future_qs_list[new_phase_list==PHASE_JUMP] = jumpModel.predict(new_current_states[new_phase_list==PHASE_JUMP]) if >0 jump_states
        # future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, new_field, new_move, new_phase, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                if PHASE_SET == new_phase:
                    validIdx = getPossibleMovesSetIndex(self.player, new_field, new_move)
                elif PHASE_SHIFT == new_phase:
                    validIdx = getPossibleMovesShiftIndex(self.player, new_field, new_move)
                else: # PHASE_JUMP
                    validIdx = getPossibleMovesJumpIndex(self.player, new_field, new_move)
                max_future_q = np.max(future_qs_list[index][validIdx])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        if PHASE_SET == phase:
            self.model_set.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)
        elif PHASE_SHIFT == phase:
            self.model_shift.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)
        else: # PHASE_JUMP
            self.model_jump.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter >= UPDATE_TARGET_EVERY:
            self.target_model_set.set_weights(self.model_set.get_weights())
            self.target_model_shift.set_weights(self.model_shift.get_weights())
            self.target_model_jump.set_weights(self.model_jump.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    # state: list with values for input neurons of the network
    def get_qs(self, state, phase):
        if PHASE_SET == phase:
            return self.model_set.predict(np.array(state).reshape(-1, len(state)))[0]
        elif PHASE_SHIFT == phase:
            return self.model_shift.predict(np.array(state).reshape(-1, len(state)))[0]
        else: # PHASE_JUMP
            return self.model_jump.predict(np.array(state).reshape(-1, len(state)))[0]



################################# Training class ################################
class DQNTraining:
    episode_reward1 = None
    episode_reward2 = None
    step = None
    
    player = None
    field = None
    move = None
    inputVector = None
    phase = None
    
    def __init__(self, initParams=None):
        ####CDI#####
        tf.keras.utils.disable_interactive_logging()
        ####CDI#####
        
        # create a instanceof the game
        self.GameMillLogic = MillLogic(checkMoves=False)

        # create two agents to play against each other
        self.agent1 = DQNAgent(PLAYER1)
        self.agent2 = DQNAgent(PLAYER2)
        
        # start value for esilon greedy algorithm
        self.epsilon = 1
        
        # store relevant old game parameters
        self.lastInputVector = {PLAYER1:None, PLAYER2:None} # save last inputVectors of both players
        self.lastSelectedAction = {PLAYER1:None, PLAYER2:None} # save last selected action of both players
        self.lastPhase = {PLAYER1:None, PLAYER2:None} # save last phase of both players
        self.lastActionTypes = {PLAYER1:enAction.SetPlayer1, PLAYER2:enAction.SetPlayer2}
        
        # training stats 
        self.ep_rewards_1 = []
        self.ep_rewards_2 = []
        self.ep_steps = []
        

    def startGame(self):
        self.episode_reward1 = 0
        self.episode_reward2 = 0
        self.step = 1

        # Reset environment and get initial state
        self.GameMillLogic.restartGame()
        self.field, actionType, _, inStockTokens, remainingTokens = self.GameMillLogic.getFullState() # initial state, empty field
        self.player = getPlayerFromActionType(actionType) # player 1 starts
        self.inputVector, self.phase, self.move = buildInputVector(self.player, self.field, actionType, inStockTokens, remainingTokens) # phase is "set", move ist "true"
        
        if PHASE_SET == self.phase: # start in phase set
            validIdx = getPossibleMovesSetIndex(self.player, self.field, self.move) # all set moves possible
        elif PHASE_SHIFT == self.phase:
            validIdx = getPossibleMovesShiftIndex(self.player, self.field, self.move) 
        else: # PHASE_JUMP
            validIdx = getPossibleMovesJumpIndex(self.player, self.field, self.move) 

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > self.epsilon:
            # Get action from Q table
            if PLAYER1 == self.player:
                qVals = self.agent1.get_qs(self.inputVector, self.phase)
            else:   # PLAYER2
                qVals = self.agent2.get_qs(self.inputVector, self.phase)
            selectedPossibleAction = np.argmax(qVals[validIdx])
        else:
            # Get random action
            selectedPossibleAction = np.random.randint(0, len(validIdx))
        selectedAction = validIdx[selectedPossibleAction] # select a set action
        
        if PHASE_SET == self.phase: # start in phase set
            newField = getMoveFromIndexSet(self.player, self.field, selectedAction) # new field with 1 token from player 1 
        elif PHASE_SHIFT == self.phase:
            newField = getMoveFromIndexShift(self.player, self.field, selectedAction)  
        else: # PHASE_JUMP
            newField = getMoveFromIndexJump(self.player, self.field, selectedAction) 
            
        self.lastInputVector[self.player] = self.inputVector # save values of player 1's first move
        self.lastSelectedAction[self.player] = selectedAction
        self.lastPhase[self.player] = self.phase
            
        self.GameMillLogic.setMove(newField) # make player 1's first move
        self.field, actionType, _, inStockTokens, remainingTokens = self.GameMillLogic.getFullState() # field with 1 token from player 1 
        self.player = getPlayerFromActionType(actionType) # player 2 makes the second move
        self.inputVector, self.phase, self.move = buildInputVector(self.player, self.field, actionType, inStockTokens, remainingTokens) # phase is "set", move ist "true"

        
    def stepGame(self):
        if PHASE_SET == self.phase:
            validIdx = getPossibleMovesSetIndex(self.player, self.field, self.move) 
        elif PHASE_SHIFT == self.phase:
            validIdx = getPossibleMovesShiftIndex(self.player, self.field, self.move) 
        else: # PHASE_JUMP
            validIdx = getPossibleMovesJumpIndex(self.player, self.field, self.move) 

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > self.epsilon:
            # Get action from Q table
            if PLAYER1 == self.player:
                qVals = self.agent1.get_qs(self.inputVector, self.phase)
            else:   # PLAYER2
                qVals = self.agent2.get_qs(self.inputVector, self.phase)
            selectedPossibleAction = np.argmax(qVals[validIdx])
        else:
            # Get random action
            selectedPossibleAction = np.random.randint(0, len(validIdx))
        selectedAction = validIdx[selectedPossibleAction]
        
        if PHASE_SET == self.phase:
            newField = getMoveFromIndexSet(self.player, self.field, selectedAction) 
        elif PHASE_SHIFT == self.phase:
            newField = getMoveFromIndexShift(self.player, self.field, selectedAction)  
        else: # PHASE_JUMP
            newField = getMoveFromIndexJump(self.player, self.field, selectedAction) 
            
        self.lastInputVector[self.player] = self.inputVector
        self.lastSelectedAction[self.player] = selectedAction
        self.lastPhase[self.player] = self.phase
            
        bActionValid = self.GameMillLogic.setMove(newField)
        # if not bActionValid:
        #     print("ERROR: Invalid action!")
        self.field, actionType, _, inStockTokens, remainingTokens = self.GameMillLogic.getFullState()
        self.player = getPlayerFromActionType(actionType)
        self.inputVector, self.phase, self.move = buildInputVector(self.player, self.field, actionType, inStockTokens, remainingTokens)
        reward, self.lastActionTypes, done = getReward(self.player, actionType, self.lastActionTypes)
        
        if PLAYER1 == self.player:
            # Count reward
            self.episode_reward1 += reward

            # Every step we update replay memory and train main network
            self.agent1.update_replay_memory((self.lastInputVector[self.player], self.lastSelectedAction[self.player], reward, self.inputVector, self.field, self.move, self.phase, done), self.lastPhase[self.player])
            self.agent1.train(done, self.lastPhase[self.player], self.step)
        else:   # PLAYER2
            # Count reward
            self.episode_reward2 += reward

            # Every step we update replay memory and train main network
            self.agent2.update_replay_memory((self.lastInputVector[self.player], self.lastSelectedAction[self.player], reward, self.inputVector, self.field, self.move, self.phase, done), self.lastPhase[self.player])
            self.agent2.train(done, self.lastPhase[self.player], self.step)
        if True == done: # player won the game
            reward = REWARD_LOSS # reward for opponent
            self.player = PLAYER1 + PLAYER2 - self.player # get the player who lost
            if PLAYER1 == self.player: # PLAYER1 lost
                # Count reward
                self.episode_reward1 += reward

                # Every step we update replay memory and train main network
                # inputVector, field, move and phase are irrelevent because done is True
                self.agent1.update_replay_memory((self.lastInputVector[self.player], self.lastSelectedAction[self.player], reward, self.inputVector, self.field, self.move, self.phase, done), self.lastPhase[self.player])
                self.agent1.train(done, self.lastPhase[self.player], self.step)
            else:   # PLAYER2 lost
                # Count reward
                self.episode_reward2 += reward

                # Every step we update replay memory and train main network
                # inputVector, field, move and phase are irrelevent because done is True
                self.agent2.update_replay_memory((self.lastInputVector[self.player], self.lastSelectedAction[self.player], reward, self.inputVector, self.field, self.move, self.phase, done), self.lastPhase[self.player])
                self.agent2.train(done, self.lastPhase[self.player], self.step)

        self.step += 1
        return done, self.step

        
    def finishGame(self):
        self.ep_rewards_1 += [self.episode_reward1]
        self.ep_rewards_2 += [self.episode_reward2]
        self.ep_steps += [self.step]

        # Decay epsilon
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(MIN_EPSILON, self.epsilon)

        
    def loadModels(self):
        1
    
    
    def saveModels(self):
        1

        
    def getGameStats(self, bFull=False):
        if bFull: # return stats of all played games
            return self.ep_rewards_1, self.ep_rewards_2, self.ep_steps
        else: # return stats of last played game
            return self.ep_rewards_1[-1], self.ep_rewards_2[-1], self.ep_steps[-1]




if __name__=='__main__': 
    trainingDQN = DQNTraining()
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        trainingDQN.startGame()
        done = False
        step = 1
        while not done and step < MAX_STEPS:
            done, step = trainingDQN.stepGame()
        trainingDQN.finishGame()
    ep_rewards_1, ep_rewards_2, ep_steps = trainingDQN.getGameStats(bFull=True)




















# muehle = MillLogic()
# agent1 = DQNAgent(PLAYER1)
# agent2 = DQNAgent(PLAYER2)
# lastInputVector = {PLAYER1:None, PLAYER2:None} # save last inputVectors of both players
# lastSelectedAction = {PLAYER1:None, PLAYER2:None} # save last selected action of both players
# lastPhase = {PLAYER1:None, PLAYER2:None} # save last phase of both players
# lastActionTypes = {PLAYER1:enAction.SetPlayer1, PLAYER2:enAction.SetPlayer2}
# ep_rewards_1 = []
# ep_rewards_2 = []
# ep_steps = []

# # Iterate over episodes
# for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

#     # Restarting episode - reset episode reward and step number
#     print(f"Episode {episode}")
#     episode_reward1 = 0
#     episode_reward2 = 0
#     step = 1

#     # Reset environment and get initial state
#     muehle.restartGame()
#     field, actionType, _, inStockTokens, remainingTokens = muehle.getFullState() # initial state, empty field
#     player = getPlayerFromActionType(actionType) # player 1 starts
#     inputVector, phase, move = buildInputVector(player, field, actionType, inStockTokens, remainingTokens) # phase is "set", move ist "true"
    
#     if PHASE_SET == phase: # start in phase set
#         validIdx = getPossibleMovesSetIndex(player, field, move) # all set moves possible
#     elif PHASE_SHIFT == phase:
#         validIdx = getPossibleMovesShiftIndex(player, field, move) 
#     else: # PHASE_JUMP
#         validIdx = getPossibleMovesJumpIndex(player, field, move) 

#     # This part stays mostly the same, the change is to query a model for Q values
#     if np.random.random() > epsilon:
#         # Get action from Q table
#         if PLAYER1 == player:
#             qVals = agent1.get_qs(inputVector, phase)
#         else:   # PLAYER2
#             qVals = agent2.get_qs(inputVector, phase)
#         selectedPossibleAction = np.argmax(qVals[validIdx])
#     else:
#         # Get random action
#         selectedPossibleAction = np.random.randint(0, len(validIdx))
#     selectedAction = validIdx[selectedPossibleAction] # select a set action
    
#     if PHASE_SET == phase: # start in phase set
#         newField = getMoveFromIndexSet(player, field, selectedAction) # new field with 1 token from player 1 
#     elif PHASE_SHIFT == phase:
#         newField = getMoveFromIndexShift(player, field, selectedAction)  
#     else: # PHASE_JUMP
#         newField = getMoveFromIndexJump(player, field, selectedAction) 
        
#     lastInputVector[player] = inputVector # save values of player 1's first move
#     lastSelectedAction[player] = selectedAction
#     lastPhase[player] = phase
        
#     muehle.setMove(newField) # make player 1's first move
#     field, actionType, _, inStockTokens, remainingTokens = muehle.getFullState() # field with 1 token from player 1 
#     player = getPlayerFromActionType(actionType) # player 2 makes the second move
#     inputVector, phase, move = buildInputVector(player, field, actionType, inStockTokens, remainingTokens) # phase is "set", move ist "true"
    

#     # Reset flag and start iterating until episode ends
#     done = False
#     while not done and step < MAX_STEPS:        
#         # bekannt zu beginn: player, field, move, inputVector
#         # aktueller Zustand gegeben
#         # naechste Aktion bestimmen
#         # Aktion durchfuehren
#         # aktuellen Zustand und gewaehlte Aktion speichern
#         # neuen Zustand bestimmen
#         # update_replay_memory und trainieren
        
#         if PHASE_SET == phase:
#             validIdx = getPossibleMovesSetIndex(player, field, move) 
#         elif PHASE_SHIFT == phase:
#             validIdx = getPossibleMovesShiftIndex(player, field, move) 
#         else: # PHASE_JUMP
#             validIdx = getPossibleMovesJumpIndex(player, field, move) 

#         # This part stays mostly the same, the change is to query a model for Q values
#         if np.random.random() > epsilon:
#             # Get action from Q table
#             if PLAYER1 == player:
#                 qVals = agent1.get_qs(inputVector, phase)
#             else:   # PLAYER2
#                 qVals = agent2.get_qs(inputVector, phase)
#             selectedPossibleAction = np.argmax(qVals[validIdx])
#         else:
#             # Get random action
#             selectedPossibleAction = np.random.randint(0, len(validIdx))
#         selectedAction = validIdx[selectedPossibleAction]
        
#         if PHASE_SET == phase:
#             newField = getMoveFromIndexSet(player, field, selectedAction) 
#         elif PHASE_SHIFT == phase:
#             newField = getMoveFromIndexShift(player, field, selectedAction)  
#         else: # PHASE_JUMP
#             newField = getMoveFromIndexJump(player, field, selectedAction) 
            
#         lastInputVector[player] = inputVector
#         lastSelectedAction[player] = selectedAction
#         lastPhase[player] = phase
            
#         bActionValid = muehle.setMove(newField)
#         # if not bActionValid:
#         #     print("ERROR: Invalid action!")
#         field, actionType, _, inStockTokens, remainingTokens = muehle.getFullState()
#         player = getPlayerFromActionType(actionType)
#         inputVector, phase, move = buildInputVector(player, field, actionType, inStockTokens, remainingTokens)
#         reward, lastActionTypes, done = getReward(player, actionType, lastActionTypes)
        
#         if PLAYER1 == player:
#             # Count reward
#             episode_reward1 += reward

#             # Every step we update replay memory and train main network
#             agent1.update_replay_memory((lastInputVector[player], lastSelectedAction[player], reward, inputVector, field, move, phase, done), lastPhase[player])
#             agent1.train(done, lastPhase[player], step)
#         else:   # PLAYER2
#             # Count reward
#             episode_reward2 += reward

#             # Every step we update replay memory and train main network
#             agent2.update_replay_memory((lastInputVector[player], lastSelectedAction[player], reward, inputVector, field, move, phase, done), lastPhase[player])
#             agent2.train(done, lastPhase[player], step)
#         if True == done: # player won the game
#             reward = REWARD_LOSS # reward for opponent
#             player = PLAYER1 + PLAYER2 - player # get the player who lost
#             if PLAYER1 == player: # PLAYER1 lost
#                 # Count reward
#                 episode_reward1 += reward

#                 # Every step we update replay memory and train main network
#                 # inputVector, field, move and phase are irrelevent because done is True
#                 agent1.update_replay_memory((lastInputVector[player], lastSelectedAction[player], reward, inputVector, field, move, phase, done), lastPhase[player])
#                 agent1.train(done, lastPhase[player], step)
#             else:   # PLAYER2 lost
#                 # Count reward
#                 episode_reward2 += reward

#                 # Every step we update replay memory and train main network
#                 # inputVector, field, move and phase are irrelevent because done is True
#                 agent2.update_replay_memory((lastInputVector[player], lastSelectedAction[player], reward, inputVector, field, move, phase, done), lastPhase[player])
#                 agent2.train(done, lastPhase[player], step)

#         step += 1
#     ep_rewards_1 += [episode_reward1]
#     ep_rewards_2 += [episode_reward2]
#     ep_steps += [step]

#     # Decay epsilon
#     if epsilon > MIN_EPSILON:
#         epsilon *= EPSILON_DECAY
#         epsilon = max(MIN_EPSILON, epsilon)








