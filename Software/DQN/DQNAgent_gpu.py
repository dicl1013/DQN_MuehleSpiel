# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:16:27 2023

@author: andre
"""

import numpy as np

import sys
# import os
# os.environ["CUDA-VISIBLE_DEVICES"]="-1"

import tensorflow as tf
from tensorflow import keras
#import keras.backend.tensorflow_backend as backend
# from tensorflow.keras import Input
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import TensorBoard

import multiprocessing

from collections import deque
# import time
import random
from tqdm import tqdm

from timeit import default_timer as timer

from MuehleLogik.muehle_logik import MillLogic
from MuehleLogik.muehle_logik import enAction 
from MuehleLogik.muehle_logik import PLAYER1, PLAYER2, EMPTY



DISCOUNT = 0.99 # decides how relevant the future q is
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_024  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 1024  # How many steps (samples) to use for training
TRAIN_EVERY_X_STEPS = 16 # The training will only made in one of 16 steps of the player
UPDATE_TARGET_EVERY = 5 # number of terminal states (end of episodes) before updating target models
LEARNING_RATE = 1e-5 # learning rate for all models

# Environment settings
EPISODES = 40#20_000
MAX_STEPS = 400

# Exploration settings
EPSILON_DECAY = 0.99975 # after each episode epsilon = epsilon*EPSILON_DECAY
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
# start positions of all possible shift moves
# 0:outer ring, upper-left-corner, then clockwise 0->1->...->7
# 8..15 the same for middle ring
# 16..23 the same for inner ring
SHIFT_FROM_PLACES = 2*[0] +3*[1] +2*[2] +3*[3] +2*[4] +3*[5] +2*[6] +3*[7] \
                   +2*[8] +4*[9] +2*[10]+4*[11]+2*[12]+4*[13]+2*[14]+4*[15]\
                   +2*[16]+3*[17]+2*[18]+3*[19]+2*[20]+3*[21]+2*[22]+3*[23]

# new positions of all possible shift moves (0->1, 0->7, 1->0, 1->2, 1->9, 2->1, ...)                   
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

# Agents
AGENT1 = 1 # used for training
AGENT2 = 2
AGENT_TRAINED = AGENT1 # used for validation
AGENT_REFERENCE = AGENT2

tf.keras.utils.disable_interactive_logging() # COMMENT OUT IF ERROR OCCURS, disables too much logging in some versions of tensorflow
# tf.get_logger().setLevel('ERROR')
tf.get_logger().setLevel('WARNING')

######### test code for helper functions #########
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

######### test functions to deactivate some predictions #########
# def dummy_predict_set(current_states):
#     return np.ones((current_states.shape[0], SET_ACTIONS_SIZE))/100

# def dummy_predict_shift(current_states):
#     return np.ones((current_states.shape[0], SHIFT_ACTIONS_SIZE))/100

# def dummy_predict_jump(current_states):
#     return np.ones((current_states.shape[0], JUMP_ACTIONS_SIZE))/100


############################ Global helper functions ############################
def getReward(player, nextAction, dicLastActions, rewardList):
    """
    calculate the reward for the player that has to make the next action to evaluate his last action
    the reward depends on the next action of the player and the last action of the opposite player

    Parameters
    ----------
    player : the player who made the last move
    nextAction : the next action that results from the made move
    dicLastActions : dictionary with the last action for each player
    rewardList : the reward value for all different situations

    Returns
    -------
    reward : value of the calculated reward
    dicLastActions : updated dictonary for last actions
    done : bool, True if the game is over

    """
    done = False
    rewardMove, rewardOwnMill, rewardEnemyMill, rewardWin, rewardLoss = rewardList
    
    if enAction.Player1Wins == nextAction or enAction.Player2Wins == nextAction:
        reward = rewardWin # player won the game
        done = True # game is over
    elif enAction.RemoveTokenPlayer1 == nextAction or enAction.RemoveTokenPlayer2 == nextAction:
        reward = rewardOwnMill # player can remove a token, therefore he has closed a mill with the last move
    elif enAction.RemoveTokenPlayer1 == dicLastActions[PLAYER1+PLAYER2-player] or enAction.RemoveTokenPlayer2 == dicLastActions[PLAYER1+PLAYER2-player]:
        reward = rewardEnemyMill # the opposite has removed a token by closing a mill
    else:
        reward = rewardMove # a move without special consequences
    
    # save the new action as last action for the next calculation
    dicLastActions[player] = nextAction
    
    return reward, dicLastActions, done

def getPlayerFromActionType(actionType):
    """
    get the player that is on move from the action type
    even values represent player 1

    Parameters
    ----------
    actionType : from enum enAction (mill logic)
        next expected action

    Returns
    -------
    player : player that is on move

    """
    if actionType.value % 2:
        player = PLAYER2
    else:
        player = PLAYER1
    return player
    
def buildInputVector(player, field, actionType, inStockTokens, remainingTokens):
    """
    build a vector that serves as input for the model (neural network) for the actual game phase

    Parameters
    ----------
    player : player that is on move
    field : array with shape (3, 8)
        state of the millField
    actionType: from enum enAction
        next expected action
    inStockTokens: dictionary with number of tokens still to place for each player
    remainingTokens: dictionary with number of tokens still in game for each player

    Returns
    -------
    inputVector : the input vector for the model of the actual game phase
    phase : actual phase of the game
    move : bool
        True if the network has to predict a move action
        False if the network has to predict the remove of a token from the opposite

    """
    inputVector = []
    
    # field, actionType, _, inStockTokens, remainingTokens = muehle.getFullState()
    
    # calculate the game phase from the parameters of the mill logic
    # if enAction.Player1Wins == actionType or enAction.Player2Wins == actionType:
    #     phase = PHASE_GAMEFINISHED
    if inStockTokens[player] > 0: # tokens to set are available
        phase = PHASE_SET
    elif remainingTokens[player] > 3: # more than 3 tokens -> no jumping
        phase = PHASE_SHIFT
    else:
        phase = PHASE_JUMP
    
    move = 0
    remove = 0
    if enAction.RemoveTokenPlayer1 == actionType or enAction.RemoveTokenPlayer2 == actionType:
        remove = 1 # remove token from the opposite as next action
    else:
        move = 1 # normal move as next action (set, move, jump depending on phase)

    # put all relevant information into the input vector
    inputVector += [player,move,remove]
    inputVector += field.flatten().tolist()
    if PHASE_SET == phase:
        inputVector += [inStockTokens[player]]
    
    return inputVector, phase, move

def getMoveFromIndex(phase, player, field, index):
    """
    the index of the highest valid prediction of the model determines the move to make
    this function calculates the new mill field as move to set into the mill logic from this index

    Parameters
    ----------
    phase : actual phase of the game and the used model
    player : player that makes the move
    field : array with shape (3, 8)
        old state of the millField
    index : index of the selected move

    Returns
    -------
    newField : array with shape (3, 8)
        new state of the millField

    """
    # use the concrete funktion of the selected phase
    if PHASE_SET == phase: # start in phase set
        newField = getMoveFromIndexSet(player, field, index) 
    elif PHASE_SHIFT == phase:
        newField = getMoveFromIndexShift(player, field, index)  
    else: # PHASE_JUMP
        newField = getMoveFromIndexJump(player, field, index) 
    return newField

def getMoveFromIndexSet(player, field, index):
    """
    calculation of the move from the index in set phase
    only called by getMoveFromIndex()
    """
    newField=field.copy()   # do not change given field
    if index < 24:  #remove
        newField[index//8,index%8]=EMPTY # remove a token by setting the place to empty
    else:   # move
        newField[(index-24)//8,index%8]=player # set a token by setting the place to player
    return newField
    
def getMoveFromIndexShift(player, field, index):
    """
    calculation of the move from the index in shift phase
    only called by getMoveFromIndex()
    """
    newField=field.copy()   # do not change given field
    if index < 24:  #remove
        newField[index//8,index%8] = EMPTY # remove a token by setting the place to empty
    else:   # move
        # the moves belonging to an index are stored in SHIFT_FROM_PLACES and SHIFT_TO_PLACES
        fromIndex = SHIFT_FROM_PLACES[index-24]
        newField[fromIndex//8,fromIndex%8] = EMPTY # delete the token from the old place
        toIndex = SHIFT_TO_PLACES[index-24]
        newField[toIndex//8,toIndex%8] = player # set the token to the new place
    return newField

def getMoveFromIndexJump(player, field, index):
    """
    calculation of the move from the index in jump phase
    only called by getMoveFromIndex()
    """
    newField=field.copy()   # do not change given field
    if index < 24:  #remove
        newField[index//8,index%8]=EMPTY # remove a token by setting the place to empty
    else:   # move
        index-=24 # start with 0 for the first move index 24
        fromIndex=index//23
        toIndex=index%23
        if toIndex >= fromIndex: # toIndex and fromIndex cannot be even -> 23 possibilites of toIndex correspond to the 24 places without fromIndex
            toIndex += 1
        newField[fromIndex//8,fromIndex%8]=EMPTY # delete the token from the old place
        newField[toIndex//8,toIndex%8]=player # set the token to the new place
    return newField

def getPossibleMovesIndex(phase, player, field, move):
    """
    get a list of indices for all moves that are possible in the current state of the mill logic

    Parameters
    ----------
    phase : actual phase of the game and the used model
    player : player that makes the move
    field : array with shape (3, 8)
        current state of the millField
    move : bool
        True if a move action is required
        False if a remove action is required

    Returns
    -------
    possibleIndexList : list of indices for all moves that are possible

    """
    # use the concrete funktion of the selected phase
    if PHASE_SET == phase:
        possibleIndexList = getPossibleMovesSetIndex(player, field, move)
    elif PHASE_SHIFT == phase:
        possibleIndexList = getPossibleMovesShiftIndex(player, field, move)
    else: # PHASE_JUMP
        possibleIndexList = getPossibleMovesJumpIndex(player, field, move)
    return possibleIndexList

def getPossibleMovesSetIndex(player, field, move):
    """
    calculation of a list of indices for all moves that are possible in set phase
    only called by getPossibleMovesIndex()
    """
    possibleIndexList = []
    
    if move:
        # all indices where the corresponding place in the field is empty, are possible
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
    """
    calculation of a list of indices for all moves that are possible in shift phase
    only called by getPossibleMovesIndex()
    """
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
    """
    calculation of a list of indices for all moves that are possible in jump phase
    only called by getPossibleMovesIndex()
    """
    possibleIndexList = []
    
    if move:
        field = field.ravel()
        fromArr=field==player
        toArr=field==EMPTY
        validMat = fromArr[:, np.newaxis] * toArr[np.newaxis, :] # True for all combinations where fromArr is player and toArr is empty
        validArr = validMat[~np.eye(validMat.shape[0],dtype=bool)] # remove the main diagonal where from and to are equal
        possibleIndexList=np.arange(24,JUMP_ACTIONS_SIZE)[validArr]
        ### version of code that is easier to understand but slower ###
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
    """
    calculation of a list of indices for all moves that are possible if removing a token is required
    only called by getPossibleMovesSetIndex(), getPossibleMovesShiftIndex(), getPossibleMovesJumpIndex()

    Parameters
    ----------
    field : array with shape (3, 8)
        current state of the millField
    playerOfFormedMill : the player who formed the mill and can remove a token from the opposite

    Returns
    -------
    possibleIndexList : list of indices for all moves that are possible

    """
    possibleIndexList = []
    playerToRemoveFrom = PLAYER1 + PLAYER2 - playerOfFormedMill
    notPartOfMillTokens = np.zeros([3,8], dtype=bool)
    
    # find all tokens of the player to remove from that are not part of a mill
    for indexRing in range(3):  # iterate through rings
        for indexPos, place in enumerate(field[indexRing,:]):  # iterate through positions in the ring
            # place is EMPTY, PLAYER1 or PLAYER2
            if playerToRemoveFrom == place:
                if False == isPartOfMill(field, playerToRemoveFrom, indexRing, indexPos):
                    notPartOfMillTokens[indexRing,indexPos] = True
    
    bOnlyMills = False
    if np.all(notPartOfMillTokens == False):    # every token of the player to remove from is part of a mill
        bOnlyMills = True
    
    # find all possible moves            
    for indexRing in range(3):  # iterate through rings
        for indexPos, place in enumerate(field[indexRing,:]):  # iterate through positions in the ring
            # place is EMPTY, PLAYER1 or PLAYER2
            if playerToRemoveFrom == place:
                    if bOnlyMills or True == notPartOfMillTokens[indexRing,indexPos]:   # the token is not part of a mill or there are only mills
                        possibleIndexList += [indexRing*8+indexPos]
                        
    return possibleIndexList

def isPartOfMill(field, player, indexRing, indexPos):
    """
    checks if the token from player on position (indexRing, indexPos) is part of a mill
    returns True if the selected token is part of a mill
    equal to same function in mill logic but usable for any field without the need of changing the state of the mill logic
    """
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




################################################################### Agent class ###################################################################
class DQNAgent:
    def __init__(self, bCreateModels, replayMemSize, learningRate):
        """
        initialize a new agent that can used for training models for all phases in the mill game or for playing a game

        Parameters
        ----------
        bCreateModels : bool
            True : new models with random weights for all phases of the mill game will be created
            False : no models will be created and it is necessary to load models before using the agent
        replayMemSize : number of the moves to save in the replay memory for each phase to use for training
        learningRate : learning rate that is used for all new created models
        """
        
        # if create models is inactive, loading models is required
        if bCreateModels:
            self.learningRate = learningRate
            # Main models
            self.model_set = self.create_model_set()
            self.model_shift = self.create_model_shift()
            self.model_jump = self.create_model_jump()
    
            # Target networks, more stable models that are used to predict the q values of future states
            # this future q values a used to calculate the new q values of current actions
            self.target_model_set = self.create_model_set()
            self.target_model_set.set_weights(self.model_set.get_weights()) # use the same weights as the normal models
            self.target_model_shift = self.create_model_shift()
            self.target_model_shift.set_weights(self.model_shift.get_weights())
            self.target_model_jump = self.create_model_jump()
            self.target_model_jump.set_weights(self.model_jump.get_weights())

        # double ended queues with last n steps to save states and moves as training data
        self.replay_memory = [None]*3   # for 3 phases: set, shift, jump
        self.replay_memory[PHASE_SET] = deque(maxlen=replayMemSize)
        self.replay_memory[PHASE_SHIFT] = deque(maxlen=replayMemSize)
        self.replay_memory[PHASE_JUMP] = deque(maxlen=replayMemSize)
        
        # counts the number of played steps to decide, when to start a training
        # training starts after reaching self.trainEveryXSteps
        self.step_counter = {PHASE_SET: 0, 
                             PHASE_SHIFT: 0,
                             PHASE_JUMP: 0}
        
        # define which models shall be trained
        self.bTrainActive = {PHASE_SET: True, 
                             PHASE_SHIFT: True,
                             PHASE_JUMP: True}

        # default configuration parameters, can be adapted with function configureAgent()
        self.discount = DISCOUNT # influece of future q values (following states of mill logic) for the new q values
        self.minReplayMemSize = MIN_REPLAY_MEMORY_SIZE # Minimum number of steps in a memory to start training
        self.minibatchSize = MINIBATCH_SIZE # How many steps (samples) to use for training
        self.trainEveryXSteps = TRAIN_EVERY_X_STEPS # The training will only be made in one of X steps of the player
        self.updateTargetEvery = UPDATE_TARGET_EVERY # number of terminal states (end of episodes) before updating target models

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
        
        # Used to calculate the mean MSE (mean squared error) during training
        # the MSE indicates the difference between the predicted Q values and the calculated new Q values from reward and future Q values
        self.mse_sum = 0
        self.mse_counter = 0

    def create_model_set(self):
        """
        create a new model for the set phase of the defined architecture
        """
        model = Sequential()
        
        model.add(InputLayer(input_shape=(SET_STATE_SIZE,)))
        
        # the number and size of hidden layers can be adapted for achieving a better result
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1024, activation='relu'))

        model.add(Dense(SET_ACTIONS_SIZE, activation='linear'))  # SET_ACTIONS_SIZE = how many choices (48)
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learningRate), metrics=['accuracy'])
        return model
    
    def create_model_shift(self):
        """
        create a new model for the shift phase of the defined architecture
        """
        model = Sequential()

        model.add(InputLayer(input_shape=(SHIFT_STATE_SIZE,)))
        
        # the number and size of hidden layers can be adapted for achieving a better result
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1024, activation='relu'))

        model.add(Dense(SHIFT_ACTIONS_SIZE, activation='linear'))  # SHIFT_ACTIONS_SIZE = how many choices (88)
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learningRate), metrics=['accuracy'])
        return model
    
    def create_model_jump(self):
        """
        create a new model for the jump phase of the defined architecture
        """
        model = Sequential()

        model.add(InputLayer(input_shape=(JUMP_STATE_SIZE,)))
        
        # the number and size of hidden layers can be adapted for achieving a better result
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1024, activation='relu'))

        model.add(Dense(JUMP_ACTIONS_SIZE, activation='linear'))  # JUMP_ACTIONS_SIZE = how many choices (576)
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learningRate), metrics=['accuracy'])
        return model
    
    def configureAgent(self, discount, minReplayMemSize, minibatchSize, trainEveryXSteps, updateTargetEvery):
        """
        set different parameters to configure the behavior of the agent

        Parameters
        ----------
        discount : influece of future q values (following states of mill logic) for the new q values
        minReplayMemSize : Minimum number of steps in the replay memory of a phase to start training
        minibatchSize : How many steps (samples) to use for training
        trainEveryXSteps : The training will only be made in one of X steps of the player
        updateTargetEvery : number of terminal states (end of episodes) before updating target models
        """
        self.discount = discount
        self.minReplayMemSize = minReplayMemSize
        self.minibatchSize = minibatchSize
        self.trainEveryXSteps = trainEveryXSteps
        self.updateTargetEvery = updateTargetEvery

    def update_replay_memory(self, transition, phase):
        """
        adds the data of given transition to the replay memory of the current phase

        Parameters
        ----------
        transition : all information of a transition between two states
            (current_state, action, reward, player, new_current_state, new_field, new_move, new_phase, done)
            current_state : input vector with information about the current state of the mill logic
            action : index of the selected action / output neuron
            reward : reward for the made action
            player : the player who made the selected action
            new_current_state : input vector with information about the following state of the mill logic
            new_field : array with shape (3, 8), following state of the mill field
            new_move : True if a move action is required as next action, False if a remove action is required as next action 
            new_phase : phase of the game in the new state
            done : True if the game is over after the selected action
        phase : game phase of the current state

        Returns
        -------
        None.

        """
        self.replay_memory[phase].append(transition)

    # 
    def train(self, terminal_state, phase, step):
        """
        test for training main network every step during episode and train if all conditions are passed
        function is called after every step but this is only used to control the number of trainings in each phase
        the training uses a batch of random transitions from the replay memories 

        Parameters
        ----------
        terminal_state : bool
            True : game is over, the last state of an episode
        phase : game phase to train
        step : step number in the episode, not used
        """
        
        if False == self.bTrainActive[phase]:
            # Training of this game phase is deactivated
            return
        
        self.step_counter[phase] += 1
        if self.trainEveryXSteps > self.step_counter[phase]:
            # defined number of steps until training not reached
            return
        else:
            # defined number of steps reached: start training
            self.step_counter[phase] = 0

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory[phase]) < self.minReplayMemSize:
            return
        
        # all conditions are passed, training the model starts
        
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory[phase], self.minibatchSize)
        
        # Get current states from minibatch
        current_states = np.array([transition[0] for transition in minibatch])
        current_states_tf = tf.cast(tf.convert_to_tensor(current_states), dtype=tf.float32)
        # with tf.device('/GPU:0'):
        # query NN model for Q values
        if PHASE_SET == phase:
            current_qs_list = self.model_set.predict(current_states_tf)
        elif PHASE_SHIFT == phase:
            current_qs_list = self.model_shift.predict(current_states_tf)
        else: # PHASE_JUMP
            current_qs_list = self.model_jump.predict(current_states_tf)
        
        # Get future states from minibatch
        new_current_states = np.array([transition[4] for transition in minibatch], dtype=object) # dtype=object is necessary because the states of different phases have different length
        new_phase_list = np.array([transition[7] for transition in minibatch])
        # future_qs_list with length of minibatch
        future_qs_list = []
        # with tf.device('/GPU:0'):
        # query NN target models for Q values
        if np.any(new_phase_list==PHASE_SET): # predict is only possible if new_current_states[new_phase_list==PHASE_SET] is not empty
            new_states_np = np.array(list(new_current_states[new_phase_list==PHASE_SET]),dtype=int) # in one phase the states have the same length -> normal numpy array can be used
            new_states_tf = tf.cast(tf.convert_to_tensor(new_states_np), dtype=tf.float32)
            future_qs_list_set = self.target_model_set.predict(new_states_tf)
            # future_qs_list_set = np.ones((np.sum(new_phase_list==PHASE_SET),SET_ACTIONS_SIZE))
            # future_qs_list_set = self.target_model_set.predict(np.array(list(new_current_states[new_phase_list==PHASE_SET]),dtype=int))
        if np.any(new_phase_list==PHASE_SHIFT):
            new_states_np = np.array(list(new_current_states[new_phase_list==PHASE_SHIFT]),dtype=int)
            new_states_tf = tf.cast(tf.convert_to_tensor(new_states_np), dtype=tf.float32)
            future_qs_list_shift = self.target_model_shift.predict(new_states_tf)
            # future_qs_list_shift = np.ones((np.sum(new_phase_list==PHASE_SHIFT),SHIFT_ACTIONS_SIZE))
            # future_qs_list_shift = self.target_model_shift.predict(np.array(list(new_current_states[new_phase_list==PHASE_SHIFT]),dtype=int))
        if np.any(new_phase_list==PHASE_JUMP):
            new_states_np = np.array(list(new_current_states[new_phase_list==PHASE_JUMP]),dtype=int)
            new_states_tf = tf.cast(tf.convert_to_tensor(new_states_np), dtype=tf.float32)
            future_qs_list_jump = self.target_model_jump.predict(new_states_tf)
            # future_qs_list_jump = np.ones((np.sum(new_phase_list==PHASE_JUMP),JUMP_ACTIONS_SIZE))
            # future_qs_list_jump = self.target_model_jump.predict(np.array(list(new_current_states[new_phase_list==PHASE_JUMP]),dtype=int))
        
        # merge the predicted Q values into one list (future_qs_list)
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

        X = [] # training data input
        y = [] # training data desired output

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, player, new_current_state, new_field, new_move, new_phase, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                validIdx = getPossibleMovesIndex(new_phase, player, new_field, new_move)
                max_future_q = np.max(future_qs_list[index][validIdx])
                new_q = reward + self.discount * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index].copy()
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)
           
            
        self.mse_sum += (abs(y-current_qs_list).sum() / self.minibatchSize) # only one Q value per transition has changed: minibatchSize=1024 -> 1024 qs are not equal
        self.mse_counter += 1
        
        X_tf = tf.cast(tf.convert_to_tensor(np.array(X)), dtype=tf.float32)
        y_tf = tf.cast(tf.convert_to_tensor(np.array(y)), dtype=tf.float32)
        # with tf.device('/GPU:0'):
        # Fit on all samples as one batch
        if PHASE_SET == phase:
            self.model_set.fit(X_tf, y_tf, batch_size=self.minibatchSize, verbose=0, shuffle=False)
            # self.model_set.fit(np.array(X), np.array(y), batch_size=self.minibatchSize, verbose=0, shuffle=False)
        elif PHASE_SHIFT == phase:
            self.model_shift.fit(X_tf, y_tf, batch_size=self.minibatchSize, verbose=0, shuffle=False)
            # self.model_shift.fit(np.array(X), np.array(y), batch_size=self.minibatchSize, verbose=0, shuffle=False)
        else: # PHASE_JUMP
            self.model_jump.fit(X_tf, y_tf, batch_size=self.minibatchSize, verbose=0, shuffle=False)
            # self.model_jump.fit(np.array(X), np.array(y), batch_size=self.minibatchSize, verbose=0, shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter >= self.updateTargetEvery:
            self.target_model_set.set_weights(self.model_set.get_weights())
            self.target_model_shift.set_weights(self.model_shift.get_weights())
            self.target_model_jump.set_weights(self.model_jump.get_weights())
            self.target_update_counter = 0
             
    def get_qs(self, states, phase):
        """
        Queries main network for Q values given current states

        Parameters
        ----------
        states : list with values for input neurons of the network
        phase : current phase of the mill game

        Returns
        -------
        Q values that are predicted from the model

        """
        # state_np = np.array(state).reshape(-1, len(state))
        # state_tf = tf.cast(tf.convert_to_tensor(state_np), dtype=tf.float32)
        states_tf = tf.cast(tf.convert_to_tensor(states), dtype=tf.float32)
        # with tf.device('/GPU:0'):
        if PHASE_SET == phase:
            return self.model_set.predict(states_tf)
            # return self.model_set.predict(np.array(state).reshape(-1, len(state)))[0]
        elif PHASE_SHIFT == phase:
            return self.model_shift.predict(states_tf)
            # return self.model_shift.predict(np.array(state).reshape(-1, len(state)))[0]
        else: # PHASE_JUMP
            return self.model_jump.predict(states_tf)
            # return self.model_jump.predict(np.array(state).reshape(-1, len(state)))[0]
        
    
    def selectModelsToTrain(self, bSet, bShift, bJump):
        """
        select which models shall be trained

        Parameters
        ----------
        bSet : bool
            True : model set will be trained
            False : model set will not be changed
        bShift : bool
            True : model shift will be trained
            False : model shift will not be changed
        bJump : bool
            True : model shift will be trained
            False : model shift will not be changed
        """
        self.bTrainActive[PHASE_SET] = bSet
        self.bTrainActive[PHASE_SHIFT] = bShift
        self.bTrainActive[PHASE_JUMP] = bJump
        
    
    def setModels(self, modelSet, modelShift, modelJump):
        """
        set models to be used for training

        Parameters
        ----------
        modelSet : model to use for set phase
        modelShift : model to use for shift phase
        modelJump : model to use for jump phase

        Returns
        -------
        bOkay : bool
            True : using the models was successful
            False : at least one model is invalid
        """
        bOkay = True
        
        # Main models
        self.model_set = modelSet
        self.model_shift = modelShift
        self.model_jump = modelJump

        try:
            # Target networks
            self.target_model_set = tf.keras.models.clone_model(self.model_set)
            self.target_model_set.set_weights(self.model_set.get_weights())
            self.target_model_shift = tf.keras.models.clone_model(self.model_shift)
            self.target_model_shift.set_weights(self.model_shift.get_weights())
            self.target_model_jump = tf.keras.models.clone_model(self.model_jump)
            self.target_model_jump.set_weights(self.model_jump.get_weights())
        except:
            bOkay = False
        
        return bOkay
    
    
    def getModels(self):
        """
        returns the models of the agent for all phases
        """
        return self.model_set, self.model_shift, self.model_jump
    
    def getMeanWeights(self):
        """
        returns the mean absolute weights of the models for all phases
        """
        # read out the model weights
        model_set_weights = self.model_set.get_weights()
        model_shift_weights = self.model_shift.get_weights()
        model_jump_weights = self.model_jump.get_weights()

        # calculate the mean value of the absolute values for the three phases
        weights_mean = []
        for model_weights in [model_set_weights, model_shift_weights, model_jump_weights]:
            weights_sum = 0
            weights_size = 0
            # iterate through the weights of all layers of the model
            for layer_weights in model_weights:
                weights_sum += abs(layer_weights).sum()
                weights_size += layer_weights.size
            
            weights_mean += [weights_sum / weights_size]
            
        return weights_mean # [weights_mean_set, weights_mean_shift, weights_mean_jump]
    
    def getMeanSquaredError(self):
        """
        returns the mean MSE (mean squared error) over all training steps after the last call
        
        the MSE indicates the difference between the predicted Q values 
        and the calculated new Q values from reward and future Q values
        """
        
        if self.mse_counter > 0:
            mean_mse = self.mse_sum / self.mse_counter
        else:
            mean_mse = 0
        self.mse_sum = 0
        self.mse_counter = 0
        
        return mean_mse




################################################################## Training class #################################################################
class DQNTraining:
    def __init__(self, bCreateModels=True, numInstances=1, replayMemSize=REPLAY_MEMORY_SIZE, learningRate=LEARNING_RATE):
        """
        initialize the training of one new or existing DQN agent 
        the agent uses the moves of both players to train

        Parameters
        ----------
        bCreateModels : bool
            True : new models with random weights will be created from the agent
            False : no models will be created and it is necessary to load models before using the agent
            optional. The default is True.
        numInstances : number of games to play at the same time for generating training data
            optional. The default is 1.
        replayMemSize : number of the moves to save in the replay memories of each agent
            optional. The default is REPLAY_MEMORY_SIZE.
        """
        
        self.numInstances = numInstances
        self.gameStarted = [False]*self.numInstances # start new games for all instances
        
        
        # create defined number of instances of the game
        self.GameMillLogic = [None]*self.numInstances
        for inst in range(self.numInstances):
            self.GameMillLogic[inst] = MillLogic(checkMoves=False) # don't check moves in training, forbidden moves are already prevented

        # create agent that predicts the moves of both players, plays against himself
        self.agent = DQNAgent(bCreateModels, replayMemSize, learningRate)
                
        # default configuration parameters, can be adapted with function configureTraining() 
        # configuration of epsilon greedy algorithm
        self.epsilon = 1 # start value for esilon greedy algorithm
        self.epsilonDecay = EPSILON_DECAY # new epsilon = epsilon * decay (at the end of each episode)
        self.epsilonMin = MIN_EPSILON # lowest possible epsilon
        # maximum steps before breaking a game
        self.maxSteps = MAX_STEPS
        
        # default rewards, can be adapted with function configureRewards() 
        self.rewardList=[REWARD_MOVE, REWARD_OWNMILL, REWARD_ENEMYMILL, REWARD_WIN, REWARD_LOSS]
        
        # game parameters for all instances
        self.episode_reward = [None]*self.numInstances # reward of the current game for each player
        self.step = [None]*self.numInstances # number of moves in the current episode
        self.field = [None]*self.numInstances # mill field
        self.player = [None]*self.numInstances # player who has to move
        self.inputVector = [None]*self.numInstances # input vector for the NN models
        self.phase = [None]*self.numInstances # game phase (PHASE_SET, PHASE_SHIFT, PHASE_JUMP)
        self.move = [None]*self.numInstances # bool, True for normal move, False for remove action
        
        # store relevant old game parameters
        self.lastInputVector = [None]*self.numInstances
        self.lastSelectedAction = [None]*self.numInstances
        self.lastPhase = [None]*self.numInstances
        self.lastActionTypes = [None]*self.numInstances
        for inst in range(self.numInstances):
            self.lastInputVector[inst] = {PLAYER1:None, PLAYER2:None} # save last inputVectors of both players
            self.lastSelectedAction[inst] = {PLAYER1:None, PLAYER2:None} # save index of the last selected action of both players
            self.lastPhase[inst] = {PLAYER1:None, PLAYER2:None} # save last phase of both players
            self.lastActionTypes[inst] = {PLAYER1:enAction.SetPlayer1, PLAYER2:enAction.SetPlayer2} # save type of the last action of both players for calculating the reward
        
        # training stats 
        self.ep_rewards_1 = [] # rewards from player 1 over all episodes
        self.ep_rewards_2 = [] # rewards from player 2 over all episodes
        self.ep_steps = [] # total number of moves from all episodes
        
        # model stats
        self.weights_set = [] # mean of all weights from the model for set phase
        self.weights_shift = [] # mean of all weights from the model for shift phase
        self.weights_jump = [] # mean of all weights from the model for jump phase
        
        self.ep_mse = [] # mean squared error in training over all episodes
        
        
        
    def configureTraining(self, epsilonStart=1, epsilonDecay=0.999, epsilonMin=0.001, maxSteps=400,
                          discount=0.99, minReplayMemSize=1024, minibatchSize=1024, 
                          trainEveryXSteps=16, updateTargetEvery=5):
        """
        set different parameters to configure the training

        Parameters - directly for training class
        ----------
        epsilonStart : start value for esilon greedy algorithm
            optional. The default is 1.
        epsilonDecay : new epsilon = epsilon * decay (at the end of each episode)
            optional. The default is 0.999.
        epsilonMin : lowest possible epsilon
            optional. The default is 0.001.
        maxSteps : maximum steps before breaking a game
            optional. The default is 400.
            
        Parameters - for training in the agent
        ----------
        discount : influece of future q values (following states of mill logic) for the new q values
            optional. The default is 0.99.
        minReplayMemSize : Minimum number of steps in the replay memory of a phase to start training
            optional. The default is 1024.
        minibatchSize : How many steps (samples) to use for training
            optional. The default is 1024.
        trainEveryXSteps : The training will only be made in one of X steps of the player
            optional. The default is 16.
        updateTargetEvery : number of terminal states (end of episodes) before updating target models
            optional. The default is 5.
        """
        self.epsilon = epsilonStart
        self.epsilonDecay = epsilonDecay
        self.epsilonMin = epsilonMin
        self.maxSteps = maxSteps
        self.agent.configureAgent(discount, minReplayMemSize, minibatchSize, trainEveryXSteps, updateTargetEvery)
        
    def configureRewards(self, rewardList):
        """
        set the reward values for all different situations

        Parameters
        ----------
        rewardList : [REWARD_MOVE, REWARD_OWNMILL, REWARD_ENEMYMILL, REWARD_WIN, REWARD_LOSS]
            REWARD_MOVE : normal move without special consequences, should be negative and small
            REWARD_OWNMILL : a own mill was created, should be positive and moderate
            REWARD_ENEMYMILL : the opponent has created a mill, should be negative and moderate
            REWARD_WIN : game won, should be positive and high
            REWARD_LOSS : game lost, should be negative and high
        """
        self.rewardList = rewardList
        
    def initializeGame(self):
        """
        begin with new games in all instances and reset the lists for training statistic
        """
        self.gameStarted = [False]*self.numInstances
        # reset training stats 
        self.ep_rewards_1 = []
        self.ep_rewards_2 = []
        self.ep_steps = []
        # reset model stats
        self.weights_set = []
        self.weights_shift = []
        self.weights_jump = []
        
    def startGame(self, inst):
        """
        start a new game for the given instance and make the first move because there is no previous move to use for training
        to train we need two consecutive moves of which the first move is evaluated

        Parameters
        ----------
        inst : instance of the game to start
        """
        self.episode_reward[inst] = {PLAYER1:0, PLAYER2:0} # reset reward for each player
        self.step[inst] = 1 # reset number of moves

        # Reset environment and get initial state
        self.GameMillLogic[inst].restartGame()
        self.field[inst], actionType, _, inStockTokens, remainingTokens = self.GameMillLogic[inst].getFullState() # initial state, empty field
        self.player[inst] = getPlayerFromActionType(actionType) # player 1 has to start
        self.inputVector[inst], self.phase[inst], self.move[inst] = buildInputVector(self.player[inst], self.field[inst], actionType, inStockTokens, remainingTokens) # phase is "set", move ist "true"
        
        validIdx = getPossibleMovesIndex(self.phase[inst], self.player[inst], self.field[inst], self.move[inst]) # start in phase set, all set moves possible

        # query a model for Q values to select the next action or use a random action depending on the decision of the epsilon greedy algorithm
        if np.random.random() >= self.epsilon:
            # Get action from model
            qVals = self.agent.get_qs(np.array([self.inputVector[inst]]), self.phase[inst])[0]
            selectedPossibleAction = np.argmax(qVals[validIdx])
        else:
            # Get random action
            selectedPossibleAction = np.random.randint(0, len(validIdx))
        selectedAction = validIdx[selectedPossibleAction] # select a set action
        
        newField = getMoveFromIndex(self.phase[inst], self.player[inst], self.field[inst], selectedAction) # start in phase set, new field with 1 token from player 1 
            
        self.lastInputVector[inst][self.player[inst]] = self.inputVector[inst] # save values of player 1's first move
        self.lastSelectedAction[inst][self.player[inst]] = selectedAction
        self.lastPhase[inst][self.player[inst]] = self.phase[inst]
            
        bActionValid = self.GameMillLogic[inst].setMove(newField) # make player 1's first move
        # if not bActionValid:
        #     print("ERROR: Invalid action!")
        self.field[inst], actionType, _, inStockTokens, remainingTokens = self.GameMillLogic[inst].getFullState() # field with 1 token from player 1 
        self.player[inst] = getPlayerFromActionType(actionType) # player has to 2 make the second move
        self.inputVector[inst], self.phase[inst], self.move[inst] = buildInputVector(self.player[inst], self.field[inst], actionType, inStockTokens, remainingTokens) # phase is "set", move ist "true"
        
    def stepGame(self):
        """
        make one move in all instances of the game
        start a new game in an instance if necessary
        call this function in a loop for training

        Returns
        -------
        finishedEpisodes : number of episodes/games finished in this step - normally 0 or 1, in extreme case numInstances
            sum up this value to get the total number of finished episodes

        """
        finishedEpisodes = 0
        
        # start new games in the instances where it is necessary
        for inst in range(self.numInstances):
            if False == self.gameStarted[inst]:
                self.startGame(inst)
                self.gameStarted[inst] = True

        # list of instances where Q values have to be predicted because no random move is selected   
        qsToPredict = np.random.random(self.numInstances) >= self.epsilon 
        
        # lists to select which instances are in which phase of the game
        setInstances = np.array(self.phase) == PHASE_SET
        shiftInstances = np.array(self.phase) == PHASE_SHIFT
        jumpInstances = np.array(self.phase) == PHASE_JUMP
        
        # lists to select the instances in which the Q values of the different phases should be predicted
        predictsSet = qsToPredict & setInstances
        predictsShift = qsToPredict & shiftInstances
        predictsJump = qsToPredict & jumpInstances

        qVals = [None]*self.numInstances # start with empty lists for Q values of all instances
        
        # use numpy array where input vectors of all instances as lists are the elements
        # then we can use list as indexes to filter out the correct input vectors
        # dtype=object is important because the length of the input vectors in the phases are different
        states = np.array(self.inputVector, dtype=object)


        # if np.any(predictsSet):
        #     firstInst = np.argmax(predictsSet)
        #     states_np = np.array(list(states[predictsSet]),dtype=int)
        #     qs = self.agent.get_qs(states_np, PHASE_SET) # PHASE_SET replaces self.phase[firstInst]
        #     idx=0
        #     for inst, bPredicted in enumerate(predictsSet):
        #         if True == bPredicted:
        #             qVals[inst] = qs[idx]
        #             idx += 1
        #     # qVals[predictsSet] = qs
                    
        # if np.any(predictsShift):
        #     firstInst = np.argmax(predictsShift)
        #     states_np = np.array(list(states[predictsShift]),dtype=int)
        #     qs = self.agent.get_qs(states_np, PHASE_SHIFT) 
        #     idx=0
        #     for inst, bPredicted in enumerate(predictsShift):
        #         if True == bPredicted:
        #             qVals[inst] = qs[idx]
        #             idx += 1
            
        # if np.any(predictsJump):
        #     firstInst = np.argmax(predictsJump)
        #     states_np = np.array(list(states[predictsJump]),dtype=int)
        #     qs = self.agent.get_qs(states_np, PHASE_JUMP)
        #     idx=0
        #     for inst, bPredicted in enumerate(predictsJump):
        #         if True == bPredicted:
        #             qVals[inst] = qs[idx]
        #             idx += 1
                    
        predictsList = [predictsSet, predictsShift, predictsJump]
        for predicts in predictsList: # calculate the Q values for all phases
            if np.any(predicts): # only calculate if at least one instance is selected
                firstInst = np.argmax(predicts) # the first instance where the corresponding element in predicts is True, used to get the correct phase
                states_np = np.array(list(states[predicts]),dtype=int) # all selected states are in the same phase with the same length -> we can use a normal numpy array
                qs = self.agent.get_qs(states_np, self.phase[firstInst]) # query the model of the selected phase for Q values 
                idx=0
                for inst, bPredicted in enumerate(predicts):
                    if True == bPredicted:
                        qVals[inst] = qs[idx] # order the predicted Q values to the corresponding instances
                        idx += 1
                    
   
                
        for inst in range(self.numInstances):
            # get the indices of all valid moves in the current state of the mill logic
            validIdx = getPossibleMovesIndex(self.phase[inst], self.player[inst], self.field[inst], self.move[inst])
            # the length of validIdx is the number of possible moves
            # an element of validIdx contains the index of a possible move out of all available moves
            
            if True == qsToPredict[inst]:
                # Get action from model
                selectedPossibleAction = np.argmax(qVals[inst][validIdx]) # use index of the maximum out of the Q values for valid moves
            else:
                # Get random action
                selectedPossibleAction = np.random.randint(0, len(validIdx)) # use a random index out of the length of valid moves
            
            selectedAction = validIdx[selectedPossibleAction] # select a set action by mapping the index of a possible action to the real index of the selected action
            
            # convert the index of an action to the new mill field as state to set into the mill logic
            newField = getMoveFromIndex(self.phase[inst], self.player[inst], self.field[inst], selectedAction)
            
            # store parameters that are needed after making the next move in the game
            self.lastInputVector[inst][self.player[inst]] = self.inputVector[inst]
            self.lastSelectedAction[inst][self.player[inst]] = selectedAction
            self.lastPhase[inst][self.player[inst]] = self.phase[inst]
                
            # set the new move into the mill logic
            bActionValid = self.GameMillLogic[inst].setMove(newField)
            if not bActionValid:
                print("ERROR: Invalid action!")
            # get the new game parameters
            self.field[inst], actionType, _, inStockTokens, remainingTokens = self.GameMillLogic[inst].getFullState() # all information expect possibleMoves needed
            self.player[inst] = getPlayerFromActionType(actionType) # the player who has to move next
            self.inputVector[inst], self.phase[inst], self.move[inst] = buildInputVector(self.player[inst], self.field[inst], actionType, inStockTokens, remainingTokens) # input for the model, game phase and move (if True, else remove)
            reward, self.lastActionTypes[inst], done = getReward(self.player[inst], actionType, self.lastActionTypes[inst], self.rewardList) # calculate reward of the made move, done=True if game is over
            
            # Count total reward of each player
            self.episode_reward[inst][self.player[inst]] += reward
    
            # Every step we update replay memory and train main network
            self.agent.update_replay_memory((self.lastInputVector[inst][self.player[inst]], self.lastSelectedAction[inst][self.player[inst]], reward, self.player[inst], self.inputVector[inst], self.field[inst], self.move[inst], self.phase[inst], done), self.lastPhase[inst][self.player[inst]]) # save (last state, last action, reward, player, state, field, move, phase, done) in replay memory of last phase
            self.agent.train(done, self.lastPhase[inst][self.player[inst]], self.step[inst]) # start a training cycle for the last phase (the transition from the last state to the current state is now finished)
    
            if True == done: # player won the game, the opponent has no next move to realize he has lost -> handle losing now
                reward = self.rewardList[-1] # reward=REWARD_LOSS for opponent
                self.player[inst] = PLAYER1 + PLAYER2 - self.player[inst] # get the player who lost
                
                # add loss to total reward of opponent
                self.episode_reward[inst][self.player[inst]] += reward
    
                # Every step we update replay memory and train main network
                # inputVector, field, move and phase are irrelevent because done is True
                self.agent.update_replay_memory((self.lastInputVector[inst][self.player[inst]], self.lastSelectedAction[inst][self.player[inst]], reward, self.player[inst], self.inputVector[inst], self.field[inst], self.move[inst], self.phase[inst], done), self.lastPhase[inst][self.player[inst]])
                self.agent.train(done, self.lastPhase[inst][self.player[inst]], self.step[inst])
    
            self.step[inst] += 1 # the step of the game is made
            
            if done or self.step[inst] >= self.maxSteps: # break game if it is over or nobody wins for too long
                # game finished
                self.gameStarted[inst] = False # in the next step a new game must be started 
                finishedEpisodes += 1 # an instance finished the game
                self.finishGame(inst) # store stats of this episode and decay epsilon
        
        return finishedEpisodes
            
            
        
    def finishGame(self, inst):
        """
        store the rewards from each player and the total number of steps in this episode
        decay epsilon at the end of this episode

        Parameters
        ----------
        inst : instance of the game to finish
        """
        # store episode stats
        self.ep_rewards_1 += [self.episode_reward[inst][PLAYER1]]
        self.ep_rewards_2 += [self.episode_reward[inst][PLAYER2]]
        self.ep_steps += [self.step[inst]]
        
        # store model stats
        weights_mean = self.agent.getMeanWeights()
        self.weights_set += [weights_mean[0]]
        self.weights_shift += [weights_mean[1]]
        self.weights_jump += [weights_mean[2]]
        
        self.ep_mse += [self.agent.getMeanSquaredError()]


        # Decay epsilon
        if self.epsilon > self.epsilonMin:
            self.epsilon *= self.epsilonDecay
            self.epsilon = max(self.epsilonMin, self.epsilon)

        
    def getGameStats(self, bFull=False):
        """
        get the rewards from each player and the total number of steps from the last episode or all episodes

        Parameters
        ----------
        bFull : bool, optional
            True : return the statistic of all finished episodes
            False : default, return only the statistic of the last finished episode

        Returns
        -------
        ep_rewards_1
            total episode reward of player 1
        ep_rewards_2
            total episode reward of player 2
        ep_steps
            total episode steps

        """
        if bFull: # return stats of all played games
            # self.ep_steps = list((np.array(self.weights_set)+np.array(self.weights_shift)+np.array(self.weights_jump))/3)
            # self.ep_steps = self.ep_mse
            return self.ep_rewards_1, self.ep_rewards_2, self.ep_steps
        else: # return stats of last played game
            return self.ep_rewards_1[-1], self.ep_rewards_2[-1], self.ep_steps[-1]
        
    def getModelWeights(self, bTotalMean=True):
        """
        get the mean absolute weights of the models for all phase from all episodes

        Parameters
        ----------
        bTotalMean : bool, optional
            True : default, return the mean value of the mean weights of the models for the three phases
            False : return the mean weights of the models for all three phases individually

        Returns
        -------
        mean weights of the different models or mean weight over all models
        """
        if bTotalMean:
            return list((np.array(self.weights_set)+np.array(self.weights_shift)+np.array(self.weights_jump))/3)
        else:
            return self.weights_set, self.weights_shift, self.weights_jump
        
    def getMeanSquaredErrors(self, bFull=False):
        """
        get the MSE (mean squared error) from the last episode or all episodes

        Parameters
        ----------
        bFull : bool, optional
            True : return the MSE of all finished episodes
            False : default, return only the MSE of the last finished episode
        """
        if bFull: # return stats of all played games
            return self.ep_mse
        else: # return stats of last played game
            return self.ep_mse[-1]
        
        
    def selectModelsToTrain(self, bSet, bShift, bJump):
        """
        select which models shall be trained

        Parameters
        ----------
        bSet : bool
            True : model set will be trained
            False : model set will not be changed
        bShift : bool
            True : model shift will be trained
            False : model shift will not be changed
        bJump : bool
            True : model shift will be trained
            False : model shift will not be changed
        """
        self.agent.selectModelsToTrain(bSet, bShift, bJump)
                  
    def setModels(self, models):
        """
        set models to be used for training in the DQN agent

        Parameters
        ----------
        models : [modelSet, modelShift, modelJump]
            modelSet : model to use for set phase
            modelShift : model to use for shift phase
            modelJump : model to use for jump phase

        Returns
        -------
        bOkay : bool
            True : using the models was successful
            False : at least one model is invalid
        """
        modelSet, modelShift, modelJump = models
        # modelSet = tf.keras.models.load_model('models_test/agent1_model_set.h5')
        # modelShift = tf.keras.models.load_model('models_test/agent1_model_shift.h5')
        # modelJump = tf.keras.models.load_model('models_test/agent1_model_jump.h5')
        bOkay = self.agent.setModels(modelSet, modelShift, modelJump)
        return bOkay

    
    def getModels(self):
        """
        get the models for all phases from the DQN agent and returns the models
        """
        # modelSet, modelShift, modelJump = self.agent.getModels()
        # modelSet.save('models_test/agent1_model_set.h5')
        # modelShift.save('models_test/agent1_model_shift.h5')
        # modelJump.save('models_test/agent1_model_jump.h5')

        return self.agent.getModels()
            




################################################################## Validation class #################################################################
class DQNValidation:
    # bCreateRandomReference=True: generate a reference agent with random weights
    def __init__(self, bCreateRandomReference=True, numInstances=1):
        """
        initialize the validation where a trained agent plays against a new or existing reference agent
        loading models for the trained agent is necessary
        loading models for the reference agent is optional, 
            instead bRandomReferencePlayer=True in configureValidation() or bCreateRandomReference=True can be used

        Parameters
        ----------
        bCreateRandomReference : bool
            True : new models with random weights will be created from the reference agent
            False : no models will be created and it is necessary to load models before using the reference agent
            optional. The default is True.
        numInstances : number of games to play at the same time for validation
            optional. The default is 1.
        """
        
        self.numInstances = numInstances
        self.gameStarted = [False]*self.numInstances # start new games for all instances
        
        
        # create defined number of instances of the game
        self.GameMillLogic = [None]*self.numInstances
        for inst in range(self.numInstances):
            self.GameMillLogic[inst] = MillLogic(checkMoves=False) # don't check moves in validation, forbidden moves are already prevented

        # create two agents to play against each other
        dummyReplayMemSize=10 # use dummy value, because training will not be used
        dummyLearningRate=LEARNING_RATE # use dummy value, because training will not be used
        self.agentTrained = DQNAgent(bCreateModels=False, replayMemSize=dummyReplayMemSize, learningRate=dummyLearningRate) # don't create models, load trained models instead
        self.agentReference = DQNAgent(bCreateModels=bCreateRandomReference, replayMemSize=dummyReplayMemSize, learningRate=dummyLearningRate)
        self.agents = [None]*self.numInstances # save the assignment of the agents to the players for each instance
        self.playerAgents = [None]*self.numInstances # save the assignment of the players to the agents for each instance
        for inst in range(self.numInstances):
            if 0 == inst%2: # agentTrained starts half of the games (0,2,4,...)
                self.agents[inst] = {PLAYER1:self.agentTrained, PLAYER2:self.agentReference} # agentTrained starts
                self.playerAgents[inst] = {AGENT_TRAINED:PLAYER1, AGENT_REFERENCE:PLAYER2}
            else: # (1,3,5,...)
                self.agents[inst] = {PLAYER1:self.agentReference, PLAYER2:self.agentTrained} # agentReference starts
                self.playerAgents[inst] = {AGENT_TRAINED:PLAYER2, AGENT_REFERENCE:PLAYER1}
        
        # default configuration parameters, can be adapted with function configureValidation()
        # maximum steps before breaking a game
        self.maxSteps = MAX_STEPS
        # True: the first move of player 1 is random to have different games
        self.bFirstMoveRandom = False
        # True: the reference agent always makes random moves
        self.bRandomReferencePlayer = False
        
        # default rewards, can be adapted with function configureRewards() 
        self.rewardList=[REWARD_MOVE, REWARD_OWNMILL, REWARD_ENEMYMILL, REWARD_WIN, REWARD_LOSS]
        
        # game parameters for all instances
        self.episode_reward = [None]*self.numInstances # reward of the current game for each player
        self.step = [None]*self.numInstances # number of moves in the current episode
        self.field = [None]*self.numInstances # mill field
        self.player = [None]*self.numInstances # player who has to move
        self.inputVector = [None]*self.numInstances # input vector for the NN models
        self.phase = [None]*self.numInstances # game phase (PHASE_SET, PHASE_SHIFT, PHASE_JUMP)
        self.move = [None]*self.numInstances # bool, True for normal move, False for remove action
        
        # store relevant old game parameters
        self.lastActionTypes = [None]*self.numInstances
        for inst in range(self.numInstances):
            self.lastActionTypes[inst] = {PLAYER1:enAction.SetPlayer1, PLAYER2:enAction.SetPlayer2} # save type of the last action of both players for calculating the reward
        
        # validation stats 
        self.ep_rewards_1 = [] # rewards from player 1 over all episodes
        self.ep_rewards_2 = [] # rewards from player 2 over all episodes
        self.ep_steps = [] # total number of moves from all episodes
        self.wins = 0 # number of wins
        self.losses = 0 # number of losses
        self.draws = 0 # number of draws
        
        
    def configureValidation(self, maxSteps=400, bFirstMoveRandom=False, bRandomReferencePlayer=False):
        """
        set different parameters to configure the validation

        Parameters
        ----------
        maxSteps : maximum steps before breaking a game
            optional. The default is 400.
        bFirstMoveRandom : bool, optional
            True: the first move of player 1 is random to have different games
            False: default, the agent who starts the game predicts the first move with the model for set phase
        bRandomReferencePlayer : bool, optional
            True: the reference agent always makes random moves
            False: default, the reference agent predicts the moves with his models
        """
        self.maxSteps = maxSteps
        self.bFirstMoveRandom = bFirstMoveRandom
        self.bRandomReferencePlayer = bRandomReferencePlayer
        
    '''
    rewardList=[REWARD_MOVE, REWARD_OWNMILL, REWARD_ENEMYMILL, REWARD_WIN, REWARD_LOSS]
    '''
    def configureRewards(self, rewardList):
        """
        set the reward values for all different situations

        Parameters
        ----------
        rewardList : [REWARD_MOVE, REWARD_OWNMILL, REWARD_ENEMYMILL, REWARD_WIN, REWARD_LOSS]
            REWARD_MOVE : normal move without special consequences, should be negative and small
            REWARD_OWNMILL : a own mill was created, should be positive and moderate
            REWARD_ENEMYMILL : the opponent has created a mill, should be negative and moderate
            REWARD_WIN : game won, should be positive and high
            REWARD_LOSS : game lost, should be negative and high
        """
        self.rewardList = rewardList
        
    def initializeGame(self):
        """
        begin with new games in all instances and reset the lists for validation statistic
        """
        self.gameStarted = [False]*self.numInstances
        # reset validation stats 
        self.ep_rewards_1 = []
        self.ep_rewards_2 = []
        self.ep_steps = []
        self.wins = 0
        self.losses = 0
        self.draws = 0
        
    def startGame(self, inst):
        """
        start a new game for the given instance and make the first move because there is no previous move to calculate the reward
        the reward is always calculated for the previous move

        Parameters
        ----------
        inst : instance of the game to start
        """
        self.episode_reward[inst] = {PLAYER1:0, PLAYER2:0} # reset reward for each player
        self.step[inst] = 1 # reset number of moves

        # Reset environment and get initial state
        self.GameMillLogic[inst].restartGame()
        self.field[inst], actionType, _, inStockTokens, remainingTokens = self.GameMillLogic[inst].getFullState() # initial state, empty field
        self.player[inst] = getPlayerFromActionType(actionType) # player 1 has to start
        self.inputVector[inst], self.phase[inst], self.move[inst] = buildInputVector(self.player[inst], self.field[inst], actionType, inStockTokens, remainingTokens) # phase is "set", move ist "true"
        
        validIdx = getPossibleMovesIndex(self.phase[inst], self.player[inst], self.field[inst], self.move[inst]) # start in phase set, all set moves possible

        bRefOnMove = self.agents[inst][self.player[inst]] == self.agentReference # check if agentReference hast to move
        if (self.bFirstMoveRandom or # first move is always random or
            (self.bRandomReferencePlayer and bRefOnMove) # reference player makes random moves and reference player has to move
            ):
            # Get random action as first move to have different games during validation (without epsilon greedy algorithm all games would be the same)
            # or random action of reference player
            selectedPossibleAction = np.random.randint(0, len(validIdx))
        else:
            # Get action from model
            qVals = self.agents[inst][self.player[inst]].get_qs(np.array([self.inputVector[inst]]), self.phase[inst])[0]
            selectedPossibleAction = np.argmax(qVals[validIdx])
        
        selectedAction = validIdx[selectedPossibleAction] # select a set action
        
        newField = getMoveFromIndex(self.phase[inst], self.player[inst], self.field[inst], selectedAction) # start in phase set, new field with 1 token from player 1 
            
        self.GameMillLogic[inst].setMove(newField) # make player 1's first move
        self.field[inst], actionType, _, inStockTokens, remainingTokens = self.GameMillLogic[inst].getFullState() # field with 1 token from player 1 
        self.player[inst] = getPlayerFromActionType(actionType) # player has to 2 make the second move
        self.inputVector[inst], self.phase[inst], self.move[inst] = buildInputVector(self.player[inst], self.field[inst], actionType, inStockTokens, remainingTokens) # phase is "set", move ist "true"
        
    def stepGame(self, bSwitchPlayers=True):
        """
        make one move in all instances of the game
        start a new game in an instance if necessary
        call this function in a loop for training

        Parameters
        ----------
        bSwitchPlayers : bool, optional
            True : default, after the game in an instance is finished, the agents switch the players they play as in this instance
            False : the agents in the same instance always play as the same player but in different instances they are still different players
            
        Returns
        -------
        finishedEpisodes : number of episodes/games finished in this step - normally 0 or 1, in extreme case numInstances
            sum up this value to get the total number of finished episodes

        """
        finishedEpisodes = 0
        
        # start new games in the instances where it is necessary
        for inst in range(self.numInstances):
            if False == self.gameStarted[inst]:
                self.startGame(inst)
                self.gameStarted[inst] = True
                
           
        agent1OnMove = np.array(self.player) == np.array([self.playerAgents[inst][AGENT_TRAINED] for inst in range(self.numInstances)]) # list of instances where agentTrained has to move
        agent1Predicts = agent1OnMove # list of instances where agentTrained predicts the move
        agent2Predicts = ~agent1OnMove # list of instances where agentReference predicts the move (if random moves not selected)
        
        # lists to select which instances are in which phase of the game
        setInstances = np.array(self.phase) == PHASE_SET
        shiftInstances = np.array(self.phase) == PHASE_SHIFT
        jumpInstances = np.array(self.phase) == PHASE_JUMP
        
        # lists to select the instances in which the Q values of the different phases should be predicted from agentTrained
        agent1PredictsSet = agent1Predicts & setInstances
        agent1PredictsShift = agent1Predicts & shiftInstances
        agent1PredictsJump = agent1Predicts & jumpInstances
        # lists to select the instances in which the Q values of the different phases should be predicted from agentReference
        agent2PredictsSet = agent2Predicts & setInstances
        agent2PredictsShift = agent2Predicts & shiftInstances
        agent2PredictsJump = agent2Predicts & jumpInstances
        
        qVals = [None]*self.numInstances # start with empty lists for Q values of all instances
        
        # use numpy array where input vectors of all instances as lists are the elements
        # then we can use list as indexes to filter out the correct input vectors
        # dtype=object is important because the length of the input vectors in the phases are different
        states = np.array(self.inputVector, dtype=object)
        
        # if np.any(agent1PredictsSet):
        #     firstInst = np.argmax(agent1PredictsSet)
        #     states_np = np.array(list(states[agent1PredictsSet]),dtype=int)
        #     qs = self.agents[firstInst][self.player[firstInst]].get_qs(states_np, PHASE_SET) # PHASE_SET replaces self.phase[firstInst]
        #     idx=0
        #     for inst, bPredicted in enumerate(agent1PredictsSet):
        #         if True == bPredicted:
        #             qVals[inst] = qs[idx]
        #             idx += 1
                    
        # if np.any(agent1PredictsShift):
        #     firstInst = np.argmax(agent1PredictsShift)
        #     states_np = np.array(list(states[agent1PredictsShift]),dtype=int)
        #     qs = self.agents[firstInst][self.player[firstInst]].get_qs(states_np, PHASE_SHIFT) 
        #     idx=0
        #     for inst, bPredicted in enumerate(agent1PredictsShift):
        #         if True == bPredicted:
        #             qVals[inst] = qs[idx]
        #             idx += 1
            
        # if np.any(agent1PredictsJump):
        #     firstInst = np.argmax(agent1PredictsJump)
        #     states_np = np.array(list(states[agent1PredictsJump]),dtype=int)
        #     qs = self.agents[firstInst][self.player[firstInst]].get_qs(states_np, PHASE_JUMP)
        #     idx=0
        #     for inst, bPredicted in enumerate(agent1PredictsJump):
        #         if True == bPredicted:
        #             qVals[inst] = qs[idx]
        #             idx += 1
         
        # calculate the Q values for agentTrained           
        predictsList = [agent1PredictsSet, agent1PredictsShift, agent1PredictsJump]
        for predicts in predictsList: # calculate the Q values for all phases
            if np.any(predicts): # only calculate if at least one instance is selected
                firstInst = np.argmax(predicts) # the first instance where the corresponding element in predicts is True, used to get the correct phase
                states_np = np.array(list(states[predicts]),dtype=int) # all selected states are in the same phase with the same length -> we can use a normal numpy array
                qs = self.agentTrained.get_qs(states_np, self.phase[firstInst]) # query the model of the selected phase for Q values 
                idx=0
                for inst, bPredicted in enumerate(predicts):
                    if True == bPredicted:
                        qVals[inst] = qs[idx] # order the predicted Q values to the corresponding instances
                        idx += 1
        
        if False == self.bRandomReferencePlayer: # calc Q values for reference player only when necessary
        #     if np.any(agent2PredictsSet):
        #         firstInst = np.argmax(agent2PredictsSet)
        #         states_np = np.array(list(states[agent2PredictsSet]),dtype=int)
        #         qs = self.agents[firstInst][self.player[firstInst]].get_qs(states_np, PHASE_SET)
        #         idx=0
        #         for inst, bPredicted in enumerate(agent2PredictsSet):
        #             if True == bPredicted:
        #                 qVals[inst] = qs[idx]
        #                 idx += 1
        #         # qVals[agent2PredictsSet] = qs
                
        #     if np.any(agent2PredictsShift):
        #         firstInst = np.argmax(agent2PredictsShift)
        #         states_np = np.array(list(states[agent2PredictsShift]),dtype=int)
        #         qs = self.agents[firstInst][self.player[firstInst]].get_qs(states_np, PHASE_SHIFT)
        #         idx=0
        #         for inst, bPredicted in enumerate(agent2PredictsShift):
        #             if True == bPredicted:
        #                 qVals[inst] = qs[idx]
        #                 idx += 1
                
        #     if np.any(agent2PredictsJump):
        #         firstInst = np.argmax(agent2PredictsJump)
        #         states_np = np.array(list(states[agent2PredictsJump]),dtype=int)
        #         qs = self.agents[firstInst][self.player[firstInst]].get_qs(states_np, PHASE_JUMP)
        #         idx=0
        #         for inst, bPredicted in enumerate(agent2PredictsJump):
        #             if True == bPredicted:
        #                 qVals[inst] = qs[idx]
        #                 idx += 1
                
            # calculate the Q values for agentReference       
            predictsList = [agent2PredictsSet, agent2PredictsShift, agent2PredictsJump]
            for predicts in predictsList: # calculate the Q values for all phases
                if np.any(predicts): # only calculate if at least one instance is selected
                    firstInst = np.argmax(predicts) # the first instance where the corresponding element in predicts is True, used to get the correct phase
                    states_np = np.array(list(states[predicts]),dtype=int) # all selected states are in the same phase with the same length -> we can use a normal numpy array
                    qs = self.agentReference.get_qs(states_np, self.phase[firstInst]) # query the model of the selected phase for Q values 
                    idx=0
                    for inst, bPredicted in enumerate(predicts):
                        if True == bPredicted:
                            qVals[inst] = qs[idx] # order the predicted Q values to the corresponding instances
                            idx += 1
        
                
        for inst in range(self.numInstances):
            # get the indices of all valid moves in the current state of the mill logic
            validIdx = getPossibleMovesIndex(self.phase[inst], self.player[inst], self.field[inst], self.move[inst])
            # the length of validIdx is the number of possible moves
            # an element of validIdx contains the index of a possible move out of all available moves
            
            bRefOnMove = self.agents[inst][self.player[inst]] == self.agentReference # check if agentReference has to move
            if (self.bRandomReferencePlayer and bRefOnMove): # reference player makes random moves and reference player has to move
                # Get random action as first move to have different games during validation (without epsilon greedy algorithm all games would be the same)
                # or random action of reference player
                selectedPossibleAction = np.random.randint(0, len(validIdx)) # use a random index out of the length of valid moves
            else:
                # Get action from model
                selectedPossibleAction = np.argmax(qVals[inst][validIdx]) # use index of the maximum out of the Q values for valid moves
            
            selectedAction = validIdx[selectedPossibleAction] # select a set action by mapping the index of a possible action to the real index of the selected action
            
            # convert the index of an action to the new mill field as state to set into the mill logic
            newField = getMoveFromIndex(self.phase[inst], self.player[inst], self.field[inst], selectedAction)
              
            # set the new move into the mill logic
            bActionValid = self.GameMillLogic[inst].setMove(newField)
            # if not bActionValid:
            #     print("ERROR: Invalid action!")
            # get the new game parameters
            self.field[inst], actionType, _, inStockTokens, remainingTokens = self.GameMillLogic[inst].getFullState() # all information expect possibleMoves needed
            self.player[inst] = getPlayerFromActionType(actionType) # the player who has to move next
            self.inputVector[inst], self.phase[inst], self.move[inst] = buildInputVector(self.player[inst], self.field[inst], actionType, inStockTokens, remainingTokens) # input for the model, game phase and move (if True, else remove)
            reward, self.lastActionTypes[inst], done = getReward(self.player[inst], actionType, self.lastActionTypes[inst], self.rewardList) # calculate reward of the made move, done=True if game is over
            
            # Count total reward of each player
            self.episode_reward[inst][self.player[inst]] += reward
    
            if True == done: # player won the game, the opponent has no next move to realize he has lost -> handle losing now
                if self.agents[inst][self.player[inst]] == self.agentTrained:   # trained model won
                    self.wins += 1
                else:   # trained model lost
                    self.losses += 1
            
                reward = self.rewardList[-1] # reward=REWARD_LOSS for opponent
                self.player[inst] = PLAYER1 + PLAYER2 - self.player[inst] # get the player who lost
                
                # add loss to total reward of opponent
                self.episode_reward[inst][self.player[inst]] += reward
    
            self.step[inst] += 1 # the step of the game is made
            
            if self.step[inst] >= self.maxSteps: # game will be finished because maximum number of moves is reached -> it is a draw
                self.draws += 1
            
            if done or self.step[inst] >= self.maxSteps: # someone won or maximum number of moves is reached
                # game finished
                self.gameStarted[inst] = False # in the next step a new game must be started 
                finishedEpisodes += 1 # an instance finished the game
                self.finishGame(inst) # store stats of this episode
                if bSwitchPlayers:
                    self.switchPlayers(inst) # switch the players as which the agents play if feature is active
        
        return finishedEpisodes
            
            
        
    def finishGame(self, inst):
        """
        store the rewards from each agent and the total number of steps in this episode

        Parameters
        ----------
        inst : instance of the game to finish
        """
        # store episode stats
        self.ep_rewards_1 += [self.episode_reward[inst][self.playerAgents[inst][AGENT_TRAINED]]]
        self.ep_rewards_2 += [self.episode_reward[inst][self.playerAgents[inst][AGENT_REFERENCE]]]
        self.ep_steps += [self.step[inst]]
        
      
        
    def setPlayerAgent1(self, player, inst):
        """
        do not call inside a game
        agentTrained will play as the given player, agentReference will play as the opponent

        Parameters
        ----------
        player : agentTrained will play as this player
        inst : instance of the game where the agents are to be assigned to the players 
        """
        self.playerAgents[inst][AGENT_TRAINED] = player
        self.playerAgents[inst][AGENT_REFERENCE] = PLAYER1+PLAYER2-player
        self.agents[inst][player] = self.agentTrained
        self.agents[inst][PLAYER1+PLAYER2-player] = self.agentReference

    def switchPlayers(self, inst):
        """
        do not call inside a game
        agentTrained and agentReference will switch the player they play -> the agent who starts the game changes

        Parameters
        ----------
        inst : instance of the game where the agents are switch the player they play

        """
        # set agentTrained as the player who agentReference actually is
        self.setPlayerAgent1(self.playerAgents[inst][AGENT_REFERENCE], inst)

        
    def getGameStats(self, bFull=False):
        """
        get the rewards from each agent and the total number of steps from the last episode or all episodes

        Parameters
        ----------
        bFull : bool, optional
            True : return the statistic of all finished episodes
            False : default, return only the statistic of the last finished episode

        Returns
        -------
        ep_rewards_1
            total episode reward of agentTrained
        ep_rewards_2
            total episode reward of agentReference
        ep_steps
            total episode steps

        """
        if bFull: # return stats of all played games
            return self.ep_rewards_1, self.ep_rewards_2, self.ep_steps
        else: # return stats of last played game
            return self.ep_rewards_1[-1], self.ep_rewards_2[-1], self.ep_steps[-1]
        
    def getGameResults(self):
        """
        get the results (wins, losses, draws) of all played games

        Returns
        -------
        wins
            number of wins of the trained againt (agent to validate)
        losses
            number of losses of the trained agent (agent to validate)
        draws
            number of draws

        """
        return self.wins, self.losses, self.draws
        
       
    # def loadModels(self, agent):           
    def setModels(self, agent, models):
        """
        set models to be used for agentTrained or agentReference in validation

        Parameters
        ----------
        agent : select the agent
            AGENT_TRAINED : use the given models as models for agentTrained
            AGENT_REFERENCE : use the given models as models for agentReference
        models : [modelSet, modelShift, modelJump]
            modelSet : model to use for set phase
            modelShift : model to use for shift phase
            modelJump : model to use for jump phase

        Returns
        -------
        bOkay : bool
            True : using the models was successful
            False : at least one model is invalid
        """
        modelSet, modelShift, modelJump = models
        if AGENT_TRAINED == agent:
            # modelSet = tf.keras.models.load_model('models_test/agent1_model_set.h5')
            # modelShift = tf.keras.models.load_model('models_test/agent1_model_shift.h5')
            # modelJump = tf.keras.models.load_model('models_test/agent1_model_jump.h5')
            bOkay = self.agentTrained.setModels(modelSet, modelShift, modelJump)
        elif AGENT_REFERENCE == agent:
            # modelSet = tf.keras.models.load_model('models_test/agent2_model_set.h5')
            # modelShift = tf.keras.models.load_model('models_test/agent2_model_shift.h5')
            # modelJump = tf.keras.models.load_model('models_test/agent2_model_jump.h5')
            bOkay = self.agentReference.setModels(modelSet, modelShift, modelJump)
        else:
            print('Invalid agent !!!')
            
        return bOkay
  




################################################################## Player class #################################################################
class DQNPlayer:
    def __init__(self, player, models=None):
        """
        initialize a ai player for the game mill

        Parameters
        ----------
        player : the player as which the instance play
        models : directly load models into the player
                ATTENTION - invalid models are not recognized, use setModels() to avoid errors
            optional. The default is None.

        Returns
        -------
        None.

        """
        self.player = player # the instance knows which player it has to play
        
        if models != None: # use models if they are given
            self.modelSet, self.modelShift, self.modelJump = models

    def setModels(self, models):
        """
        set models to be used by the player

        Parameters
        ----------
        models : [modelSet, modelShift, modelJump]
            modelSet : model to use for set phase
            modelShift : model to use for shift phase
            modelJump : model to use for jump phase

        Returns
        -------
        bOkay : bool
            True : using the models was successful
            False : at least one model is invalid

        """
        bOkay = True
        self.modelSet, self.modelShift, self.modelJump = models # take over the models
        try: # try a simple action to validate using the models is possible
            self.modelSet.get_weights()
            self.modelShift.get_weights()
            self.modelJump.get_weights()
        except:
            bOkay = False
            
        return bOkay
        
    def setPlayer(self, player):
        """
        select if the instance plays player 1 or player 2

        Parameters
        ----------
        player : the player as which the instance play
        """
        self.player = player
        
    def getMove(self, fullState):
        """
        

        Parameters
        ----------
        fullState : [field, actionType, _, inStockTokens, remainingTokens]
            all available information from the mill logic
            use fullState = GameMillLogic.getFullState() to get this

        Returns
        -------
        bValid : bool
            True : the player is on move and the given newField makes sense
            False : the player is not on move and the given newField makes no sense
        newField : array with shape (3, 8)
            new state of the millField
        """
        bValid = True
        field, actionType, _, inStockTokens, remainingTokens = fullState # extract the information from the mill logic
        player = getPlayerFromActionType(actionType) # get the player who has to make the next move from the actionType
        if player != self.player: # if the players don't match, the called instance is not on move
            bValid = False
            
        # get the input for the model, the game phase and if a normal move or a remove action is required
        inputVector, phase, move = buildInputVector(self.player, field, actionType, inStockTokens, remainingTokens)
        
        # get the indices of all valid moves in the current state of the mill logic
        validIdx = getPossibleMovesIndex(phase, self.player, field, move)        
        
        # query the model of the current game phase for Q values
        states_tf = tf.cast(tf.convert_to_tensor(np.array([inputVector])), dtype=tf.float32)
        if PHASE_SET == phase:
            qVals = self.modelSet.predict(states_tf)[0]
        elif PHASE_SHIFT == phase:
            qVals = self.modelShift.predict(states_tf)[0]
        else: # PHASE_JUMP
            qVals = self.modelJump.predict(states_tf)[0]
        
        # get action from model by using the index of the maximum out of the Q values for valid moves
        selectedPossibleAction = np.argmax(qVals[validIdx])
        # select a set action by mapping the index of a possible action to the real index of the selected action
        selectedAction = validIdx[selectedPossibleAction]
        
        # convert the index of an action to the new mill field as state to set into the mill logic
        newField = getMoveFromIndex(phase, self.player, field, selectedAction)

        return bValid, newField








if __name__=='__main__': 
    ''' Play '''
    # modelSet = tf.keras.models.load_model('models_test/agent1_model_set.h5')
    # modelShift = tf.keras.models.load_model('models_test/agent1_model_shift.h5')
    # modelJump = tf.keras.models.load_model('models_test/agent1_model_jump.h5')
    # playerA = DQNPlayer(PLAYER1)
    # bOkay = playerA.setModels([modelSet, modelShift, modelJump])
    
    # modelSet = tf.keras.models.load_model('models_test/agent2_model_set.h5')
    # modelShift = tf.keras.models.load_model('models_test/agent2_model_shift.h5')
    # modelJump = tf.keras.models.load_model('models_test/agent2_model_jump.h5')
    # playerB = DQNPlayer(PLAYER2)
    # bOkay = playerB.setModels([modelSet, modelShift, modelJump])

    # GameMillLogic = MillLogic(checkMoves=True)
    
    # actionType = enAction.SetPlayer1  
    # while enAction.Player1Wins != actionType and enAction.Player2Wins != actionType:
    #     fullState = GameMillLogic.getFullState()
    #     field, actionType, _, inStockTokens, remainingTokens = fullState
    #     print(fullState[0])
    #     input()
    #     actionType = fullState[1]
    #     if actionType.value % 2: # Player 2(B)
    #         bValid, newField = playerB.getMove(fullState)
    #     else: # Player 1(A)
    #         bValid, newField = playerA.getMove(fullState)
    #     if not bValid:
    #             print("ERROR: Wrong Player selected !!!")
        
    #     GameMillLogic.setMove(newField)
    
    # TODO Markierung
    ''' Training 1 Agent '''
    trainingDQN = DQNTraining(bCreateModels=True, numInstances=1, replayMemSize=REPLAY_MEMORY_SIZE, learningRate=LEARNING_RATE)
    # bOkay = trainingDQN.setModels(models=[modelSet, modelShift, modelJump])
    trainingDQN.selectModelsToTrain(bSet=False, bShift=True, bJump=False)
    trainingDQN.configureTraining(epsilonStart=1, epsilonDecay=1, epsilonMin=0)
    pbar = tqdm(total=EPISODES, ascii=True, unit='episodes', position=0)
    # pbar = tqdm(total=EPISODES, ascii=True, unit='episodes')
    episode = 0
    trainingDQN.initializeGame()
    while episode < EPISODES:
        # print(f"Episode {episode}")
        finishedEpisodes = trainingDQN.stepGame()
        episode += finishedEpisodes
        if finishedEpisodes > 0:
            if episode > EPISODES: # more episodes trained than required, not displayable in process bar
                finishedEpisodes -= episode-EPISODES # only update until required episodes reached
            pbar.update(finishedEpisodes)
            
    pbar.close()
    ep_rewards_1, ep_rewards_2, ep_steps = trainingDQN.getGameStats(bFull=True)
    weights_mean = trainingDQN.getModelWeights(bTotalMean=True)
    ep_mse = trainingDQN.getMeanSquaredErrors(bFull=True)
    print(ep_mse)
    # trainingDQN.getModels()
    
    
    ''' Training 2 Agents '''
    # trainingDQN2 = DQNTraining2(bCreateModels=True, numInstances=32, replayMemSize=REPLAY_MEMORY_SIZE)
    # # bOkay = trainingDQN2.setModels(agent=AGENT1, models=[modelSet, modelShift, modelJump])
    # # bOkay = trainingDQN2.setModels(agent=AGENT2, models=[modelSet, modelShift, modelJump])
    # trainingDQN2.selectModelsToTrain(agent=AGENT1, bSet=False, bShift=False, bJump=False)
    # trainingDQN2.selectModelsToTrain(agent=AGENT2, bSet=False, bShift=False, bJump=False)
    # trainingDQN2.configureTraining(epsilonStart=0.5, epsilonDecay=1, epsilonMin=0)
    # pbar = tqdm(total=EPISODES, ascii=True, unit='episodes', position=0)
    # # pbar = tqdm(total=EPISODES, ascii=True, unit='episodes')
    # episode = 0
    # trainingDQN2.initializeGame()
    # while episode < EPISODES:
    #     # print(f"Episode {episode}")
    #     finishedEpisodes = trainingDQN2.stepGame(bSwitchPlayers=False)
    #     episode += finishedEpisodes
    #     if finishedEpisodes > 0:
    #         if episode > EPISODES: # more episodes trained than required, not displayable in process bar
    #             finishedEpisodes -= episode-EPISODES # only update until required episodes reached
    #         pbar.update(finishedEpisodes)
            
    # pbar.close()
    # ep_rewards_1, ep_rewards_2, ep_steps = trainingDQN2.getGameStats(bFull=True)
    # # trainingDQN.getModels(agent=AGENT1)
    # # trainingDQN.getModels(agent=AGENT2)
    
    
    
    ''' Validation '''
    validationDQN = DQNValidation(bCreateRandomReference=False, numInstances=40)
    # modelSet = tf.keras.models.load_model('models_test/model1.h5')
    # modelShift = tf.keras.models.load_model('models_test/model2.h5')
    # modelJump = tf.keras.models.load_model('models_test/model3.h5')
    # bOkay = validationDQN.setModels(agent=AGENT_TRAINED, models=[modelSet, modelShift, modelJump])
    # if bOkay == False:
    #     print('Invalid model')
    #     sys.exit(1)
    # bOkay = validationDQN.setModels(agent=AGENT_REFERENCE, models=[modelSet, modelShift, modelJump])
    # validationDQN.configureValidation(maxSteps=MAX_STEPS, bFirstMoveRandom=True, bRandomReferencePlayer=False)
    # pbar = tqdm(total=EPISODES, ascii=True, unit='episodes', position=0)
    # # pbar = tqdm(total=EPISODES, ascii=True, unit='episodes')
    # episode = 0
    # validationDQN.initializeGame()
    # while episode < EPISODES:
    #     # print(f"Episode {episode}")
    #     finishedEpisodes = validationDQN.stepGame(bSwitchPlayers=True)
    #     episode += finishedEpisodes
    #     if finishedEpisodes > 0:
    #         if episode > EPISODES: # more episodes trained than required, not displayable in process bar
    #             finishedEpisodes -= episode-EPISODES # only update until required episodes reached
    #         pbar.update(finishedEpisodes)
            
    # pbar.close()
    # ep_rewards_1, ep_rewards_2, ep_steps = validationDQN.getGameStats(bFull=True)
    n_wins, n_losses, n_draws = validationDQN.getGameResults()
    
    
    
    



'''
20 Episoden nur get_qs: 425 s
40 Episoden (davon 20-28 trainiert) nur train: 97 s 




mehrere Spiele gleichzeitig durchführen
Annahmen: 
    set: 10%, shift: 80%, jump 10%
    agent1: 50%, agent2: 50%

16 Spiele gleichzeitig
Zu erwartende benötigte Modelle:
2*(1-0.6**16)+4*(1-0.95**16)
= 4.238929103411447

Vorraussichtlicher Beschleunigungsfaktor get_qs:
16 / 4.238929103411447 = 3.77453824059557

32 Spiele gleichzeitig
Zu erwartende benötigte Modelle:
2*(1-0.6**32)+4*(1-0.95**32)
= 5.2251539029927745

Vorraussichtlicher Beschleunigungsfaktor get_qs:
32 / 5.2251539029927745 = 6.124221524206509


'''


















    
    
# ################################################################## Training class #################################################################
# class DQNTraining2:
#     def __init__(self, bCreateModels=True, numInstances=1, replayMemSize=REPLAY_MEMORY_SIZE):
        
#         self.numInstances = numInstances
#         self.gameStarted = [False]*self.numInstances
        
        
#         # create a instance of the game
#         self.GameMillLogic = [None]*self.numInstances
#         for inst in range(self.numInstances):
#             self.GameMillLogic[inst] = MillLogic(checkMoves=False)

#         # create two agents to play against each other
#         self.agent1 = DQNAgent(bCreateModels, replayMemSize)
#         self.agent2 = DQNAgent(bCreateModels, replayMemSize)
#         self.agents = [None]*self.numInstances
#         self.playerAgents = [None]*self.numInstances
#         for inst in range(self.numInstances):
#             if 0 == inst%2: # agent1 starts half of the games
#                 self.agents[inst] = {PLAYER1:self.agent1, PLAYER2:self.agent2}
#                 self.playerAgents[inst] = {AGENT1:PLAYER1, AGENT2:PLAYER2}
#             else:
#                 self.agents[inst] = {PLAYER1:self.agent2, PLAYER2:self.agent1}
#                 self.playerAgents[inst] = {AGENT1:PLAYER2, AGENT2:PLAYER1}
                
        
#         # configuration of epsilon greedy algorithm
#         self.epsilon = 1 # start value for esilon greedy algorithm
#         self.epsilonDecay = EPSILON_DECAY
#         self.epsilonMin = MIN_EPSILON
#         # maximum steps before breaking a game
#         self.maxSteps = MAX_STEPS
        
#         # rewards
#         self.rewardList=[REWARD_MOVE, REWARD_OWNMILL, REWARD_ENEMYMILL, REWARD_WIN, REWARD_LOSS]
        
#         # game parameters
#         self.episode_reward = [None]*self.numInstances
#         self.step = [None]*self.numInstances
#         self.field = [None]*self.numInstances
#         self.player = [None]*self.numInstances
#         self.inputVector = [None]*self.numInstances
#         self.phase = [None]*self.numInstances
#         self.move = [None]*self.numInstances
        
#         # store relevant old game parameters
#         self.lastInputVector = [None]*self.numInstances
#         self.lastSelectedAction = [None]*self.numInstances
#         self.lastPhase = [None]*self.numInstances
#         self.lastActionTypes = [None]*self.numInstances
#         for inst in range(self.numInstances):
#             self.lastInputVector[inst] = {PLAYER1:None, PLAYER2:None} # save last inputVectors of both players
#             self.lastSelectedAction[inst] = {PLAYER1:None, PLAYER2:None} # save last selected action of both players
#             self.lastPhase[inst] = {PLAYER1:None, PLAYER2:None} # save last phase of both players
#             self.lastActionTypes[inst] = {PLAYER1:enAction.SetPlayer1, PLAYER2:enAction.SetPlayer2}
        
#         # training stats 
#         self.ep_rewards_1 = []
#         self.ep_rewards_2 = []
#         self.ep_steps = []
        
        
#     def configureTraining(self, epsilonStart=1, epsilonDecay=0.999, epsilonMin=0.001, maxSteps=400,
#                           discount=0.99, minReplayMemSize=1024, minibatchSize=1024, 
#                           trainEveryXSteps=16, updateTargetEvery=5):
#         self.epsilon = epsilonStart
#         self.epsilonDecay = epsilonDecay
#         self.epsilonMin = epsilonMin
#         self.maxSteps = maxSteps
#         self.agent1.configureAgent(discount, minReplayMemSize, minibatchSize, trainEveryXSteps, updateTargetEvery)
#         self.agent2.configureAgent(discount, minReplayMemSize, minibatchSize, trainEveryXSteps, updateTargetEvery)
        
#     '''
#     rewardList=[REWARD_MOVE, REWARD_OWNMILL, REWARD_ENEMYMILL, REWARD_WIN, REWARD_LOSS]
#     '''
#     def configureRewards(self, rewardList):
#         self.rewardList = rewardList
        
#     def initializeGame(self):
#         self.gameStarted = [False]*self.numInstances
#         # reset training stats 
#         self.ep_rewards_1 = []
#         self.ep_rewards_2 = []
#         self.ep_steps = []
        
#     def startGame(self, inst):
#         self.episode_reward[inst] = {PLAYER1:0, PLAYER2:0}
#         self.step[inst] = 1

#         # Reset environment and get initial state
#         self.GameMillLogic[inst].restartGame()
#         self.field[inst], actionType, _, inStockTokens, remainingTokens = self.GameMillLogic[inst].getFullState() # initial state, empty field
#         self.player[inst] = getPlayerFromActionType(actionType) # player 1 starts
#         self.inputVector[inst], self.phase[inst], self.move[inst] = buildInputVector(self.player[inst], self.field[inst], actionType, inStockTokens, remainingTokens) # phase is "set", move ist "true"
        
#         validIdx = getPossibleMovesIndex(self.phase[inst], self.player[inst], self.field[inst], self.move[inst]) # start in phase set, all set moves possible

#         # This part stays mostly the same, the change is to query a model for Q values
#         if np.random.random() >= self.epsilon:
#             # Get action from Q table
#             qVals = self.agents[inst][self.player[inst]].get_qs(np.array([self.inputVector[inst]]), self.phase[inst])[0]
#             selectedPossibleAction = np.argmax(qVals[validIdx])
#         else:
#             # Get random action
#             selectedPossibleAction = np.random.randint(0, len(validIdx))
#         selectedAction = validIdx[selectedPossibleAction] # select a set action
        
#         newField = getMoveFromIndex(self.phase[inst], self.player[inst], self.field[inst], selectedAction) # start in phase set, new field with 1 token from player 1 
            
#         self.lastInputVector[inst][self.player[inst]] = self.inputVector[inst] # save values of player 1's first move
#         self.lastSelectedAction[inst][self.player[inst]] = selectedAction
#         self.lastPhase[inst][self.player[inst]] = self.phase[inst]
            
#         self.GameMillLogic[inst].setMove(newField) # make player 1's first move
#         self.field[inst], actionType, _, inStockTokens, remainingTokens = self.GameMillLogic[inst].getFullState() # field with 1 token from player 1 
#         self.player[inst] = getPlayerFromActionType(actionType) # player 2 makes the second move
#         self.inputVector[inst], self.phase[inst], self.move[inst] = buildInputVector(self.player[inst], self.field[inst], actionType, inStockTokens, remainingTokens) # phase is "set", move ist "true"
        
#     def stepGame(self, bSwitchPlayers=True):
#         finishedEpisodes = 0
#         validIdx = [None]*self.numInstances
        
#         for inst in range(self.numInstances):
#             if False == self.gameStarted[inst]:
#                 self.startGame(inst)
#                 self.gameStarted[inst] = True
                
#             validIdx[inst] = getPossibleMovesIndex(self.phase[inst], self.player[inst], self.field[inst], self.move[inst])
           
           
#         qsToPredict = np.random.random(self.numInstances) >= self.epsilon 
#         agent1OnMove = np.array(self.player) == np.array([self.playerAgents[inst][AGENT1] for inst in range(self.numInstances)]) # instances where agent1 has to move
#         agent1Predicts = qsToPredict & agent1OnMove
#         agent2Predicts = qsToPredict & ~agent1OnMove
        
#         setInstances = np.array(self.phase) == PHASE_SET
#         shiftInstances = np.array(self.phase) == PHASE_SHIFT
#         jumpInstances = np.array(self.phase) == PHASE_JUMP
        
#         agent1PredictsSet = agent1Predicts & setInstances
#         agent1PredictsShift = agent1Predicts & shiftInstances
#         agent1PredictsJump = agent1Predicts & jumpInstances
#         agent2PredictsSet = agent2Predicts & setInstances
#         agent2PredictsShift = agent2Predicts & shiftInstances
#         agent2PredictsJump = agent2Predicts & jumpInstances
        
#         qVals = [None]*self.numInstances
#         states = np.array(self.inputVector, dtype=object)
        
#         if np.any(agent1PredictsSet):
#             firstInst = np.argmax(agent1PredictsSet)
#             states_np = np.array(list(states[agent1PredictsSet]),dtype=int)
#             qs = self.agents[firstInst][self.player[firstInst]].get_qs(states_np, PHASE_SET) # PHASE_SET replaces self.phase[firstInst]
#             idx=0
#             for inst, bPredicted in enumerate(agent1PredictsSet):
#                 if True == bPredicted:
#                     qVals[inst] = qs[idx]
#                     idx += 1
                    
#         if np.any(agent1PredictsShift):
#             firstInst = np.argmax(agent1PredictsShift)
#             states_np = np.array(list(states[agent1PredictsShift]),dtype=int)
#             qs = self.agents[firstInst][self.player[firstInst]].get_qs(states_np, PHASE_SHIFT) 
#             idx=0
#             for inst, bPredicted in enumerate(agent1PredictsShift):
#                 if True == bPredicted:
#                     qVals[inst] = qs[idx]
#                     idx += 1
            
#         if np.any(agent1PredictsJump):
#             firstInst = np.argmax(agent1PredictsJump)
#             states_np = np.array(list(states[agent1PredictsJump]),dtype=int)
#             qs = self.agents[firstInst][self.player[firstInst]].get_qs(states_np, PHASE_JUMP)
#             idx=0
#             for inst, bPredicted in enumerate(agent1PredictsJump):
#                 if True == bPredicted:
#                     qVals[inst] = qs[idx]
#                     idx += 1
            
#         if np.any(agent2PredictsSet):
#             firstInst = np.argmax(agent2PredictsSet)
#             states_np = np.array(list(states[agent2PredictsSet]),dtype=int)
#             qs = self.agents[firstInst][self.player[firstInst]].get_qs(states_np, PHASE_SET)
#             idx=0
#             for inst, bPredicted in enumerate(agent2PredictsSet):
#                 if True == bPredicted:
#                     qVals[inst] = qs[idx]
#                     idx += 1
#             # qVals[agent2PredictsSet] = qs
            
#         if np.any(agent2PredictsShift):
#             firstInst = np.argmax(agent2PredictsShift)
#             states_np = np.array(list(states[agent2PredictsShift]),dtype=int)
#             qs = self.agents[firstInst][self.player[firstInst]].get_qs(states_np, PHASE_SHIFT)
#             idx=0
#             for inst, bPredicted in enumerate(agent2PredictsShift):
#                 if True == bPredicted:
#                     qVals[inst] = qs[idx]
#                     idx += 1
            
#         if np.any(agent2PredictsJump):
#             firstInst = np.argmax(agent2PredictsJump)
#             states_np = np.array(list(states[agent2PredictsJump]),dtype=int)
#             qs = self.agents[firstInst][self.player[firstInst]].get_qs(states_np, PHASE_JUMP)
#             idx=0
#             for inst, bPredicted in enumerate(agent2PredictsJump):
#                 if True == bPredicted:
#                     qVals[inst] = qs[idx]
#                     idx += 1
   
                
#         for inst in range(self.numInstances):
#             if True == qsToPredict[inst]:
#                 # Get action from Q table
#                 selectedPossibleAction = np.argmax(qVals[inst][validIdx[inst]])
#             else:
#                 # Get random action
#                 selectedPossibleAction = np.random.randint(0, len(validIdx[inst]))  
            
#             selectedAction = validIdx[inst][selectedPossibleAction] # select a set action
            
#             newField = getMoveFromIndex(self.phase[inst], self.player[inst], self.field[inst], selectedAction)
            
#             self.lastInputVector[inst][self.player[inst]] = self.inputVector[inst]
#             self.lastSelectedAction[inst][self.player[inst]] = selectedAction
#             self.lastPhase[inst][self.player[inst]] = self.phase[inst]
                
#             bActionValid = self.GameMillLogic[inst].setMove(newField)
#             # if not bActionValid:
#             #     print("ERROR: Invalid action!")
#             self.field[inst], actionType, _, inStockTokens, remainingTokens = self.GameMillLogic[inst].getFullState()
#             self.player[inst] = getPlayerFromActionType(actionType) # player 2 makes the second move
#             self.inputVector[inst], self.phase[inst], self.move[inst] = buildInputVector(self.player[inst], self.field[inst], actionType, inStockTokens, remainingTokens)
#             reward, self.lastActionTypes[inst], done = getReward(self.player[inst], actionType, self.lastActionTypes[inst], self.rewardList)
            
#             # Count reward
#             self.episode_reward[inst][self.player[inst]] += reward
    
#             # Every step we update replay memory and train main network
#             self.agents[inst][self.player[inst]].update_replay_memory((self.lastInputVector[inst][self.player[inst]], self.lastSelectedAction[inst][self.player[inst]], reward, self.player[inst], self.inputVector[inst], self.field[inst], self.move[inst], self.phase[inst], done), self.lastPhase[inst][self.player[inst]])
#             self.agents[inst][self.player[inst]].train(done, self.lastPhase[inst][self.player[inst]], self.step[inst])
    
#             if True == done: # player won the game
#                 reward = self.rewardList[-1] # reward=REWARD_LOSS for opponent
#                 self.player[inst] = PLAYER1 + PLAYER2 - self.player[inst] # get the player who lost
                
#                 # Count reward
#                 self.episode_reward[inst][self.player[inst]] += reward
    
#                 # Every step we update replay memory and train main network
#                 # inputVector, field, move and phase are irrelevent because done is True
#                 self.agents[inst][self.player[inst]].update_replay_memory((self.lastInputVector[inst][self.player[inst]], self.lastSelectedAction[inst][self.player[inst]], reward, self.player[inst], self.inputVector[inst], self.field[inst], self.move[inst], self.phase[inst], done), self.lastPhase[inst][self.player[inst]])
#                 self.agents[inst][self.player[inst]].train(done, self.lastPhase[inst][self.player[inst]], self.step[inst])
    
#             self.step[inst] += 1
            
#             if done or self.step[inst] >= self.maxSteps:
#                 # game finished
#                 self.gameStarted[inst] = False
#                 finishedEpisodes += 1
#                 self.finishGame(inst)
#                 if bSwitchPlayers:
#                     self.switchPlayers(inst)
        
#         return finishedEpisodes
            
            
        
#     def finishGame(self, inst):
#         self.ep_rewards_1 += [self.episode_reward[inst][self.playerAgents[inst][AGENT1]]]
#         self.ep_rewards_2 += [self.episode_reward[inst][self.playerAgents[inst][AGENT2]]]
#         self.ep_steps += [self.step[inst]]

#         # Decay epsilon
#         if self.epsilon > self.epsilonMin:
#             self.epsilon *= self.epsilonDecay
#             self.epsilon = max(self.epsilonMin, self.epsilon)
        
      
        
#     # do not call inside a game
#     # agent 1 will play as the given player, agent 2 will play as the opponent
#     def setPlayerAgent1(self, player, inst):
#         self.playerAgents[inst][AGENT1] = player
#         self.playerAgents[inst][AGENT2] = PLAYER1+PLAYER2-player
#         self.agents[inst][player] = self.agent1
#         self.agents[inst][PLAYER1+PLAYER2-player] = self.agent2

#     # do not call inside a game
#     # agent 1 and agent 2 will switch the player they play -> the agent who starts the game changes
#     def switchPlayers(self, inst):
#         # set agent 1 as the player who agent2 actually is
#         self.setPlayerAgent1(self.playerAgents[inst][AGENT2], inst)

        
#     def getGameStats(self, bFull=False):
#         if bFull: # return stats of all played games
#             return self.ep_rewards_1, self.ep_rewards_2, self.ep_steps
#         else: # return stats of last played game
#             return self.ep_rewards_1[-1], self.ep_rewards_2[-1], self.ep_steps[-1]
        
#     def selectModelsToTrain(self, agent, bSet, bShift, bJump):
#         if AGENT1 == agent:
#             self.agent1.selectModelsToTrain(bSet, bShift, bJump)
#         elif AGENT2 == agent:
#             self.agent2.selectModelsToTrain(bSet, bShift, bJump)
#         else:
#             print('Invalid agent !!!')
              
#     def setModels(self, agent, models):
#         modelSet, modelShift, modelJump = models
#         if AGENT1 == agent:
#             # modelSet = tf.keras.models.load_model('models_test/agent1_model_set.h5')
#             # modelShift = tf.keras.models.load_model('models_test/agent1_model_shift.h5')
#             # modelJump = tf.keras.models.load_model('models_test/agent1_model_jump.h5')
#             bOkay = self.agent1.setModels(modelSet, modelShift, modelJump)
#         elif AGENT2 == agent:
#             # modelSet = tf.keras.models.load_model('models_test/agent2_model_set.h5')
#             # modelShift = tf.keras.models.load_model('models_test/agent2_model_shift.h5')
#             # modelJump = tf.keras.models.load_model('models_test/agent2_model_jump.h5')
#             bOkay = self.agent2.setModels(modelSet, modelShift, modelJump)
#         else:
#             print('Invalid agent !!!')
#         return bOkay
    
#     def getModels(self, agent):
#         if AGENT1 == agent:
#             modelSet, modelShift, modelJump = self.agent1.getModels()
#             # modelSet.save('models_test/agent1_model_set.h5')
#             # modelShift.save('models_test/agent1_model_shift.h5')
#             # modelJump.save('models_test/agent1_model_jump.h5')
#         elif AGENT2 == agent:
#             modelSet, modelShift, modelJump = self.agent2.getModels()
#             # modelSet.save('models_test/agent2_model_set.h5')
#             # modelShift.save('models_test/agent2_model_shift.h5')
#             # modelJump.save('models_test/agent2_model_jump.h5')
#         else:
#             print('Invalid agent !!!')
#         return [modelSet, modelShift, modelJump]




