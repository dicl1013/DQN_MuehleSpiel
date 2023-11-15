# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:09:20 2023

@author: andre
"""

import numpy as np
from enum import Enum

### defines ###

# possible states of one place in the mill field
PLAYER1=1   # token of player 1
EMPTY=0     # no token
PLAYER2=2  # token of player 2

# possible states of the game, private
class enState(Enum):
    SetPlayer1 = 0
    SetPlayer2 = 1
    SetMillPlayer1 = 2
    SetMillPlayer2 = 3
    ShiftPlayer1 = 4
    ShiftPlayer2 = 5
    ShiftMillPlayer1 = 6
    ShiftMillPlayer2 = 7
    GameFinished = 8
   
# possible actions, public
class enAction(Enum):
    SetPlayer1 = 0
    SetPlayer2 = 1
    ShiftPlayer1 = 2
    ShiftPlayer2 = 3
    RemoveTokenPlayer1 = 4
    RemoveTokenPlayer2 = 5
    Player1Wins = 6
    Player2Wins = 7


### main class ###        
class MillLogic:
    millField = np.zeros([3,8], dtype=int)
    state = enState.SetPlayer1
    action = enAction.SetPlayer1
    inStockTokens = {PLAYER1: 9, PLAYER2:9}
    remainingTokens = {PLAYER1: 9, PLAYER2:9}
    possibleMoves = []
    
    def __init__(self):
        self.calcMovesSet(PLAYER1)
        pass
    
    ### public functions ###
    
    ####################### communication with the class ######################
    def restartGame(self):
        self.millField = np.zeros([3,8], dtype=int)
        self.state = enState.SetPlayer1
        self.action = enAction.SetPlayer1
        self.inStockTokens = {PLAYER1: 9, PLAYER2:9}
        self.remainingTokens = {PLAYER1: 9, PLAYER2:9}
        self.calcMovesSet(PLAYER1)

    # returns:
    # - millField: state of the millField
    # - action: next expected action
    def getState(self):
        field = np.copy(self.millField)
        act = self.action
        return field, act
    
    def getFullState(self):
        return np.copy(self.millField), np.copy(self.action), np.copy(self.possibleMoves), np.copy(self.inStockTokens), np.copy(self.remainingTokens)
    
    # returns a list of possible new states of the mill field
    def getPossibleMoves(self):
        return np.copy(self.possibleMoves)
    
    def getMovesFromSelectedToken(self, indexRing, indexPos):
        possibleMovesSpecific = []
        bSelectionValid = False
        player = EMPTY
        # if switch player 1 and player 1 selected or switch player 2 and player 2 selected
        if self.action == enAction.ShiftPlayer1 and PLAYER1 == self.millField[indexRing,indexPos]:
            player = PLAYER1
            bSelectionValid = True
        elif self.action == enAction.ShiftPlayer2 and PLAYER2 == self.millField[indexRing,indexPos]:
            player = PLAYER2
            bSelectionValid = True
            
        if bSelectionValid:          
            if self.remainingTokens[player] > 3: # move, no jumping phase
                newIndices = []
                newIndices += [[indexRing, self.mappedIndex(indexPos+1)]]
                newIndices += [[indexRing, self.mappedIndex(indexPos-1)]]
                if indexRing >=1 and indexPos%2 == 1:
                    newIndices += [[indexRing-1, indexPos]]
                if indexRing <=1 and indexPos%2 == 1:
                    newIndices += [[indexRing+1, indexPos]]
                
                for (newIndexRing, newIndexPos) in newIndices:
                    if EMPTY == self.millField[newIndexRing, newIndexPos]:
                        newMillField = np.copy(self.millField)
                        newMillField[indexRing, indexPos] = EMPTY # remove token from old position
                        newMillField[newIndexRing, newIndexPos] = player # add token on new position
                        possibleMovesSpecific += [newMillField]
                                
            else:   # 3 remaining tokens -> jumping phase
                for newIndexRing in range(3):  # iterate through rings
                    for newIndexPos, place in enumerate(self.millField[indexRing,:]):  # iterate through positions in the ring
                        if EMPTY == place:   # jump to every empty place is possible
                            newMillField = np.copy(self.millField)
                            newMillField[indexRing, indexPos] = EMPTY # remove token from old position
                            newMillField[newIndexRing, newIndexPos] = player # add token on new position
                            possibleMovesSpecific += [newMillField]
                            
        return possibleMovesSpecific
                    
            
    
    def getInStockTokens(self):
        return self.inStockTokens       # GEÄNDERT!

    def getRemainingTokens(self):
        return np.copy(self.remainingTokens)
    
    # set the new state of the mill field that matches the requested action
    # returns:
    # - bActionValid: bool that specifies if last action was valid
    def setMove(self, newMillField):
        bActionValid = False
        cNewMillField=np.copy(newMillField)
        
        if enState.SetPlayer1 == self.state:
            bActionValid = self.stateSetPlayer1(cNewMillField)
        elif enState.SetPlayer2 == self.state:
            bActionValid = self.stateSetPlayer2(cNewMillField)
        elif enState.SetMillPlayer1 == self.state:
            bActionValid = self.stateSetMillPlayer1(cNewMillField)
        elif enState.SetMillPlayer2 == self.state:
            bActionValid = self.stateSetMillPlayer2(cNewMillField)
        elif enState.ShiftPlayer1 == self.state:
            bActionValid = self.stateShiftPlayer1(cNewMillField)
        elif enState.ShiftPlayer2 == self.state:
            bActionValid = self.stateShiftPlayer2(cNewMillField)
        elif enState.ShiftMillPlayer1 == self.state:
            bActionValid = self.stateShiftMillPlayer1(cNewMillField)
        elif enState.ShiftMillPlayer2 == self.state:
            bActionValid = self.stateShiftMillPlayer2(cNewMillField)
        elif enState.GameFinished == self.state:
            bActionValid = self.stateGameFinished(cNewMillField)
        else:
            print("ERROR: invalid state")
            self.restartGame()
        
        return bActionValid

    def initializeSpecificState(self, newMillField, bPlayer1Starts=True, dicInStockTokens={PLAYER1: 0, PLAYER2:0} ):
        self.millField = np.copy(newMillField)
        self.inStockTokens = dicInStockTokens.copy()
        self.remainingTokens = dicInStockTokens.copy()
        for indexRing in range(3):  # iterate through rings
                for indexPos, place in enumerate(self.millField[indexRing,:]):  # iterate through positions in the ring
                    if EMPTY != place:
                        self.remainingTokens[place] += 1
        
        if True == bPlayer1Starts:
            self.calcMovesShift(PLAYER1)
            if 0 == len(self.possibleMoves):    # player 1 can't move -> player 2 wins
                self.state = enState.GameFinished
                self.action = enAction.Player2Wins
            else:
                self.state = enState.ShiftPlayer1
                self.action = enAction.ShiftPlayer1
        else:
            self.calcMovesShift(PLAYER2)
            if 0 == len(self.possibleMoves):    # player 2 can't move -> player 1 wins
                self.state = enState.GameFinished
                self.action = enAction.Player1Wins
            else:
                self.state = enState.ShiftPlayer2
                self.action = enAction.ShiftPlayer2

        
    ### private functions ### 
    
    ############################### game states ###############################
    def stateSetPlayer1(self, newMillField):
        bActionValid = False
        
        if self.isNewStatePossible(newMillField):
            self.inStockTokens[PLAYER1] -= 1
            bActionValid = True
            bNewMill = self.isNewMillFormed(PLAYER1, newMillField)
            self.millField = newMillField
            
            if bNewMill:
                self.calcMovesRemoveToken(PLAYER1)
                self.state = enState.SetMillPlayer1
                self.action = enAction.RemoveTokenPlayer1
            else:
                self.calcMovesSet(PLAYER2)
                self.state = enState.SetPlayer2
                self.action = enAction.SetPlayer2
        
        return bActionValid
        
        
    def stateSetPlayer2(self, newMillField):
        bActionValid = False

        if self.isNewStatePossible(newMillField):
            self.inStockTokens[PLAYER2] -= 1
            bActionValid = True
            bNewMill = self.isNewMillFormed(PLAYER2, newMillField)
            self.millField = newMillField
            
            if bNewMill:
                self.calcMovesRemoveToken(PLAYER2)
                self.state = enState.SetMillPlayer2
                self.action = enAction.RemoveTokenPlayer2
            else:
                if 0 == self.inStockTokens[PLAYER1]:
                    self.calcMovesShift(PLAYER1)
                    if 0 == len(self.possibleMoves):    # player 1 can't move -> player 2 wins
                        self.state = enState.GameFinished
                        self.action = enAction.Player2Wins
                    else:
                        self.state = enState.ShiftPlayer1
                        self.action = enAction.ShiftPlayer1
                else:
                    self.calcMovesSet(PLAYER1)
                    self.state = enState.SetPlayer1
                    self.action = enAction.SetPlayer1
        
        return bActionValid
        
    def stateSetMillPlayer1(self, newMillField):
        bActionValid = False
        
        if self.isNewStatePossible(newMillField):
            self.remainingTokens[PLAYER2] -= 1
            bActionValid = True
            self.millField = newMillField
            
            self.calcMovesSet(PLAYER2)
            self.state = enState.SetPlayer2
            self.action = enAction.SetPlayer2
        
        return bActionValid
        
    def stateSetMillPlayer2(self, newMillField):
        bActionValid = False
        
        if self.isNewStatePossible(newMillField):
            self.remainingTokens[PLAYER1] -= 1
            bActionValid = True
            self.millField = newMillField
            
            if self.remainingTokens[PLAYER1] <= 2:  # player 2 wins
                self.possibleMoves = []
                self.state = enState.GameFinished
                self.action = enAction.Player2Wins
            else:
                if 0 == self.inStockTokens[PLAYER1]:
                    self.calcMovesShift(PLAYER1)
                    if 0 == len(self.possibleMoves):    # player 1 can't move -> player 2 wins
                        self.state = enState.GameFinished
                        self.action = enAction.Player2Wins
                    else:
                        self.state = enState.ShiftPlayer1
                        self.action = enAction.ShiftPlayer1
                else:
                    self.calcMovesSet(PLAYER1)
                    self.state = enState.SetPlayer1
                    self.action = enAction.SetPlayer1
        
        return bActionValid
        
    def stateShiftPlayer1(self, newMillField):
        bActionValid = False
        
        if self.isNewStatePossible(newMillField):
            bActionValid = True
            bNewMill = self.isNewMillFormed(PLAYER1, newMillField)
            self.millField = newMillField
            
            if bNewMill:
                self.calcMovesRemoveToken(PLAYER1)
                self.state = enState.ShiftMillPlayer1
                self.action = enAction.RemoveTokenPlayer1
            else:
                self.calcMovesShift(PLAYER2)
                if 0 == len(self.possibleMoves):    # player 2 can't move -> player 1 wins
                    self.state = enState.GameFinished
                    self.action = enAction.Player1Wins
                else:
                    self.state = enState.ShiftPlayer2
                    self.action = enAction.ShiftPlayer2
        
        return bActionValid
    
    def stateShiftPlayer2(self, newMillField):
        bActionValid = False
        
        if self.isNewStatePossible(newMillField):
            bActionValid = True
            bNewMill = self.isNewMillFormed(PLAYER2, newMillField)
            self.millField = newMillField
            
            if bNewMill:
                self.calcMovesRemoveToken(PLAYER2)
                self.state = enState.ShiftMillPlayer2
                self.action = enAction.RemoveTokenPlayer2
            else:
                self.calcMovesShift(PLAYER1)
                if 0 == len(self.possibleMoves):    # player 1 can't move -> player 2 wins
                    self.state = enState.GameFinished
                    self.action = enAction.Player2Wins
                else:
                    self.state = enState.ShiftPlayer1
                    self.action = enAction.ShiftPlayer1
        
        return bActionValid
    
    def stateShiftMillPlayer1(self, newMillField):
        bActionValid = False
        
        if self.isNewStatePossible(newMillField):
            self.remainingTokens[PLAYER2] -= 1
            bActionValid = True
            self.millField = newMillField
            
            if self.remainingTokens[PLAYER2] <= 2:  # player 1 wins
                self.possibleMoves = []
                self.state = enState.GameFinished
                self.action = enAction.Player1Wins
            else:
                self.calcMovesShift(PLAYER2)
                if 0 == len(self.possibleMoves):    # player 2 can't move -> player 1 wins
                    self.state = enState.GameFinished
                    self.action = enAction.Player1Wins
                else:
                    self.state = enState.ShiftPlayer2
                    self.action = enAction.ZugPlayer2
        
        return bActionValid
    
    def stateShiftMillPlayer2(self, newMillField):
        bActionValid = False
        
        if self.isNewStatePossible(newMillField):
            self.remainingTokens[PLAYER1] -= 1
            bActionValid = True
            self.millField = newMillField
            
            if self.remainingTokens[PLAYER1] <= 2:  # player 2 wins
                self.possibleMoves = []
                self.state = enState.GameFinished
                self.action = enAction.Player2Wins
            else:
                self.calcMovesShift(PLAYER1)
                if 0 == len(self.possibleMoves):    # player 1 can't move -> player 2 wins
                    self.state = enState.GameFinished
                    self.action = enAction.Player2Wins
                else:
                    self.state = enState.ShiftPlayer1
                    self.action = enAction.ShiftPlayer1
        
        return bActionValid
        
    def stateGameFinished(self, newMillField):
        bActionValid = False # no valid actions if game is finished
        return bActionValid
    
          
    ###################### calculation of possible moves ######################
    def calcMovesSet(self, player):
        self.possibleMoves = [] # reset possibleMoves
        for indexRing in range(3):  # iterate through rings
            for indexPos, place in enumerate(self.millField[indexRing,:]):  # iterate through positions in the ring
                if EMPTY == place:
                    newMillField = np.copy(self.millField)
                    newMillField[indexRing,indexPos] = player # move: set a token of the current player to empty place
                    self.possibleMoves += [newMillField]
        
    def calcMovesShift(self, player):
        self.possibleMoves = [] # reset possibleMoves
        
        if self.remainingTokens[player] > 3: # move, no jumping phase
            for indexRing in range(3):  # iterate through rings
                for indexPos, place in enumerate(self.millField[indexRing,:]):  # iterate through positions in the ring
                    if player == place:
                        newIndices = []
                        newIndices += [[indexRing, self.mappedIndex(indexPos+1)]]
                        newIndices += [[indexRing, self.mappedIndex(indexPos-1)]]
                        if indexRing >=1 and indexPos%2 == 1:
                            newIndices += [[indexRing-1, indexPos]]
                        if indexRing <=1 and indexPos%2 == 1:
                            newIndices += [[indexRing+1, indexPos]]
                        
                        for (newIndexRing, newIndexPos) in newIndices:
                            if EMPTY == self.millField[newIndexRing, newIndexPos]:
                                newMillField = np.copy(self.millField)
                                newMillField[indexRing, indexPos] = EMPTY # remove token from old position
                                newMillField[newIndexRing, newIndexPos] = player # add token on new position
                                self.possibleMoves += [newMillField]
                                
        else:   # 3 remaining tokens -> jumping phase
            newIndices = []
            for indexRing in range(3):  # iterate through rings
                for indexPos, place in enumerate(self.millField[indexRing,:]):  # iterate through positions in the ring
                    if EMPTY == place:   # jump to every empty place is possible
                        newIndices += [[indexRing, indexPos]]
                                       
            for indexRing in range(3):  # iterate through rings
                for indexPos, place in enumerate(self.millField[indexRing,:]):  # iterate through positions in the ring
                    if player == place:   # jump from this place to newIndices
                        for (newIndexRing, newIndexPos) in newIndices:
                            newMillField = np.copy(self.millField)
                            newMillField[indexRing, indexPos] = EMPTY # remove token from old position
                            newMillField[newIndexRing, newIndexPos] = player # add token on new position
                            self.possibleMoves += [newMillField]
        
    def calcMovesRemoveToken(self, playerOfFormedMill):
        playerToRemoveFrom = -playerOfFormedMill
        self.possibleMoves = [] # reset possibleMoves
        notPartOfMillTokens = np.zeros([3,8], dtype=bool)
        
        for indexRing in range(3):  # iterate through rings
            for indexPos, place in enumerate(self.millField[indexRing,:]):  # iterate through positions in the ring
                if playerToRemoveFrom == place:
                    if False == self.isPartOfMill(playerToRemoveFrom, indexRing, indexPos):
                        notPartOfMillTokens[indexRing,indexPos] = True
        
        bOnlyMills = False
        if np.all(notPartOfMillTokens == False):    # every token of the player to remove from is part of a mill
            bOnlyMills = True
                    
        for indexRing in range(3):  # iterate through rings
            for indexPos, place in enumerate(self.millField[indexRing,:]):  # iterate through positions in the ring
                if playerToRemoveFrom == place:
                        if bOnlyMills or True == notPartOfMillTokens[indexRing,indexPos]:   # the token is not part of a mill or there are only mills
                            newMillField = np.copy(self.millField)
                            newMillField[indexRing,indexPos] = EMPTY # remove the token
                            self.possibleMoves += [newMillField]
 
    
    ############################# helper functions ############################
    def isNewStatePossible(self, newMillField):
        bNewStatePossible = False
        # is newMillField in possibleMoves?
        for move in self.possibleMoves:
            if np.array_equal(newMillField, move):
                bNewStatePossible = True
                break
        return bNewStatePossible
       
        
    def isNewMillFormed(self, player, newMillField):
        bNewMill = False
        changesMillField = newMillField - self.millField
        # calc index of new set token or new position of moved token
        (indexRing, indexPos) = np.unravel_index( np.argmax(player == changesMillField), changesMillField.shape )
        if indexPos%2 == 1: # middle of an edge
            if (newMillField[indexRing, self.mappedIndex(indexPos-1)] == player # mill in one ring
                and newMillField[indexRing, self.mappedIndex(indexPos+1)] == player
            ):
                bNewMill = True
            elif (newMillField[0, indexPos] == player # mill over all rings
                  and newMillField[1, indexPos] == player
                  and newMillField[2, indexPos] == player
            ):
                bNewMill = True
        else:   # corner
            if (newMillField[indexRing, self.mappedIndex(indexPos-2)] == player # mill in one ring
                and newMillField[indexRing, self.mappedIndex(indexPos-1)] == player
            ):
                bNewMill = True
            elif (newMillField[indexRing, self.mappedIndex(indexPos+1)] == player # mill in one ring
                  and newMillField[indexRing, self.mappedIndex(indexPos+2)] == player
            ):
                bNewMill = True
        return bNewMill
    
    def isPartOfMill(self, player, indexRing, indexPos):
        bPartOfMill = False
        if indexPos%2 == 1: # middle of an edge
            if (self.millField[indexRing, self.mappedIndex(indexPos-1)] == player # mill in one ring
                and self.millField[indexRing, self.mappedIndex(indexPos+1)] == player
            ):
                bPartOfMill = True
            elif (self.millField[0, indexPos] == player # mill over all rings
                  and self.millField[1, indexPos] == player
                  and self.millField[2, indexPos] == player
            ):
                bPartOfMill = True
        else:   # corner
            if (self.millField[indexRing, self.mappedIndex(indexPos-2)] == player # mill in one ring
                and self.millField[indexRing, self.mappedIndex(indexPos-1)] == player
            ):
                bPartOfMill = True
            elif (self.millField[indexRing, self.mappedIndex(indexPos+1)] == player # mill in one ring
                  and self.millField[indexRing, self.mappedIndex(indexPos+2)] == player
            ):
                bPartOfMill = True
        return bPartOfMill
    
    def mappedIndex(self, ind):
        if ind > 7:
            return ind - 8
        elif ind < 0:
            return ind + 8
        else:
            return ind
    
    
    
    
    
def getPrintState():
    field, action = muehle.getState()
    print(field)
    print(action)
    return field
    
def setMove(field):
    return muehle.setMove(field)
    
if __name__=='__main__':  
    muehle = MillLogic()
    print("game started")
        
                    
    
    
    
