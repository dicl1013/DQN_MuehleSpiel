# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:09:20 2023

@author: andre
"""

import numpy as np
from enum import Enum

### defines ###

# possible states of one place in the mill field
EMPTY=0     # no token
PLAYER1=1   # token of player 1
PLAYER2=-1  # token of player 2

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
    checkMoves = True
    
    def __init__(self, checkMoves=True):
        """
        Parameters
        ----------
        checkMoves : bool
            True: the mill logic checks every move if it is possible. The default is True.
            False: no check, only recommended for training when invalid moves are impossible.
        """
        self.checkMoves=checkMoves
        self.restartGame()
    
    ### public functions ###
    
    ####################### communication with the class ######################
    # functions to communicate with the mill logic
    # - restatGame or initialize a specific state
    # - get information about the curren state
    # - set the next move that matches the required action
    
    def restartGame(self):
        """
        sets the mill logic into start position
        """
        self.millField = np.zeros([3,8], dtype=int)
        self.state = enState.SetPlayer1
        self.action = enAction.SetPlayer1
        self.inStockTokens = {PLAYER1: 9, PLAYER2:9}
        self.remainingTokens = {PLAYER1: 9, PLAYER2:9}
        self.calcMovesSet(PLAYER1)

    def getState(self):
        """
        Returns
        -------
        millField: array with shape (3, 8)
            state of the millField
        action: from enum enAction
            next expected action
        """
        return np.copy(self.millField), self.action
    
    def getFullState(self):
        """
        Returns
        -------
        millField: array with shape (3, 8)
            state of the millField
        action: from enum enAction
            next expected action
        possibleMoves: a list of all possible new millFields after the next action
        inStockTokens: dictionary with number of tokens still to place for each player
        remainingTokens: dictionary with number of tokens still in game for each player
        """
        return np.copy(self.millField), self.action, np.copy(self.possibleMoves), self.inStockTokens.copy(), self.remainingTokens.copy()
    
    def getPossibleMoves(self):
        """
        Returns
        -------
        possibleMoves: a list of all possible new millFields after the next action
        """
        return np.copy(self.possibleMoves)
    
    def getMovesFromSelectedToken(self, indexRing, indexPos):
        """
        call this function is only valid if the player has to shift the token on the selected position
        
        Parameters
        ----------
        indexRing : int (0..2)
            selects the ring of the token: 0->outer, 1->middle, 2->inner
        indexPos : int (0..7)
            selects the position inside the ring 0->up-left then clockwise

        Returns
        -------
        possibleNewIndices : list of pairs [indexRing, indexPos]
            indices of all possible next positions of the selected token
        """
        possibleNewIndices = []
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
                newIndices += [[indexRing, (indexPos+1)%8]]
                newIndices += [[indexRing, (indexPos-1)%8]]
                if indexPos%2 == 1:
                    if indexRing >=1:
                        newIndices += [[indexRing-1, indexPos]]
                    if indexRing <=1:
                        newIndices += [[indexRing+1, indexPos]]
                
                for (newIndexRing, newIndexPos) in newIndices:
                    if EMPTY == self.millField[newIndexRing, newIndexPos]:
                        possibleNewIndices += [[newIndexRing, newIndexPos]]
                                
            else:   # 3 remaining tokens -> jumping phase
                for newIndexRing in range(3):  # iterate through rings
                    for newIndexPos, place in enumerate(self.millField[newIndexRing,:]):  # iterate through positions in the ring
                        if EMPTY == place:   # jump to every empty place is possible
                            possibleNewIndices += [[newIndexRing, newIndexPos]]
                            
        return possibleNewIndices
                    
            
    
    def getInStockTokens(self):
        """
        Returns
        -------
        inStockTokens: dictionary with number of tokens still to place for each player
        """
        return self.inStockTokens.copy()

    def getRemainingTokens(self):
        """
        Returns
        -------
        remainingTokens: dictionary with number of tokens still in game for each player
        """
        return self.remainingTokens.copy()
    
    def setMove(self, newMillField):
        """
        Parameters
        ----------
        newMillField : array with shape (3, 8)
            new state of the mill field that matches the requested action

        Returns
        -------
        bActionValid : bool
            specifies if last action was valid
        """
        bActionValid = False
        cNewMillField=np.copy(newMillField)
        
        if self.checkMoves:
            if self.isNewStatePossible(newMillField):
                bActionValid = True
        else:
            bActionValid = True
        
        if bActionValid:
            if enState.SetPlayer1 == self.state:
                self.stateSetPlayer1(cNewMillField)
            elif enState.SetPlayer2 == self.state:
                self.stateSetPlayer2(cNewMillField)
            elif enState.SetMillPlayer1 == self.state:
                self.stateSetMillPlayer1(cNewMillField)
            elif enState.SetMillPlayer2 == self.state:
                self.stateSetMillPlayer2(cNewMillField)
            elif enState.ShiftPlayer1 == self.state:
                self.stateShiftPlayer1(cNewMillField)
            elif enState.ShiftPlayer2 == self.state:
                self.stateShiftPlayer2(cNewMillField)
            elif enState.ShiftMillPlayer1 == self.state:
                self.stateShiftMillPlayer1(cNewMillField)
            elif enState.ShiftMillPlayer2 == self.state:
                self.stateShiftMillPlayer2(cNewMillField)
            elif enState.GameFinished == self.state:
                bActionValid = False # no valid actions if game is finished
            else:
                print("ERROR: invalid state")
                self.restartGame()
        
        return bActionValid

    def initializeSpecificState(self, newMillField, nextAction=enAction.ShiftPlayer1, dicInStockTokens={PLAYER1: 0, PLAYER2:0} ):
        """
        set the mill logic in any desired state
        
        Parameters
        ----------
        newMillField : array with shape (3, 8)
            new state of the mill field
        nextAction : from enum enAction, optional
            the action that is required after the initialization. The default is enAction.ShiftPlayer1.
        dicInStockTokens : dictionary, optional
            number of tokens still to place for each player, only required when starting in set phase. 
            The default is {PLAYER1: 0, PLAYER2:0} -> shift/jump phase.

        Returns
        -------
        bRet : bool
            not used, set to True. Could be used to check if given state is realisitc
        """
        bRet = True
        self.millField = np.copy(newMillField)
        self.inStockTokens = dicInStockTokens.copy()
        self.remainingTokens = dicInStockTokens.copy()
        for indexRing in range(3):  # iterate through rings
                for indexPos, place in enumerate(self.millField[indexRing,:]):  # iterate through positions in the ring
                    if EMPTY != place:
                        self.remainingTokens[place] += 1
        self.action = nextAction
        
        if enAction.SetPlayer1 == self.action:
            self.calcMovesSet(PLAYER1)
            self.state = enState.SetPlayer1
        elif enAction.SetPlayer2 == self.action:
            self.calcMovesSet(PLAYER2)
            self.state = enState.SetPlayer2
        elif enAction.ShiftPlayer1 == self.action:
            self.calcMovesShift(PLAYER1)
            self.state = enState.ShiftPlayer1
        elif enAction.ShiftPlayer2 == self.action:
            self.calcMovesShift(PLAYER2)
            self.state = enState.ShiftPlayer2
        elif enAction.RemoveTokenPlayer1 == self.action:
            self.calcMovesRemoveToken(PLAYER1)
            if self.inStockTokens[PLAYER2] != 0:    # player 2 has not set all tokens -> player 1 was in set phase before
                self.state = enState.SetMillPlayer1
            else:
                self.state = enState.ShiftMillPlayer1
        elif enAction.RemoveTokenPlayer2 == self.action:
            self.calcMovesRemoveToken(PLAYER2)
            if self.inStockTokens[PLAYER2] != 0:    # player 1 has not set all tokens -> player 2 was in set phase before
                self.state = enState.SetMillPlayer2
            else:   # even if player 2 was in set phase, no inStockTokens are left -> enState.ShiftMillPlayer2 is equal to enState.SetMillPlayer2
                self.state = enState.ShiftMillPlayer2
        elif enAction.Player1Wins == self.action:
            self.possibleMoves = []
            self.state = enState.GameFinished
        elif enAction.Player2Wins == self.action:
            self.possibleMoves = []
            self.state = enState.GameFinished
        else:
            print("ERROR: invalid action")
            self.restartGame()
                
        return bRet

        
    ### private functions ### 
    # only for usage inside the class
    
    ############################### game states ###############################
    # state means set/shift/mill formed/... and not the postions of the tokens in the mill field
    # main logic, these function determine what happens in each state, the next required action and the following state
    
    def stateSetPlayer1(self, newMillField):
        """
        player 1 sets a token
        """
        self.inStockTokens[PLAYER1] -= 1
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
        
        
    def stateSetPlayer2(self, newMillField):
        """
        player 2 sets a token
        """
        self.inStockTokens[PLAYER2] -= 1
        bNewMill = self.isNewMillFormed(PLAYER2, newMillField)
        self.millField = newMillField
        
        if bNewMill:
            self.calcMovesRemoveToken(PLAYER2)
            self.state = enState.SetMillPlayer2
            self.action = enAction.RemoveTokenPlayer2
        else:
            if 0 == self.inStockTokens[PLAYER1]:    # all tokens set -> go to shift phase
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
        
    def stateSetMillPlayer1(self, newMillField):
        """
        player 1 has formed a mill in set phase and removes a token from player 2
        """
        self.remainingTokens[PLAYER2] -= 1
        self.millField = newMillField
        
        self.calcMovesSet(PLAYER2)
        self.state = enState.SetPlayer2
        self.action = enAction.SetPlayer2
        
    def stateSetMillPlayer2(self, newMillField):
        """
        player 2 has formed a mill in set phase and removes a token from player 1
        """
        self.remainingTokens[PLAYER1] -= 1
        self.millField = newMillField
        
        if self.remainingTokens[PLAYER1] <= 2:  # player 2 wins
            self.possibleMoves = []
            self.state = enState.GameFinished
            self.action = enAction.Player2Wins
        else:
            if 0 == self.inStockTokens[PLAYER1]:    # all tokens set -> go to shift phase
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
        
    def stateShiftPlayer1(self, newMillField):
        """
        player 1 shifts a token (including jump)
        """
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
    
    def stateShiftPlayer2(self, newMillField):
        """
        player 2 shifts a token (including jump)
        """
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
    
    def stateShiftMillPlayer1(self, newMillField):
        """
        player 1 has formed a mill in shift(/jump) phase and removes a token from player 2
        """
        self.remainingTokens[PLAYER2] -= 1
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
                self.action = enAction.ShiftPlayer2
    
    def stateShiftMillPlayer2(self, newMillField):
        """
        player 2 has formed a mill in shift(/jump) phase and removes a token from player 1
        """
        self.remainingTokens[PLAYER1] -= 1
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
    
          
    ###################### calculation of possible moves ######################
    # depending on the phase, all possible moves will be calculated and saved in self.possibleMoves
    # these functions are called in the states after a move was made
    
    def calcMovesSet(self, player):
        """
        the selected player can simply set a token to every empty place
        the resulting possible new millFields will be calculated 
        """
        if self.checkMoves:
            self.possibleMoves = [] # reset possibleMoves
            for indexRing in range(3):  # iterate through rings
                for indexPos, place in enumerate(self.millField[indexRing,:]):  # iterate through positions in the ring
                    if EMPTY == place:
                        newMillField = np.copy(self.millField)
                        newMillField[indexRing,indexPos] = player # move: set a token of the current player to empty place
                        self.possibleMoves += [newMillField]
        
    def calcMovesShift(self, player):
        """
        in normal shift phase the selected player can move every token to neighboring empty places
        in jump phase the selected player can move any token to every empty place
        the resulting possible new millFields will be calculated 
        """
        if self.checkMoves:
            self.possibleMoves = [] # reset possibleMoves
            
            if self.remainingTokens[player] > 3: # move, no jumping phase
                for indexRing in range(3):  # iterate through rings
                    for indexPos, place in enumerate(self.millField[indexRing,:]):  # iterate through positions in the ring
                        if player == place:
                            newIndices = []
                            newIndices += [[indexRing, (indexPos+1)%8]]
                            newIndices += [[indexRing, (indexPos-1)%8]]
                            if indexPos%2 == 1:
                                if indexRing >=1:
                                    newIndices += [[indexRing-1, indexPos]]
                                if indexRing <=1:
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
        else:   # checkMoves inactive -> check only if a move is possible or the game is over
            self.testIfAMoveIsPossible(player)
        
    def calcMovesRemoveToken(self, playerOfFormedMill):
        """
        the selected player can remove every token of the opposite player that is not part of a mill
        if the opposite player only has tokens in mills, every token can be removed
        the resulting possible new millFields will be calculated 
        """
        if self.checkMoves:
            playerToRemoveFrom = PLAYER1 + PLAYER2 - playerOfFormedMill
            self.possibleMoves = [] # reset possibleMoves
            notPartOfMillTokens = np.zeros([3,8], dtype=bool) # True in this array marks a token of the opposite player that is not part of a mill
            
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
        """
        checks if newMillField is in possibleMoves
        returns True if new state is possible
        """
        bNewStatePossible = False
        # is newMillField in possibleMoves?
        for move in self.possibleMoves:
            if np.array_equal(newMillField, move):
                bNewStatePossible = True
                break
        return bNewStatePossible
       
        
    def isNewMillFormed(self, player, newMillField):
        """
        checks if player has formed a new mill with the move newMillField
        returns True if a new mill is formed
        """
        bNewMill = False
        changesMillField = newMillField - self.millField
        # calc index of new set token or new position of moved token
        (indexRing, indexPos) = np.unravel_index( np.argmax(player == changesMillField), changesMillField.shape )
        if indexPos%2 == 1: # middle of an edge
            if (newMillField[indexRing, (indexPos-1)%8] == player # mill in one ring
                and newMillField[indexRing, (indexPos+1)%8] == player
            ):
                bNewMill = True
            elif (newMillField[0, indexPos] == player # mill over all rings
                  and newMillField[1, indexPos] == player
                  and newMillField[2, indexPos] == player
            ):
                bNewMill = True
        else:   # corner
            if (newMillField[indexRing, (indexPos-2)%8] == player # mill in one ring
                and newMillField[indexRing, (indexPos-1)%8] == player
            ):
                bNewMill = True
            elif (newMillField[indexRing, (indexPos+1)%8] == player # mill in one ring
                  and newMillField[indexRing, (indexPos+2)%8] == player
            ):
                bNewMill = True
        return bNewMill
    
    def isPartOfMill(self, player, indexRing, indexPos):
        """
        checks if the token from player on position (indexRing, indexPos) is part of a mill
        returns True if the selected token is part of a mill
        """
        bPartOfMill = False
        if indexPos%2 == 1: # middle of an edge
            if (self.millField[indexRing, (indexPos-1)%8] == player # mill in one ring
                and self.millField[indexRing, (indexPos+1)%8] == player
            ):
                bPartOfMill = True
            elif (self.millField[0, indexPos] == player # mill over all rings
                  and self.millField[1, indexPos] == player
                  and self.millField[2, indexPos] == player
            ):
                bPartOfMill = True
        else:   # corner
            if (self.millField[indexRing, (indexPos-2)%8] == player # mill in one ring
                and self.millField[indexRing, (indexPos-1)%8] == player
            ):
                bPartOfMill = True
            elif (self.millField[indexRing, (indexPos+1)%8] == player # mill in one ring
                  and self.millField[indexRing, (indexPos+2)%8] == player
            ):
                bPartOfMill = True
        return bPartOfMill
    
    # only used if checkMoves=False to test if the game ends because a player cannot move
    def testIfAMoveIsPossible(self, player):
        """
        checks if the selected player can make a move in normal shift phase or is in jumping phase
        writes one element to self.possibleMoves to indicate that there is at least one possible move and the game is not over
        if no possible move is found, self.possibleMoves is empty
        """        
        if self.remainingTokens[player] > 3: # move, no jumping phase
            self.possibleMoves = [] # reset possibleMoves
            for indexRing in range(3):  # iterate through rings
                for indexPos, place in enumerate(self.millField[indexRing,:]):  # iterate through positions in the ring
                    if player == place:
                        newIndices = []
                        newIndices += [[indexRing, (indexPos+1)%8]]
                        newIndices += [[indexRing, (indexPos-1)%8]]
                        if indexPos%2 == 1:
                            if indexRing >=1:
                                newIndices += [[indexRing-1, indexPos]]
                            if indexRing <=1:
                                newIndices += [[indexRing+1, indexPos]]
                        
                        for (newIndexRing, newIndexPos) in newIndices:
                            if EMPTY == self.millField[newIndexRing, newIndexPos]:
                                self.possibleMoves = [1]    # set len(self.possibleMoves)>0
                                return
                                
        else:   # 3 remaining tokens -> jumping phase
            self.possibleMoves = [1]    # in jumping phase always a move is possible -> set len(self.possibleMoves)>0
    
    
    
# only for testing    
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
        
                    
    
    
    
