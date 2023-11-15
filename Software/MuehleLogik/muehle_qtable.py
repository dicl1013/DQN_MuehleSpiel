# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 19:07:11 2023

@author: Andre Kronsbein
"""

import numpy as np
from muehle_logik import *
import time
from timeit import default_timer as timer

MILLFIELD = 0 #definition beim Anlegen
NEXT_ACTION = 1 #definition beim Anlegen
INSTOCK_TOKENS = 2 #definition beim Anlegen
REMAINING_TOKENS = 3 #definition beim Anlegen
ACTUAL_PLAYER = 4 #definition beim Anlegen
QVALUE = 5 #definition beim Anlegen
POSSIBLE_MOVES = 6 #definition bei Bearbeitung durch Erreichen von nextIndex
PROCESSING_COUNTER = 7 #definition beim Anlegen
STATE = 8

QTable = []
nextIndex = 0
lengthTable = 0


if __name__=='__main__':  
    muehle = MillLogic()
    print("game started")
    tstart = timer()
    # erstelle ersten Eintrag für die QTable
    newEntry = {}
    newEntry[MILLFIELD], newEntry[NEXT_ACTION], possibleMoves, newEntry[INSTOCK_TOKENS], newEntry[REMAINING_TOKENS] = muehle.getFullState()
    fieldState = 0
    for i,place in enumerate(newEntry[MILLFIELD].flatten()):
        fieldState += place * (3**i)
    newEntry[STATE] = fieldState*100000 + newEntry[NEXT_ACTION].value*10000 + newEntry[INSTOCK_TOKENS][PLAYER1]*1000 + newEntry[INSTOCK_TOKENS][PLAYER2]*100 + newEntry[REMAINING_TOKENS][PLAYER1]*10 + newEntry[REMAINING_TOKENS][PLAYER2]
    # newEntry[COMPARE_LIST] = [newEntry[MILLFIELD].tolist(),newEntry[NEXT_ACTION].value, newEntry[INSTOCK_TOKENS][PLAYER1], newEntry[INSTOCK_TOKENS][PLAYER2], newEntry[REMAINING_TOKENS][PLAYER1], newEntry[REMAINING_TOKENS][PLAYER2]]
    if newEntry[NEXT_ACTION]==enAction.SetPlayer1 or newEntry[NEXT_ACTION]==enAction.ShiftPlayer1 or newEntry[NEXT_ACTION]==enAction.RemoveTokenPlayer1 or newEntry[NEXT_ACTION]==enAction.Player1Wins:
        newEntry[ACTUAL_PLAYER] = PLAYER1
    else:
        newEntry[ACTUAL_PLAYER] = PLAYER2
    newEntry[QVALUE] = 0
    newEntry[POSSIBLE_MOVES] = []
    newEntry[PROCESSING_COUNTER] = 0
    # füge neuen Eintrag zur QTable hinzu
    QTable += [newEntry]
    lengthTable += 1
    
    runs = 0
    measure = True
    while runs < 10000:
        runs +=1
        if runs%10 == 0:
            print(runs)
        # befülle POSSIBLE_MOVES des aktuellen Eintrags mit Indizees der Folgezustände, erstelle dabei Einträge für noch nicht vorhandene Zustände
        newLengthTable = lengthTable
        for posMove in possibleMoves:
            bStateFound = False
            if runs%100 == 0:
                if True == measure:
                    t1 = timer()
            # initialisiere MillLogic mit aktuellem Zustand
            muehle.initializeSpecificState(QTable[nextIndex][MILLFIELD], QTable[nextIndex][NEXT_ACTION], QTable[nextIndex][INSTOCK_TOKENS] )
            if runs%100 == 0:
                if True == measure:
                    t2 = timer()
            # setMove auf neuen Zustand
            bRet = muehle.setMove(posMove)
            if runs%100 == 0:
                if True == measure:
                    t3 = timer()
            if bRet == False:
                print("Invalid Move")
            # erstelle new Entry
            newEntry = {}
            newEntry[MILLFIELD], newEntry[NEXT_ACTION], _, newEntry[INSTOCK_TOKENS], newEntry[REMAINING_TOKENS] = muehle.getFullState()
            fieldState = 0
            for i,place in enumerate(newEntry[MILLFIELD].flatten()):
                fieldState += place * (3**i)
            newEntry[STATE] = fieldState*100000 + newEntry[NEXT_ACTION].value*10000 + newEntry[INSTOCK_TOKENS][PLAYER1]*1000 + newEntry[INSTOCK_TOKENS][PLAYER2]*100 + newEntry[REMAINING_TOKENS][PLAYER1]*10 + newEntry[REMAINING_TOKENS][PLAYER2]
            # newEntry[COMPARE_LIST] = [newEntry[MILLFIELD].tolist(),newEntry[NEXT_ACTION].value, newEntry[INSTOCK_TOKENS][PLAYER1], newEntry[INSTOCK_TOKENS][PLAYER2], newEntry[REMAINING_TOKENS][PLAYER1], newEntry[REMAINING_TOKENS][PLAYER2]]
            if runs%100 == 0:
                if True == measure:
                    t4 = timer()
           
            newState = newEntry[STATE]
            for index in range(lengthTable):    #############################################ToDo: OPTIMIZE THE NEXT THREE LINES ############################################################################################################################
                if QTable[index][STATE] == newState: 
                    # Eintrag bereits vorhanden
                    bStateFound = True
                    # füge Index des gefundenen Eintrags an array an
                    QTable[nextIndex][POSSIBLE_MOVES] += [index]
                    break;
            if runs%100 == 0:
                if True == measure:
                    measure = False
                    t5 = timer()
                    print(f"Run: {runs}")
                    print(f"initState: {t2-t1}")
                    print(f"setMove: {t3-t2}")
                    print(f"search Entry: {t5-t4}")
            if (runs+50)%100 == 0:
                if False == measure:
                    measure = True            
            if False == bStateFound:    # neuer Zustand -> erstelle neuen Eintrag
                if newEntry[NEXT_ACTION]==enAction.SetPlayer1 or newEntry[NEXT_ACTION]==enAction.ShiftPlayer1 or newEntry[NEXT_ACTION]==enAction.RemoveTokenPlayer1 or newEntry[NEXT_ACTION]==enAction.Player1Wins:
                    newEntry[ACTUAL_PLAYER] = PLAYER1
                else:
                    newEntry[ACTUAL_PLAYER] = PLAYER2
                newEntry[QVALUE] = 0
                newEntry[POSSIBLE_MOVES] = []
                newEntry[PROCESSING_COUNTER] = 0
                # füge neuen Eintrag zur QTable hinzu
                QTable += [newEntry]
                newLengthTable += 1
                # füge Index des neuen Eintrags (letzter Eintrag in QTable) an array an
                QTable[nextIndex][POSSIBLE_MOVES] += [newLengthTable-1]
        lengthTable = newLengthTable
                        
        #Bearbeitung des Eintrags abgeschlossen, gehe zu nächstem Eintrag
        nextIndex += 1
        if nextIndex < lengthTable:
            # initialisiere MillLogic mit dem Zustand aus dem Nächsten Eintrag, wenn dieser vorhanden ist
            muehle.initializeSpecificState(QTable[nextIndex][MILLFIELD], QTable[nextIndex][NEXT_ACTION], QTable[nextIndex][INSTOCK_TOKENS] )
            possibleMoves = muehle.getPossibleMoves()
        else:
            break;  # QTable ist vollständig
    
    tend = timer()
    print("QTable finished")
    print(f"total time: {tend-tstart}")
          
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    