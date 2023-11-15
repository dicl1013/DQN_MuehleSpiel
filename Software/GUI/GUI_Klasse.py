# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 17:28:23 2023


Befehl um Ressoourcen zu konvertieren:
pyrcc5 -o Ressourcen_rc.py Ressourcen.qrc

@author: Clemens
"""
"""
Notizen:
    
"""

import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QVBoxLayout
from PyQt5.QtCore import QTimer, QDateTime
#from PyQtDesinger_GUI import Ui_MainWindow
from PyQtDesinger_GUI import *

# Dataclass for Struct:
import dataclasses

# pickel for file-saving:
import pickle

# For graphs:
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# possible states of one place in the mill field
LEER=0
PLAYER1=1
PLAYER2=-1
PLAYER1_PossiblePos=3
PLAYER2_PossiblePos=4

"""-------------------------------------------------------------------------"""
""" Settings-Struct:"""
"""-------------------------------------------------------------------------""" 
@dataclasses.dataclass
class Settings:
    Player1_Human: bool = True
    Player1_AI_Data_Path: str = ""
    Player2_Human: bool = True
    Player2_AI_Data_Path: str = ""
    
    TimeAIvsAI: float = 1.0

"""-------------------------------------------------------------------------"""
""" Trainingsparameter-Struct:"""
"""-------------------------------------------------------------------------""" 
@dataclasses.dataclass
class Trainingsparameter:
    NumberGames: int = 1000     
    MaxMoves: int = 200
    TrainPhase1: bool = True
    TrainPhase2: bool = True
    TrainPhase3: bool = True
        
    RewardMove: int = -1
    RewardOwnMill: int = 200
    RewardEnemyMill: int = -200
    RewardWin: int = 5000
    RewardLoss: int = -5000
    
    Discount: float = 0.99
    LearningRate: float = 0.000001
    
    Epsilon: float = 1.0
    EpsilonDecay: float = 0.99975
    EpsilonMin: float = 0.0
    
    NumTrainingInstances: int = 32
    MinReplayMemorySize: int = 1024
    ReplayMemorySize: int = 1024
    MiniBatchSize: int = 1024
    TrainEveryXSteps: int = 16
    UpdateTargetEvery: int = 5

"""-------------------------------------------------------------------------"""
""" Validationparameter-Struct:"""
"""-------------------------------------------------------------------------""" 
@dataclasses.dataclass
class Validationparameter:
    NumberGames: int = 1000     
    MaxMoves: int = 200
    RandomReferenceModel: bool = True
    FirstMoveRandom: bool = True
    ReferenceMovesRandom: bool = False
    SwitchPlayer: bool = True

    NumValidationInstances: int = 32


"""-------------------------------------------------------------------------"""
""" class Frm_main(QMainWindow, Ui_MainWindow):"""
"""-------------------------------------------------------------------------""" 
class Frm_main(QMainWindow, Ui_MainWindow):
       
    # Update Intervall in ms
    GUIUpdateIntervall = 33
    # millField = np.zeros([3,8], dtype=int)
    
    bTokenClickedForShifting = False    
    PosTokenClickedForShiftung = [0, 0] # Position of the Token which is clicked to be shifted
    
    # Array with constants to set tile color:
    lstTextToken  = {LEER: u"<html><head/><body><p align=\"center\"></p></body></html>",
                     PLAYER1: u"<html><head/><body><p align=\"center\"><img src=\":/Spielfeld/Spielstein_Blau\"/></p></body></html>",
                     PLAYER2: u"<html><head/><body><p align=\"center\"><img src=\":/Spielfeld/Spielstein_Rot\"/></p></body></html>",
                     PLAYER1_PossiblePos: u"<html><head/><body><p align=\"center\"><img src=\":/Spielfeld/Spielstein_Blau_Hell\"/></p></body></html>",
                     PLAYER2_PossiblePos: u"<html><head/><body><p align=\"center\"><img src=\":/Spielfeld/Spielstein_Rot_Hell\"/></p></body></html>"}
    
    # List with text of the next action
    lstTextNextAction = [u"<html><head/><body><p><span style=\"font-size:12pt; font-weight:600;\">Spieler 1: Spielstein setzen</span></p></body></html>",
                         u"<html><head/><body><p><span style=\"font-size:12pt; font-weight:600;\">Spieler 2: Spielstein setzen</span></p></body></html>",
                         u"<html><head/><body><p><span style=\"font-size:12pt; font-weight:600;\">Spieler 1: Spielstein verschieben</span></p></body></html>",
                         u"<html><head/><body><p><span style=\"font-size:12pt; font-weight:600;\">Spieler 2: Spielstein verschieben</span></p></body></html>",
                         u"<html><head/><body><p><span style=\"font-size:12pt; font-weight:600;\">Spieler 1: Spielstein entfernen</span></p></body></html>",
                         u"<html><head/><body><p><span style=\"font-size:12pt; font-weight:600;\">Spieler 2: Spielstein entfernen</span></p></body></html>",
                         u"<html><head/><body><p><span style=\"font-size:12pt; font-weight:600;\">Spieler 1: Gewinnt</span></p></body></html>",
                         u"<html><head/><body><p><span style=\"font-size:12pt; font-weight:600;\">Spieler 2: Gewinnt</span></p></body></html>"]
    
     
    
    def __init__(self, DataQueIn, DataQueOut, enActionVar):
        """
        Parameters
        ----------
        DataQueIn : Queue
            Data Outside->GUI.
        DataQueOut : Queue
            Data Outside<-GUI..
        enActionVar : enum
            Enum with possible Action.
        """

        super().__init__()
        self.setupUi(self)        
    
        # Set the cursor to wait:
        QApplication.setOverrideCursor(Qt.WaitCursor)
                
        # Cyclic update timer:
        self.TimerUpdate = QTimer()
        self.TimerUpdate.timeout.connect(self.UpdateGUI) # Connect function
        self.TimerUpdate.start(self.GUIUpdateIntervall)
        
        # Timer for Training is running:
        self.TimerTraining = QTimer()
        self.TimerTraining.timeout.connect(self.UpdateTrainingTimer)
        self.TimeStartTraining = QDateTime.currentDateTime()
        
        # Timer for validation is running:
        self.TimerValidation = QTimer()
        self.TimerValidation.timeout.connect(self.UpdateValidationTimer)
        self.TimeStartValidation= QDateTime.currentDateTime()
        
        # Timer for AI vs AI game:
        self.TimerAIvsAI = QTimer()
        self.TimerAIvsAI.timeout.connect(self.UpdateGameAIvsAI)
        
        
        # Variable for current status:
        self.TrainingIsRunning = False
        self.ValidationIsRunning = False
        self.GameIsRunning = False
        
        # Queues for threading:
        self.DataQueIn = DataQueIn
        self.DataQueOut = DataQueOut
        
        # Enum of the actions:
        self.enAction = enActionVar    
        
        # Buttons:
        self.btnRestart.clicked.connect(self.ResetGame)
        
        # Map click event of all possible game field positions on the click function:
        self.lblSt00.mousePressEvent = self.PositionClicked_lblSt00
        self.lblSt01.mousePressEvent = self.PositionClicked_lblSt01
        self.lblSt02.mousePressEvent = self.PositionClicked_lblSt02
        self.lblSt03.mousePressEvent = self.PositionClicked_lblSt03
        self.lblSt04.mousePressEvent = self.PositionClicked_lblSt04
        self.lblSt05.mousePressEvent = self.PositionClicked_lblSt05
        self.lblSt06.mousePressEvent = self.PositionClicked_lblSt06
        self.lblSt07.mousePressEvent = self.PositionClicked_lblSt07
        
        self.lblSt10.mousePressEvent = self.PositionClicked_lblSt10
        self.lblSt11.mousePressEvent = self.PositionClicked_lblSt11
        self.lblSt12.mousePressEvent = self.PositionClicked_lblSt12
        self.lblSt13.mousePressEvent = self.PositionClicked_lblSt13
        self.lblSt14.mousePressEvent = self.PositionClicked_lblSt14
        self.lblSt15.mousePressEvent = self.PositionClicked_lblSt15
        self.lblSt16.mousePressEvent = self.PositionClicked_lblSt16
        self.lblSt17.mousePressEvent = self.PositionClicked_lblSt17
        
        self.lblSt20.mousePressEvent = self.PositionClicked_lblSt20
        self.lblSt21.mousePressEvent = self.PositionClicked_lblSt21
        self.lblSt22.mousePressEvent = self.PositionClicked_lblSt22
        self.lblSt23.mousePressEvent = self.PositionClicked_lblSt23
        self.lblSt24.mousePressEvent = self.PositionClicked_lblSt24
        self.lblSt25.mousePressEvent = self.PositionClicked_lblSt25
        self.lblSt26.mousePressEvent = self.PositionClicked_lblSt26
        self.lblSt27.mousePressEvent = self.PositionClicked_lblSt27
    
        # Matrix with pointer on mill-field position label-classes:
        self.millField_lblSt = [[self.lblSt00, self.lblSt01, self.lblSt02, self.lblSt03, self.lblSt04, self.lblSt05, self.lblSt06, self.lblSt07],
                                [self.lblSt10, self.lblSt11, self.lblSt12, self.lblSt13, self.lblSt14, self.lblSt15, self.lblSt16, self.lblSt17],
                                [self.lblSt20, self.lblSt21, self.lblSt22, self.lblSt23, self.lblSt24, self.lblSt25, self.lblSt26, self.lblSt27]]
       
        # Matrix with pointers on Stock Label-Clases
        self.millField_lblAv = {PLAYER1: [self.lblAvP10, self.lblAvP11, self.lblAvP12, self.lblAvP13, self.lblAvP14, self.lblAvP15, self.lblAvP16, self.lblAvP17, self.lblAvP18],
                                PLAYER2: [self.lblAvP20, self.lblAvP21, self.lblAvP22, self.lblAvP23, self.lblAvP24, self.lblAvP25, self.lblAvP26, self.lblAvP27, self.lblAvP28]}

        # Settings:
        self.Settings = Settings()   
        self.tmpSettings = Settings()   # Temp settings, so you need to click save or change tab to actual set the settings
        self.btnSettingSave.clicked.connect(self.SaveSettings)
        self.tabWidget.tabBarClicked.connect(self.HandleTabChange)
        #self.tabWidget.changeEvent = self.HandleTabChange
        # Player settings:
        self.rBtnPlayer1Human.clicked.connect(self.HandleSettings)
        self.rBtnPlayer1KI.clicked.connect(self.HandleSettings)
        self.rBtnPlayer2Human.clicked.connect(self.HandleSettings)
        self.rBtnPlayer2KI.clicked.connect(self.HandleSettings)
        self.dsBoxTimeAIvsAI.valueChanged.connect(self.HandleSettings)
        
        self.btnLoadKIData1.clicked.connect(self.LoadKIData1)
        self.btnLoadKIData2.clicked.connect(self.LoadKIData2)
        
        self.FiletypesKIData = "KI-Modell-Data (*.kidat)"
        self.FiletypesTrainingsparameter = "Trainingsparameter (*.Tdat)"
        self.FiletypesValidationparameter = "Validierungsparameter (*.Vdat)"
              
        # Training-Buttons functions and enable elements:
        self.btnTrainingStart.clicked.connect(self.StartTraining)    
        self.btnTrainingStop.clicked.connect(self.StopTraining)
        self.btnLoadTrainingModel.clicked.connect(self.LoadTrainingModel)
        self.btnSaveTrainingModel.clicked.connect(self.SaveTrainingModel)
        self.btnLoadTrainingParameter.clicked.connect(self.LoadTrainingsParameter)
        self.btnSaveTrainingParameter.clicked.connect(self.SaveTrainingsParameter)
        self.btnResetTrainingModel.clicked.connect(self.ResetTrainingModel)
        self.SetTrainingElementsEnabled(True)
        self.SetCurrentNumberOfGamesMoves(1)
          
        # Validation-Buttons functions and enable elements:
        self.btnValidationStart.clicked.connect(self.StartValidation)    
        self.btnValidationStop.clicked.connect(self.StopValidation)
        self.btnLoadValidationModel.clicked.connect(self.LoadValidationModel)
        self.btnLoadReferenceModel.clicked.connect(self.LoadReferenceModel)
        self.btnLoadValidationParameter.clicked.connect(self.LoadValidationParameter)
        self.btnSaveValidationParameter.clicked.connect(self.SaveValidationParameter)
        self.SetValidationElementsEnabled(True)
        self.SetCurrentNumberOfValidationGames(1)
        
        # File-path of the current validation and reference model:
        self.FilePathValidationModel = ""
        self.FilePathReferenceModel = ""
        
        # Plots:
        self.InitTrainingPlot()
        self.InitValidationPlot()

        self.SetToolTips()
      
        # Set game field to initial state:
        self.ResetGame()
        
        
    """---------------------------------------------------------------------"""
    """ Functions for init: """
    """---------------------------------------------------------------------"""  
    def SetToolTips(self):
        """
        Function to set the tooltips. Tooltips in QtDesigner is not working,
        so it is realised in this function
        """
        self.dsBoxDiscount.setToolTip("Discount-Faktor der Q-Learning-Formel. Schwächt zukünft zukünftige Q-Werte ab.")
        self.dsBoxEpsilon.setToolTip("Startwert von Epsilon. Epsilon gibt an, mit welcher Wahrscheinlichkeit der aktuell beste Zug oder ein zufälliger Zug ausgewählt wird.")
        self.dsBoxEpsilonDecay.setToolTip("Abnahme des Epsilon-Wertes nach jedem Spiel. Dieser Wert wird mit dem aktuellen Epsilon-Wert mulipliziert.")
        self.dsBoxEpsilonMin.setToolTip("Kleinster zulässiger Epsilon-Wert.")
        
        self.sBoxNumTrainingInstances.setToolTip("Anzahl der Spiele die gleichzeitig gespielt werden.")
        self.sBoxNumTrainingInstances.setToolTip("Anzahl der Spiele die gleichzeitig gespielt werden.")
        
    
    """---------------------------------------------------------------------"""
    """ Functions to set game state: """
    """---------------------------------------------------------------------"""     
    def UpdateGUI(self):
        """
        Called cyclically to update GUI
        """
        # Enables main tabs - needed to be able to abort tab change
        # Tab is disabled in HandleTabChange(self, newIndex)
        self.tabWidget.setTabEnabled(0, True)
        self.tabWidget.setTabEnabled(2, True)
        self.tabWidget.setTabEnabled(3, True)
                
        # If a Game is running and only AI should be play, start the timer for the update:
        # Is here, so the player settings can be change due a game
        # Start only one time, otherwise it will be permantly reseted
        if self.GameIsRunning and (not self.Settings.Player1_Human or not self.Settings.Player2_Human) and not self.TimerAIvsAI.isActive():
            # print("Start")
            self.TimerAIvsAI.start(self.Settings.TimeAIvsAI * 1000)
        elif not self.GameIsRunning or (self.Settings.Player1_Human and self.Settings.Player2_Human):
            # print("Stop")
            self.TimerAIvsAI.stop()
        
        
        data = None
        
        # Try getting data from queue:
        try:            
            data = self.DataQueIn.get_nowait()
        except:
            pass
        
        # If there is data:
        if data:
            print(f"GUI:{data}") 
            
            # If the state should be set:
            if data[0] == "SetState":
                self.SetState(data[1], data[2], data[3])
                
            # If the training should be stopped::
            elif data[0] == "StopTraining":
                self.StopTraining()
            
            # If the training is running, set current status:
            elif data[0] == "TrainingRunning":
                self.TrainingRunning(data)
                                
            # If the training is finished:               
            elif data[0] == "TrainingFinished":
                self.TrainingFinished()
            
            # If the loading of a model was successfull:
            elif data[0] == "LoadTrainingModel_Success":
                self.LoadTrainingModel_Success(data)    
            
            # If the settings should be loaded:
            elif data[0] == "LoadSettings":
                self.LoadSettings(data)
            
            # If Error Message for not able to load settings:
            elif data[0] == "ShowError":
                self.ShowError(data)
            
            # If the Validation should be stopped::
            elif data[0] == "StopValidation":
                self.StopValidation()
            
            # If the Validation is running, set current status:
            elif data[0] == "ValidationRunning":
                self.ValidationRunning(data)
                                
            # If the Validation is finished:               
            elif data[0] == "ValidationFinished":
                self.ValidationFinished(data)
            
            # If the loading of a model was successfull:
            elif data[0] == "LoadValidationModel_Success":
                self.LoadValidationModel_Success(data)    
            
            # If the loading of a model was successfull:
            elif data[0] == "LoadReferenceModel_Success":
                self.LoadReferenceModel_Success(data) 
            

            
    def SetState(self, millField, nextAction, InStockTokens):
        """
        Sets the current status based on the given mill field array.
        Also indicates what action to take next and how many tokens are left 
        in stock

        Parameters
        ----------
        millField : 
            Current condition of the game field.
        nextPossibleAction : 
            Next action, e.g. player 1 takes a token from player 2.
        """
                
        # Check if a game is currently running:
        # If Player 1 has still all tokens then is no game running
        if InStockTokens[PLAYER1] == 9:
            self.GameIsRunning = False
        else:
            self.GameIsRunning = True
                
        # Go through all rings:
        for ring in range(3):
            # Go through all positions:
            for pos in range(8):
                # Determine the owner of the playing field:
                TokenOwner = millField[ring][pos]
                # Set game field position according to the owner:
                self.millField_lblSt[ring][pos].setText(self.lstTextToken[TokenOwner])
        
        self.lblNextAction.setText(self.lstTextNextAction[nextAction.value]) # Set the text of the next Action
            
        # Refresh stock:
        self.SetStockState(InStockTokens[PLAYER1], InStockTokens[PLAYER2])
    
    def ResetGame(self):
        """
        Resets the game.
        """
        # Puts the Reset Command on the Queue
        self.DataQueOut.put(["ResetGame"])
     
        
    def SetStockState(self, inStock1, inStock2):
        """
        Function to set the labels of the stock
        """
        # Loop over all possible Stock positions
        for i in range(0, 9):
            # Player 1:
            # If Position is in Stock:
            if i < inStock1:
                self.millField_lblAv[PLAYER1][i].setText(self.lstTextToken[PLAYER1])    # Set Token
            else:
                self.millField_lblAv[PLAYER1][i].setText(self.lstTextToken[LEER])       # Otherwise empty 
                
            # Player 2:   
            # If Position is in Stock:
            if i < inStock2:
                self.millField_lblAv[PLAYER2][i].setText(self.lstTextToken[PLAYER2])    # Set Token
            else:
                self.millField_lblAv[PLAYER2][i].setText(self.lstTextToken[LEER])       # Otherwise empty 


    def ShowPossibleMoves(self, millField, indexRing, indexPos, PlayerNr):
        """
        Return a mill field with possible position marked for the shifting
        """
        # Get a list with all possible moves from this position:
        lstPossibleMoves = self.ExternGetMovesFromSelectedToken(indexRing, indexPos)
        # Go trough list and mark all possible moving positions:        
        for pos in lstPossibleMoves:
            # Mark position depending on token owner
            if PlayerNr == PLAYER1:
                millField[pos[0]][pos[1]] = PLAYER1_PossiblePos
            else:
                millField[pos[0]][pos[1]] = PLAYER2_PossiblePos
        return millField
            
    
    def HidePossibleMoves(self, millField):
        """
        Return a mill field with removed possible positions
        """      
        # Remove all possible position marking in the mill field:          
        for pos in millField:
            if millField[pos[0]][pos[1]] == PLAYER1_PossiblePos or millField[pos[0]][pos[1]] == PLAYER2_PossiblePos:
                millField[pos[0]][pos[1]] = LEER
        return millField
    
    def UpdateGameAIvsAI(self):
        """
        Called from the AI vs AI timer, so the next AI move will be calculated
        """
        self.DataQueOut.put(["AIPlayerHandler"])
    
    """---------------------------------------------------------------------"""
    """ Functions for errors: """
    """---------------------------------------------------------------------"""
    def ShowError(self, data):
        """
        Shows an error message from the queue
        """            
        # Show Error Message
        retVal = QMessageBox.critical(self, 
                                    "Fehler",
                                    data[1],
                                    buttons=QMessageBox.Ok,
                                    defaultButton=QMessageBox.Ok)
        
        # Restore the normal cursor:
        QApplication.restoreOverrideCursor()
        
    """---------------------------------------------------------------------"""
    """ Functions for settings: """
    """---------------------------------------------------------------------"""
    def SaveSettings(self):
        """
        Function to save the settings
        """ 
        # Copy temp settings in settings: 
        self.Settings = dataclasses.replace(self.tmpSettings)      
        
        # Send Settings to main, so they are active:
        self.DataQueOut.put(["SaveSettings", self.Settings])
               
    def LoadSettings(self, data):
        """
        Function to load the settings from the Queue
        """       
        self.Settings = data[1]
        self.tmpSettings = data[1]

        # When the settings are loaded, the software is startet, so
        # Restore the normal cursor:
        QApplication.restoreOverrideCursor()
     
    def SetGUIElementsSettings(self):
        """
        Set the GUI elements based on the current settings
        """
        #print("SetGUIElementsSettings:", self.tmpSettings)
        # Player 1:
        self.rBtnPlayer1Human.setChecked(self.tmpSettings.Player1_Human)
        self.rBtnPlayer1KI.setChecked(not self.tmpSettings.Player1_Human)
        
        # Create String to show saved Filepath:
        DisplayTxt = f"<html><head/><body><p><span style=\"font-size:10pt;\">Geladen: {self.tmpSettings.Player1_AI_Data_Path}</span></p></body></html>"
        self.lblTxtLoadeFilePath1.setText(DisplayTxt)
        
        # Player 2:
        self.rBtnPlayer2Human.setChecked(self.tmpSettings.Player2_Human)
        self.rBtnPlayer2KI.setChecked(not self.tmpSettings.Player2_Human)
             
        # Create String to show saved Filepath:
        DisplayTxt = f"<html><head/><body><p><span style=\"font-size:10pt;\">Geladen: {self.tmpSettings.Player2_AI_Data_Path}</span></p></body></html>"
        self.lblTxtLoadeFilePath2.setText(DisplayTxt)
        
        # Time for the AI vs AI games:
        self.dsBoxTimeAIvsAI.setValue(self.tmpSettings.TimeAIvsAI)
     
    def HandleTabChange(self, newIndex):
        """
        Handles the change of a tab. index is index of clicked tab
        If tab is changed, while settings are not saved, show Dialog.
        """       
        # With click event, tab is not switched yet: 
        currentIndex = self.tabWidget.currentIndex()
        
        # Change from Anywhere->Settings:
        if (currentIndex != 1) and (newIndex == 1):
            #print("Main->Settings")
            # Copy current settings in temp and sets the GUI-elements: 
            self.tmpSettings = dataclasses.replace(self.Settings)
            self.SetGUIElementsSettings()
            
        # Change from Settings->Anywhere:
        elif (currentIndex == 1) and (newIndex != 1):
            #print("Settings->Main")
            # If settings changed:
            if self.tmpSettings != self.Settings:
                #print("tmpSettings!=Settings")
                # Show dialog if the settings should be saved:
                retVal = QMessageBox.warning(self, 
                                             "Warnung",
                                             "Einstellungen wurden geändert, aber noch nicht gespeichert. Speichern?",
                                             buttons=QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                                             defaultButton=QMessageBox.No)    
                
                # If data should be saved:
                if retVal == QMessageBox.Yes:
                    self.SaveSettings()
                # If tab change should be aborted:
                elif retVal == QMessageBox.Cancel:
                    # Disable the tab, so the change will not be happen:
                    self.tabWidget.setTabEnabled(newIndex, False)
        
    def HandleSettings(self):
        """
        Function to handle the Settings. Write the current status to the
        temp settings struct
        """   
        self.tmpSettings.Player1_Human = self.rBtnPlayer1Human.isChecked()
        self.tmpSettings.Player2_Human = self.rBtnPlayer2Human.isChecked()

        self.tmpSettings.TimeAIvsAI = self.dsBoxTimeAIvsAI.value()

        #self.DataQueOut.put(["SetPlayerSettings", Player1_Human, Player2_Human])
    

    def LoadKIData1(self):
        """
        Function to load the Data for KI Player 1
        """
        # Open a File-Dialog to select the KI-data file:
        # FilePath = QFileDialog.getOpenFileName(self, caption='Open a file', filter=self.FiletypesKIData)
        FilePath = QFileDialog().getExistingDirectory(self, caption="Ordner mit gespeicherten Modellen auswählen")
        
        # If a file was selected:
        if FilePath != "":
            # Save the path in the temp settings and set the GUI-Elements to show the path:
            # The model will first be loaded, when the settings are saved
            self.tmpSettings.Player1_AI_Data_Path = FilePath
            self.SetGUIElementsSettings()

    
    def LoadKIData2(self):
        """
        Function to load the Data for KI Player 2
        """        
        # Open a File-Dialog to select the KI-data file:
        # FilePath = QFileDialog.getOpenFileName(self, caption='Open a file', filter=self.FiletypesKIData)
        FilePath = QFileDialog().getExistingDirectory(self, caption="Ordner mit gespeicherten Modellen auswählen")
        
        # If a file was selected:
        if FilePath != "":
            # Save the path in the temp settings and set the GUI-Elements to show the path:
            # The model will first be loaded, when the settings are saved
            self.tmpSettings.Player2_AI_Data_Path = FilePath
            self.SetGUIElementsSettings()

    
    """---------------------------------------------------------------------"""
    """ Functions for the training: """
    """---------------------------------------------------------------------"""
    def StartTraining(self):
        """
        Function to start training. Puts number of games, max number of moves 
        and the game phase to be trainend on the queue.
        """
        
        # If a game is running:
        """
        if self.GameIsRunning:
            # Show warning:
            retVal = QMessageBox.warning(self, 
                                         "Warnung",
                                         "Das Spiel wurde noch nicht zu Ende gespielt. Soll das Training dennoch gestartet werden?",
                                         buttons=QMessageBox.Yes | QMessageBox.No,
                                         defaultButton=QMessageBox.No)
            
            # If the game should not be stopped:    
            if retVal == QMessageBox.No:
                return
            # If the game should be stopped:
            #else:
            #    self.ResetGame()
        """
        # Deactivate the GUI elements:
        self.SetTrainingElementsEnabled(False)
        
        # Save the time of the start moment and start timer:
        self.TimeStartTraining = QDateTime.currentDateTime()
        self.TimerTraining.start(1000)
        
        # Reset the current values:
        self.UpdateTrainingTimer()
        self.SetCurrentNumberOfGamesMoves(1)
        
        # Reset the training plot:
        self.ResetTrainingPlot()
        
        # Get Current training parameter:
        CurrentTrainingsParameter = self.GetCurrentTrainingsParameter()    
        
        #
        self.TrainingIsRunning = True
        
        # Put the Event on the queue:
        self.DataQueOut.put(["StartTraining", CurrentTrainingsParameter])
        
    def StopTraining(self):
        """
        Stop the training.
        """
        # Stop the timer:
        self.TimerTraining.stop()
            
        # Activate the GUI elements:
        self.SetTrainingElementsEnabled(True)
        
        # 
        self.TrainingIsRunning = False
        
        # Put the stop event on the queue:
        self.DataQueOut.put(["StopTraining"]) 
    
    def GetCurrentTrainingsParameter(self):
        """
        Reads the current trainingsparameter and returns the Struct with all
        parameter
        """
        # 
        CurrentPara = Trainingsparameter()
        
        # Fill the struct with all current training parameters:        
        CurrentPara.NumberGames = self.sBoxNumberGames.value()
        CurrentPara.MaxMoves = self.sBoxMaxMoves.value()
        CurrentPara.TrainPhase1 = self.cBoxTrainPhase1.isChecked()
        CurrentPara.TrainPhase2 = self.cBoxTrainPhase2.isChecked()
        CurrentPara.TrainPhase3 = self.cBoxTrainPhase3.isChecked()
                        
        CurrentPara.RewardMove = self.sBoxRewardMove.value()
        CurrentPara.RewardOwnMill = self.sBoxRewardOwnMill.value()
        CurrentPara.RewardEnemyMill = self.sBoxRewardEnemyMill.value()
        CurrentPara.RewardWin = self.sBoxRewardWin.value()
        CurrentPara.RewardLoss = self.sBoxRewardLoss.value()
        
        CurrentPara.Discount = self.dsBoxDiscount.value()     
        CurrentPara.LearningRate = self.dsBoxLearningRate.value()
        
        CurrentPara.Epsilon = self.dsBoxEpsilon.value()
        CurrentPara.EpsilonDecay = self.dsBoxEpsilonDecay.value()
        CurrentPara.EpsilonMin = self.dsBoxEpsilonMin.value()
        
        CurrentPara.NumTrainingInstances = self.sBoxNumTrainingInstances.value()
        CurrentPara.MinReplayMemorySize = self.sBoxMinReplayMemorySize.value()
        CurrentPara.ReplayMemorySize = self.sBoxReplayMemorySize.value()
        CurrentPara.MiniBatchSize = self.sBoxMiniBatchSize.value()
        CurrentPara.TrainEveryXSteps = self.sBoxTrainEveryXSteps.value()
        CurrentPara.UpdateTargetEvery = self.sBoxUpdateTargetEvery.value()

        return CurrentPara
     
    def SetCurrentTrainingsParameter(self, NewPara):
        """
        Sets new trainings parameter in the GUI based on the given data
        """   
        self.sBoxNumberGames.setValue(NewPara.NumberGames)
        self.sBoxMaxMoves.setValue(NewPara.MaxMoves)
        self.cBoxTrainPhase1.setChecked(NewPara.TrainPhase1)
        self.cBoxTrainPhase2.setChecked(NewPara.TrainPhase2)
        self.cBoxTrainPhase3.setChecked(NewPara.TrainPhase3)
                        
        self.sBoxRewardMove.setValue(NewPara.RewardMove)
        self.sBoxRewardOwnMill.setValue(NewPara.RewardOwnMill)
        self.sBoxRewardEnemyMill.setValue(NewPara.RewardEnemyMill)
        self.sBoxRewardWin.setValue(NewPara.RewardWin)
        self.sBoxRewardLoss.setValue(NewPara.RewardLoss)
       
        self.dsBoxDiscount.setValue(NewPara.Discount)   
        self.dsBoxLearningRate.setValue(NewPara.LearningRate)
        
        self.dsBoxEpsilon.setValue(NewPara.Epsilon)
        self.dsBoxEpsilonDecay.setValue(NewPara.EpsilonDecay)
        self.dsBoxEpsilonMin.setValue(NewPara.EpsilonMin)
        
        self.sBoxNumTrainingInstances.setValue(NewPara.NumTrainingInstances)
        self.sBoxReplayMemorySize.setValue(NewPara.ReplayMemorySize)
        self.sBoxMinReplayMemorySize.setValue(NewPara.MinReplayMemorySize)
        self.sBoxMiniBatchSize.setValue(NewPara.MiniBatchSize)
        self.sBoxTrainEveryXSteps.setValue(NewPara.TrainEveryXSteps)
        self.sBoxUpdateTargetEvery.setValue(NewPara.UpdateTargetEvery)
        
    def LoadTrainingsParameter(self):
        """
        Loads Trainingsparameter from a selected file
        """
        # Open a File-Dialog to save the model:
        FilePath = QFileDialog().getOpenFileName(self, caption="Trainingsparameter laden", filter=self.FiletypesTrainingsparameter)
        
        
        # If an file was selected:
        if FilePath[0] != "":
            # Reset String of the loaded file path:
            DisplayTxt = f"<html><head/><body><p><span style=\"font-size:10pt;\">Geladen:</span></p></body></html>"
            self.lblTxtLoadedTrainingsParameter.setText(DisplayTxt)
            try:
                with open(FilePath[0], "rb") as file:
                    TrainingsParameter = pickle.load(file)
                self.SetCurrentTrainingsParameter(TrainingsParameter)
                
                # Create String to show the opened file path:
                DisplayTxt = f"<html><head/><body><p><span style=\"font-size:10pt;\">Geladen: {FilePath[0]}</span></p></body></html>"
                self.lblTxtLoadedTrainingsParameter.setText(DisplayTxt)
            except:            
                # Show error: (Pop-Up need to be created in the GUI)    
                self.ShowError(["", "Trainingsparameter konnten nicht geladen werden"])
    
    def SaveTrainingsParameter(self):
        """
        Save the current Trainingsparameter to a file
        """
        # Open a File-Dialog to save the parameters:
        FilePath = QFileDialog().getSaveFileName(self, caption="Trainingsparameter speichern", filter=self.FiletypesTrainingsparameter)
        
        CurrentTrainingsParameter = self.GetCurrentTrainingsParameter()
        
        # If a file should be saved:
        if FilePath[0] != "":
            # Save settings in file:
            with open(FilePath[0], "wb") as file:
                pickle.dump(CurrentTrainingsParameter, file)
                    
    def LoadTrainingModel(self):
        """
        Function to load a model, so it can be trained more
        """
        # Open a File-Dialog to select the folder with the model data:
        FilePath = QFileDialog().getExistingDirectory(self, caption="Ordner mit gespeicherten Modellen auswählen")
        
        # If a file was selected:
        if FilePath != "":
            # Set the cursor to wait:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            # Put the Filepath on the queue, so the DQN can laod the model:
            self.DataQueOut.put(["LoadTrainingModel", FilePath])
     
    def SaveTrainingModel(self):
        """
        Function to save a model.
        """ 
        # Open a File-Dialog to save the model:
        FilePath = QFileDialog().getExistingDirectory(self, caption="Leeren Ordner für Modell auswählen")
        
        # If a file should be saved:
        if FilePath != "":
            # Put the Filepath on the queue, so the DQN can save the model:
            self.DataQueOut.put(["SaveTrainingModel", FilePath])
    
    def LoadTrainingModel_Success(self, data):
        """
        Is called, if a model was sucessfull saved
        """
        FilePath = data[1]
        
        # Set String of the loaded file path:
        DisplayTxt = f"<html><head/><body><p><span style=\"font-size:10pt;\">Geladen: {FilePath}</span></p></body></html>"
        self.lblTxtLoadedTrainingModel.setText(DisplayTxt)
        
        QApplication.restoreOverrideCursor()
    
    def ResetTrainingModel(self):
        """
        Function to reset the current model.
        """ 
        retVal = QMessageBox.warning(self, 
                                     "Warnung",
                                     "Soll ein neues Modell erstellt werden?",
                                     buttons=QMessageBox.Yes | QMessageBox.No,
                                     defaultButton=QMessageBox.No)    
        
        # If model should be resetet:
        if retVal == QMessageBox.Yes:
            # Reset String of the loaded file path:
            DisplayTxt = "<html><head/><body><p><span style=\"font-size:10pt;\">Geladen: </span></p></body></html>"
            self.lblTxtLoadedTrainingModel.setText(DisplayTxt)
            
            self.DataQueOut.put(["ResetTrainingModel"])

    
    def TrainingRunning(self, data):
        """
        Is called, when the training is running and new data are available from the queue
        """
        numberOfGames, AllStepsPerGame = data[1], data[2]
        self.SetCurrentNumberOfGamesMoves(numberOfGames)
        
        # If games where played:
        if numberOfGames > 1:
            self.UpdateTrainingPlot(AllStepsPerGame)
        
     
    def TrainingFinished(self):
        """
        Function when the training is finished
        """
        # Stop the timer:
        self.TimerTraining.stop()
            
        # Activate the GUI elements:
        self.SetTrainingElementsEnabled(True)
        
        # Sets progress bar 100 % (So it looks nice):
        MaxNumberGames = self.sBoxNumberGames.value()
        self.SetCurrentNumberOfGamesMoves(MaxNumberGames)
        
        # Deactivate Training:
        self.TrainingIsRunning = False
        
        
    def SetTrainingElementsEnabled(self, enable):
        """
        Enables or disable the GUI Elements for the training.
        Used to deactivate the elemts while training is running
        """        
        # Enable/Disable changing of training parameter settings:
        self.btnTrainingStart.setEnabled(enable)
        self.btnTrainingStop.setEnabled(not enable)
        
        self.btnLoadTrainingParameter.setEnabled(enable)
    
        self.btnSaveTrainingModel.setEnabled(enable)
        self.btnLoadTrainingModel.setEnabled(enable)
        self.btnResetTrainingModel.setEnabled(enable)
        
        self.sBoxNumberGames.setEnabled(enable)        
        self.sBoxMaxMoves.setEnabled(enable)
        self.cBoxTrainPhase1.setEnabled(enable)
        self.cBoxTrainPhase2.setEnabled(enable)
        self.cBoxTrainPhase3.setEnabled(enable)
        
        self.sBoxRewardMove.setEnabled(enable)
        self.sBoxRewardOwnMill.setEnabled(enable)
        self.sBoxRewardEnemyMill.setEnabled(enable)
        self.sBoxRewardWin.setEnabled(enable)
        self.sBoxRewardLoss.setEnabled(enable)
        
        self.dsBoxDiscount.setEnabled(enable)
        self.dsBoxLearningRate.setEnabled(enable)
        
        self.dsBoxEpsilon.setEnabled(enable)
        self.dsBoxEpsilonDecay.setEnabled(enable)
        self.dsBoxEpsilonMin.setEnabled(enable)

        self.sBoxNumTrainingInstances.setEnabled(enable)
        self.sBoxReplayMemorySize.setEnabled(enable)
        self.sBoxMinReplayMemorySize.setEnabled(enable)
        self.sBoxMiniBatchSize.setEnabled(enable)
        self.sBoxTrainEveryXSteps.setEnabled(enable)
        self.sBoxUpdateTargetEvery.setEnabled(enable)
    
    def UpdateTrainingTimer(self):
        """
        Updates the timer since when the training is running.
        Is called by the timer event
        """
        # Calculate time difference since the start of the training:
        DeltaTime = QDateTime.currentDateTime().toPyDateTime().replace(microsecond=0) \
            - self.TimeStartTraining.toPyDateTime().replace(microsecond=0)
        
        # Create String to show text:
        DisplayTxt = f"<html><head/><body><p><span style=\"font-size:12pt;\">Training läuft seit {DeltaTime} hh:mm:ss</span></p></body></html>"
        
        # Set the text in the GUI:
        self.lblTxtTrainingSince.setText(DisplayTxt)
     
    def SetCurrentNumberOfGamesMoves(self, numberOfGames):
        """
        Set the text and the progress bar for the current number of games and moves.
        """
        MaxNumberGames = self.sBoxNumberGames.value()
        
        # Create String to show text:
        DisplayTxtGames = f"<html><head/><body><p><span style=\"font-size:12pt;\">Spiel {numberOfGames:.0f} von {MaxNumberGames}</span></p></body></html>"
         
        # Set the text in the GUI:
        self.lblTxtActualGames.setText(DisplayTxtGames)
         
        # Set the progress bar, value in procent:
        self.pbarTraining.setValue(numberOfGames / MaxNumberGames * 100)
        
    """---------------------------------------------------------------------"""
    """ Functions for training graphics:"""
    """---------------------------------------------------------------------"""
    def InitTrainingPlot(self):
        """
        Inits the training plot
        """
        fig = Figure()
        fig.set_facecolor("#d9d9d9")    # Set background color
        
        # Sets a border around the complete plot
        self.frmPlotTraining.setStyleSheet("border: 1px solid #d72305;")
        
        # Creats axis as class variable, so can be used later again
        self.PlotTrainingAx = fig.add_subplot(111)
        self.PlotTrainingAx.set_xlabel('Spielnummer')

        # self.PlotTrainingAx.set_title('Züge pro Spiel während Training')
        # self.PlotTrainingAx.set_ylabel('Anzahl Züge')

        self.PlotTrainingAx.set_title('MSE der Q-Wert Änderungen')
        self.PlotTrainingAx.set_ylabel('MSE')
        
        # Creates a layout, so the plot can be added to the widget
        layout = QVBoxLayout(self.frmPlotTraining, margin=0)

        # Creates the FigureCanvas which contains the plot
        self.TrainingPlotCanvas = FigureCanvas(fig)
                
        # Add the plot to the GUI
        layout.addWidget(self.TrainingPlotCanvas)
        
    def UpdateTrainingPlot(self, AllStepsPerGame):
        """
        Updates the training plot
        """
        self.ResetTrainingPlot()
        self.PlotTrainingAx.bar(range(1, len(AllStepsPerGame) + 1), AllStepsPerGame, color="blue")
        
        # Draw the plot:
        self.TrainingPlotCanvas.draw()        
        
    
    def ResetTrainingPlot(self):
        """
        Reset the training plot
        """
        self.PlotTrainingAx.clear()
        self.PlotTrainingAx.set_xlabel('Spielnummer')
        
        # self.PlotTrainingAx.set_title('Züge pro Spiel während Training')
        # self.PlotTrainingAx.set_ylabel('Anzahl Züge')
                
        self.PlotTrainingAx.set_title('MSE der Q-Wert Änderungen')
        self.PlotTrainingAx.set_ylabel('MSE')
     
    """---------------------------------------------------------------------"""
    """ Functions for the Validation: """
    """---------------------------------------------------------------------"""
    def StartValidation(self):
        """
        Function to start validation.
        """
                
        # Get Current Validation parameter:
        CurrentValidationParameter = self.GetCurrentValidationParameter()    
                        
        # checks if a model is laoded:
        if self.FilePathValidationModel == "":
            self.ShowError([0, "Kein Validierungsmodell geladen!"])   
            return   
        
        # checks if a valide configuration for the validation is set:
        if not CurrentValidationParameter.RandomReferenceModel \
            and not CurrentValidationParameter.ReferenceMovesRandom \
            and self.FilePathReferenceModel == "":
             
            self.ShowError([0, "Ungültige Einstellungen! Es muss entweder Zufällige Züge oder Modell verwendet werden, oder ein Referenzmodell muss geladen sein."])   
            return   
                
        # Deactivate the GUI elements:
        self.SetValidationElementsEnabled(False)
        
        # Save the time of the start moment and start timer:
        self.TimeStartValidation = QDateTime.currentDateTime()
        self.TimerValidation.start(1000)
        
        # Reset the current values:
        self.UpdateValidationTimer()
        self.SetCurrentNumberOfValidationGames(1)
        
        # Reset the validation plot:
        self.ResetValidationPlot()
        
        self.ValidationIsRunning = True
        
        # Put the Event on the queue:
        self.DataQueOut.put(["StartValidation", CurrentValidationParameter])
        
    def StopValidation(self):
        """
        Stop the Validation.
        """
        # Stop the timer:
        self.TimerValidation.stop()
            
        # Activate the GUI elements:
        self.SetValidationElementsEnabled(True)
        
        # 
        self.ValidationIsRunning = False
        
        # Put the stop event on the queue:
        self.DataQueOut.put(["StopValidation"]) 
    
    def GetCurrentValidationParameter(self):
        """
        Reads the current Validation parameter and returns the Struct with all
        parameter
        """
        # 
        CurrentPara = Validationparameter()
        
        # Fill the struct with all current Validation parameters:        
        CurrentPara.NumberGames = self.sBoxValidationNumberGames.value()
        CurrentPara.MaxMoves = self.sBoxValidationMaxMoves.value()
        CurrentPara.RandomReferenceModel = self.cBoxValidationRandomReferenceModel.isChecked()
        CurrentPara.FirstMoveRandom = self.cBoxValidationFirstMoveRandom.isChecked()
        CurrentPara.ReferenceMovesRandom = self.cBoxValidationReferenceMovesRandom.isChecked()
        CurrentPara.SwitchPlayer = self.cBoxValidationSwitchPlayer.isChecked()
        
        CurrentPara.NumValidationInstances = self.sBoxValidationNumValidationInstances.value()


        return CurrentPara
     
    def SetCurrentValidationParameter(self, NewPara):
        """
        Sets new Validation parameter in the GUI based on the given data
        """   
        self.sBoxValidationNumberGames.setValue(NewPara.NumberGames)
        self.sBoxValidationMaxMoves.setValue(NewPara.MaxMoves)
        self.cBoxValidationRandomReferenceModel.setChecked(NewPara.RandomReferenceModel)
        self.cBoxValidationFirstMoveRandom.setChecked(NewPara.FirstMoveRandom)
        self.cBoxValidationReferenceMovesRandom.setChecked(NewPara.ReferenceMovesRandom)
        self.cBoxValidationSwitchPlayer.setChecked(NewPara.SwitchPlayer)

        self.sBoxValidationNumValidationInstances.setValue(NewPara.NumValidationInstances)
        
    def LoadValidationParameter(self):
        """
        Loads Validation parameter from a selected file
        """
        # Open a File-Dialog to save the model:
        FilePath = QFileDialog().getOpenFileName(self, caption="Validierungsparameter laden", filter=self.FiletypesValidationparameter)
        
        
        # If an file was selected:
        if FilePath[0] != "":
            # Reset String of the loaded file path:
            DisplayTxt = f"<html><head/><body><p><span style=\"font-size:10pt;\">Geladen:</span></p></body></html>"
            self.lblTxtLoadedValidationParameter.setText(DisplayTxt)
            try:
                with open(FilePath[0], "rb") as file:
                    ValidationParameter = pickle.load(file)
                self.SetCurrentValidationParameter(ValidationParameter)
                
                # Create String to show the opened file path:
                DisplayTxt = f"<html><head/><body><p><span style=\"font-size:10pt;\">Geladen: {FilePath[0]}</span></p></body></html>"
                self.lblTxtLoadedValidationParameter.setText(DisplayTxt)
            except:            
                # Show error:    
                self.ShowError(["", "Validierungsparameter konnten nicht geladen werden"])
    
    def SaveValidationParameter(self):
        """
        Save the current Validation parameter to a file
        """
        # Open a File-Dialog to save the parameters:
        FilePath = QFileDialog().getSaveFileName(self, caption="Validierungsparameter speichern", filter=self.FiletypesValidationparameter)
        
        CurrentValidaParameter = self.GetCurrentValidationParameter()
        
        # If a file should be saved:
        if FilePath[0] != "":
            # Save settings in file:
            with open(FilePath[0], "wb") as file:
                pickle.dump(CurrentValidaParameter, file)
                    
    def LoadValidationModel(self):
        """
        Function to load a model, so it can be validated
        """
        # Open a File-Dialog to select the folder with the model data:
        FilePath = QFileDialog().getExistingDirectory(self, caption="Ordner mit gespeicherten Modellen auswählen")
        
        # If a file was selected:
        if FilePath != "":
            # Set the cursor to wait:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            # Put the Filepath on the queue, so the DQN can laod the model:
            self.DataQueOut.put(["LoadValidationModel", FilePath])
 
    def LoadReferenceModel(self):
        """
        Function to load a model, so it can be used as refernce
        """
        # Open a File-Dialog to select the folder with the model data:
        FilePath = QFileDialog().getExistingDirectory(self, caption="Ordner mit gespeicherten Modellen auswählen")
        
        # If a file was selected:
        if FilePath != "":
            # Set the cursor to wait:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            # Put the Filepath on the queue, so the DQN can laod the model:
            self.DataQueOut.put(["LoadReferenceModel", FilePath])
    
    def LoadValidationModel_Success(self, data):
        """
        Is called, if a model was sucessfull saved
        """
        FilePath = data[1]
        
        # Save the path for start checking:
        self.FilePathValidationModel = FilePath
        
        # Set String of the loaded file path:
        DisplayTxt = f"<html><head/><body><p><span style=\"font-size:10pt;\">Geladen: {FilePath}</span></p></body></html>"
        self.lblTxtLoadedValidationModel.setText(DisplayTxt)
        
        QApplication.restoreOverrideCursor()
        
    def LoadReferenceModel_Success(self, data):
        """
        Is called, if a model was sucessfull saved
        """
        FilePath = data[1]
        
        # Save the path for start checking:
        self.FilePathReferenceModel = FilePath
        
        # Set String of the loaded file path:
        DisplayTxt = f"<html><head/><body><p><span style=\"font-size:10pt;\">Geladen: {FilePath}</span></p></body></html>"
        self.lblTxtLoadedReferenceModel.setText(DisplayTxt)
        
        QApplication.restoreOverrideCursor()
    
    def ValidationRunning(self, data):
        """
        Is called, when the validation is running and new data are available from the queue
        """
        numberOfGames, wins, losses, draws = data[1], data[2], data[3], data[4]
        
        # Update plot first, when min one value is not equal 0
        if wins != 0 or losses != 0 or draws != 0:
            self.UpdateValidationPlot(wins, losses, draws)
            
        self.SetCurrentNumberOfValidationGames(numberOfGames)
        
        # self.UpdateValidationPlot(numberOfGames)
        
     
    def ValidationFinished(self, data):
        """
        Function when the validation is finished
        """
        # Plot the diagram:
        wins, losses, draws = data[1], data[2], data[3]
        self.UpdateValidationPlot(wins, losses, draws)
        
        # Stop the timer:
        self.TimerValidation.stop()
            
        # Activate the GUI elements:
        self.SetValidationElementsEnabled(True)
        
        # Sets progress bar 100 % (So it looks nice):
        MaxNumberGames = self.sBoxValidationNumberGames.value()
        self.SetCurrentNumberOfValidationGames(MaxNumberGames)
        
        # Deactivate Validation:
        self.ValidationIsRunning = False
        
        
    def SetValidationElementsEnabled(self, enable):
        """
        Enables or disable the GUI Elements for the  Validation.
        Used to deactivate the elements while  Validation is running
        """        
        # Enable/Disable changing of  Validation parameter settings:
        self.btnValidationStart.setEnabled(enable)
        self.btnValidationStop.setEnabled(not enable)
        
        self.btnLoadValidationParameter.setEnabled(enable)
    
        self.btnLoadValidationModel.setEnabled(enable)
        self.btnLoadReferenceModel.setEnabled(enable)
        
        self.sBoxValidationNumberGames.setEnabled(enable)        
        self.sBoxValidationMaxMoves.setEnabled(enable)
        self.cBoxValidationRandomReferenceModel.setEnabled(enable)
        self.cBoxValidationFirstMoveRandom.setEnabled(enable)
        self.cBoxValidationReferenceMovesRandom.setEnabled(enable)
        self.cBoxValidationSwitchPlayer.setEnabled(enable)

        self.sBoxValidationNumValidationInstances.setEnabled(enable)
    
    def UpdateValidationTimer(self):
        """
        Updates the timer since when the validation is running.
        Is called by the timer event
        """
        # Calculate time difference since the start of the Validation:
        DeltaTime = QDateTime.currentDateTime().toPyDateTime().replace(microsecond=0) \
            - self.TimeStartValidation.toPyDateTime().replace(microsecond=0)
        
        # Create String to show text:
        DisplayTxt = f"<html><head/><body><p><span style=\"font-size:12pt;\">Validierung läuft seit {DeltaTime} hh:mm:ss</span></p></body></html>"
        
        # Set the text in the GUI:
        self.lblTxtValidationSince.setText(DisplayTxt)
     
    def SetCurrentNumberOfValidationGames(self, numberOfGames):
        """
        Set the text and the progress bar for the current number of games of the validation.
        """
        MaxNumberGames = self.sBoxValidationNumberGames.value()
        
        # Create String to show text:
        DisplayTxtGames = f"<html><head/><body><p><span style=\"font-size:12pt;\">Spiel {numberOfGames:.0f} von {MaxNumberGames}</span></p></body></html>"
        
        # Set the text in the GUI:
        self.lblTxtValidationActualGames.setText(DisplayTxtGames)
        
        # Set the progress bar, value in procent:
        self.pbarValidation.setValue(numberOfGames / MaxNumberGames * 100)
        
    """---------------------------------------------------------------------"""
    """ Functions for validation graphics:"""
    """---------------------------------------------------------------------"""
    def InitValidationPlot(self):
        """
        Inits the Validation plot
        """
        fig = Figure()
        fig.set_facecolor("#d9d9d9")    # Set background color
        
        # Sets a border around the complete plot
        self.frmPlotValidation.setStyleSheet("border: 1px solid #d72305;")
        
        # Creats axis as class variable, so can be used later again
        self.PlotValidationAx = fig.add_subplot(111)
        self.PlotValidationAx.set_title('Aufteilung Siege/Niederlagen/Unentschieden')
        self.PlotValidationAx.pie(x=[1, 1, 1], 
                                  labels=["Siege", "Niederlagen", "Unentschieden"],
                                  autopct='%.0f%%')
        
        # Creates a layout, so the plot can be added to the widget
        layout = QVBoxLayout(self.frmPlotValidation, margin=0)

        # Creates the FigureCanvas which contains the plot
        self.ValidationPlotCanvas = FigureCanvas(fig)
                
        # Add the plot to the GUI
        layout.addWidget(self.ValidationPlotCanvas)
            
    def UpdateValidationPlot(self, wins, losses, draws):
        """
        Updates the Validation plot
        """
        self.PlotValidationAx.clear()
        self.PlotValidationAx.set_title('Aufteilung Siege/Niederlangen/Unentschieden')
        self.PlotValidationAx.pie(x=[wins, losses, draws], 
                                  labels=["Siege", "Niederlagen", "Unentschieden"], 
                                  autopct='%.0f%%')
        
        # Draw the plot:
        self.ValidationPlotCanvas.draw()        
        
    def ResetValidationPlot(self):
        """
        Reset the Validation plot
        """
        self.PlotValidationAx.clear()
        self.PlotValidationAx.set_title('Aufteilung Siege/Niederlangen/Unentschieden')
        self.PlotValidationAx.pie(x=[1, 1, 1], 
                                  labels=["Siege", "Niederlagen", "Unentschieden"],
                                  autopct='%.0f%%')
      
    """---------------------------------------------------------------------"""
    """ Functions to detect click events of the game field:"""
    """---------------------------------------------------------------------"""            
    def PositionClicked(self, indexRing: int, indexPos: int):
        """
        Function to evaluate a click on a game field position
        """
        #print(indexRing, indexPos)          
        self.DataQueOut.put(["GamePosClicked" ,indexRing, indexPos])
        
        
    """---------------------------------------------------------------------"""
    """ Mapping of single lable click events on one function:"""
    """---------------------------------------------------------------------"""                  
    def PositionClicked_lblSt00(self, event): 
        self.PositionClicked(0, 0)        
        
    def PositionClicked_lblSt01(self, event): 
        self.PositionClicked(0, 1)
        
    def PositionClicked_lblSt02(self, event): 
        self.PositionClicked(0, 2)   
        
    def PositionClicked_lblSt03(self, event): 
        self.PositionClicked(0, 3)  
        
    def PositionClicked_lblSt04(self, event): 
        self.PositionClicked(0, 4)   
        
    def PositionClicked_lblSt05(self, event): 
        self.PositionClicked(0, 5)   
        
    def PositionClicked_lblSt06(self, event): 
        self.PositionClicked(0, 6)   
        
    def PositionClicked_lblSt07(self, event): 
        self.PositionClicked(0, 7)
        
    def PositionClicked_lblSt10(self, event): 
        self.PositionClicked(1, 0)
        
    def PositionClicked_lblSt11(self, event): 
        self.PositionClicked(1, 1)
        
    def PositionClicked_lblSt12(self, event): 
        self.PositionClicked(1, 2)   
        
    def PositionClicked_lblSt13(self, event): 
        self.PositionClicked(1, 3)  
        
    def PositionClicked_lblSt14(self, event): 
        self.PositionClicked(1, 4)   
        
    def PositionClicked_lblSt15(self, event): 
        self.PositionClicked(1, 5)   
        
    def PositionClicked_lblSt16(self, event): 
        self.PositionClicked(1, 6)   
        
    def PositionClicked_lblSt17(self, event): 
        self.PositionClicked(1, 7)

    def PositionClicked_lblSt20(self, event): 
        self.PositionClicked(2, 0)
        
    def PositionClicked_lblSt21(self, event): 
        self.PositionClicked(2, 1)
        
    def PositionClicked_lblSt22(self, event): 
        self.PositionClicked(2, 2)   
        
    def PositionClicked_lblSt23(self, event): 
        self.PositionClicked(2, 3)  
        
    def PositionClicked_lblSt24(self, event): 
        self.PositionClicked(2, 4)   
        
    def PositionClicked_lblSt25(self, event): 
        self.PositionClicked(2, 5)   
        
    def PositionClicked_lblSt26(self, event): 
        self.PositionClicked(2, 6)   
        
    def PositionClicked_lblSt27(self, event): 
        self.PositionClicked(2, 7)

if __name__== '__main__':    
        
    # Logic only for testing
    from muehle_logik_Kopie15032023 import MillLogic
    # Muss später aus zentraller Stelle kommen:
    from muehle_logik_Kopie15032023 import enAction 
    
    GameMillLogic = MillLogic()
    
    lstExternFunctions = [GameMillLogic.restartGame, GameMillLogic.getState, GameMillLogic.setMove, GameMillLogic.getMovesFromSelectedToken, GameMillLogic.getInStockTokens]
    
    app = QApplication([])
    frm_main = Frm_main(lstExternFunctions)
    frm_main.show()
    app.exec()

