# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 20:02:29 2023

@author: Clemens
"""

"""
Notizen:
"""

# Add file paths of custom modules:
import sys
sys.path.append("GUI")
sys.path.append("MuehleLogik")
sys.path.append("DQN")

# GUI:
from PyQt5.QtWidgets import QApplication
import GUI.GUI_Klasse as GUI
from GUI.GUI_Klasse import *

# Mill logic:
from MuehleLogik.muehle_logik import MillLogic, enAction

# DQN: 
from DQN.DQNAgent_gpu import *

# Multiprocessing:
import multiprocessing


"""-------------------------------------------------------------------------"""
""" def Process1_GUI(DataQueIn, DataQueOut)"""
"""-------------------------------------------------------------------------""" 
def Process1_GUI(DataQueIn, DataQueOut):
    """
    Thread 1 for the GUI.
    Parameters
    ----------
    DataQueIn : Queue
        Queue for input data. Outside->GUI
    DataQueOut : Queue
        Queue for output data. Outside<-GUI
    """
    
    # Create the QApplication and the GUI   
    app = QApplication([])
    frm_main = GUI.Frm_main(DataQueIn, DataQueOut, enAction)
    # Show the GUI and start the QApplication
    frm_main.show()
    app.exec()  # Wait until GUI is closed
    
    # Signal to terminate the thread:
    # Will only be called after the GUI has been closed
    DataQueOut.put(["Terminate"])
    
  
"""-------------------------------------------------------------------------"""
""" class Process2_Logic()"""
"""-------------------------------------------------------------------------"""      
class Process2_Logic():
    """
    Class to handle to logic of the software. As class, so functions can better
    be used.
    """     
    
    def __init__(self, DataQueGUIIn, DataQueGUIOut, DataQueDQNIn, DataQueDQNOut):
        """
        Thread 2 for the logic.
        Parameters
        ----------
        DataQueGUIIn : Queue
            Queue for input data. GUI->Logic
        DataQueGUIOut : Queue
            Queue for output data. GUI<-Logic
        DataQueDQNIn : Queue
            Queue for input data. DQN->Logic
        DataQueDQNOut : Queue
            Queue for output data. DQN<-Logic
        """
            
        # Mill logic:
        self.GameMillLogic = MillLogic()
                
        # KI-Players:
        self.Player1AI = DQNPlayer(PLAYER1)
        self.Player2AI = DQNPlayer(PLAYER2)
        
        # Local variable to show possible Moves.
        self.bTokenClickedForShifting = False
        self.PosTokenClickedForShiftung = [0, 0] # Position of the Token which is clicked to be shifted
    
        # locale variables:
        self.TrainingIsRunning = False
        self.ValidationIsRunning = False
        
        # Queues for communication between threads/processes:
        self.DataQueGUIIn = DataQueGUIIn  
        self.DataQueGUIOut = DataQueGUIOut
        
        self.DataQueDQNIn = DataQueDQNIn  
        self.DataQueDQNOut = DataQueDQNOut
        
        # Init Setting:
        self.Settings = Settings()
        self.LoadSettings()
        
        # Start main function:
        self.run()
        
        
    """---------------------------------------------------------------------"""
    """ Locale functions:"""
    """---------------------------------------------------------------------"""             
    def ShowPossibleMoves(self, millField, indexRing, indexPos, PlayerNr):
        """
        Return a mill field with possible position marked for the shifting
        """
        # Get a list with all possible moves from this position:
        lstPossibleMoves = self.GameMillLogic.getMovesFromSelectedToken(indexRing, indexPos)
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


    def GamePosClicked(self, data):
        """
        Function to handle event of an clicked game position.
        """
        # Get position clicked in the GUI:
        indexRing = data[1]
        indexPos = data[2]
        
        # Get actual State from the game logic: 
        millField, nextAction = self.GameMillLogic.getState()

        # Change field, depending on next action and Player settings:
        # Set Tokens:
        if (nextAction == enAction.SetPlayer1) and self.Settings.Player1_Human:
            millField[indexRing][indexPos] = PLAYER1
            
        elif nextAction == enAction.SetPlayer2 and self.Settings.Player2_Human:
            millField[indexRing][indexPos] = PLAYER2   
            
        # Shift Tokens:    
        elif ((nextAction == enAction.ShiftPlayer1) and self.Settings.Player1_Human) or \
            ((nextAction == enAction.ShiftPlayer2) and self.Settings.Player2_Human):
            # Distinction of the Player:
            if (nextAction == enAction.ShiftPlayer1):
                Player = PLAYER1
            else:
                Player = PLAYER2
            
            # If an Token was already clicked:
            if self.bTokenClickedForShifting:
                # Shift Token from saved position to clicked position:
                millField[self.PosTokenClickedForShiftung[0]][self.PosTokenClickedForShiftung[1]] = LEER
                millField[indexRing][indexPos] = Player    
                
                # Hide possible moves -> need to hide possible position also if no shifting was taken
                millField = self.HidePossibleMoves(millField)
                self.bTokenClickedForShifting = False
            else:
                # If clicked position is owned by the player:
                if millField[indexRing][indexPos] == Player:
                    # Save Position:
                    self.PosTokenClickedForShiftung[0] = indexRing
                    self.PosTokenClickedForShiftung[1] = indexPos
                    self.bTokenClickedForShifting = True
                    # Show possible positions:
                    print(f"possible positions: {millField}, {indexRing}, {indexPos}, {Player}")
                    millField = self.ShowPossibleMoves(millField, indexRing, indexPos, Player)
            
        # Remove Tokens:
        elif ((nextAction == enAction.RemoveTokenPlayer1) and self.Settings.Player1_Human) or \
            ((nextAction == enAction.RemoveTokenPlayer2) and self.Settings.Player2_Human):
            millField[indexRing][indexPos] = LEER
        
        #print(f"millField: {millField}")
            
        # Changed mill field only to logic, if no possible moves are shown:
        if self.bTokenClickedForShifting == False:
            self.GameMillLogic.setMove(millField)   
            millField, nextAction = self.GameMillLogic.getState()
             
        # Get Token in stock:      
        InStockTokens = self.GameMillLogic.getInStockTokens()
        
        # Put Data in Queque for the GUI:
        self.DataQueGUIOut.put(["SetState" ,millField, nextAction, InStockTokens])  
  
        if self.bTokenClickedForShifting == False:
            self.AIPlayerHandler()
  
      
    def AIPlayerHandler(self):
        """
        Function handle the AI-Player
        """
        # Send only data, if the data was able/allowed to move
        AIMoved = False
        
        # Get the full state of the game:
        fullState = self.GameMillLogic.getFullState()
        actionType = fullState[1]
        
        # Make the next move:
        # Player 1:
        if (actionType.value % 2) == 0 and not self.Settings.Player1_Human:
            bValid, newField = self.Player1AI.getMove(fullState)
            #print(f"Spieler 1: {newField}")
            AIMoved = bValid
        # Player 2:
        elif (actionType.value % 2) and not self.Settings.Player2_Human:
            bValid, newField = self.Player2AI.getMove(fullState)
            #print(f"Spieler 2: {newField}")
            AIMoved = bValid
        
        # If AI was able to move:
        if AIMoved:
            self.GameMillLogic.setMove(newField)
            millField, nextAction = self.GameMillLogic.getState()
                 
            # Get Token in stock:      
            InStockTokens = self.GameMillLogic.getInStockTokens()
            
            # Put Data in Queque for the GUI:
            self.DataQueGUIOut.put(["SetState" ,millField, nextAction, InStockTokens])  
  
    
    def RestartGame(self, data):
        """
        Function to handle the game restart.
        """
        
        # Restart game logic:
        self.GameMillLogic.restartGame()
        
        # Get current state:
        millField, nextAction = self.GameMillLogic.getState()
        
        # Get Token in stock:      
        InStockTokens = self.GameMillLogic.getInStockTokens()                
        # Put Data in Queque for the GUI:
        self.DataQueGUIOut.put(["SetState", millField, nextAction, InStockTokens])  

    """---------------------------------------------------------------------"""
    """ Functions for settings: """
    """---------------------------------------------------------------------"""
    def SaveSettings(self, data):
        """
        Saves the settings of the players
        """
        # Read Settings from Queue:
        self.Settings = data[1]
        
        # Save settings in file:
        with open("settings.dat", "wb") as file:
            pickle.dump(self.Settings, file)
        
        # If KI-Player is selected, load the path:
        if self.Settings.Player1_Human == False and self.Settings.Player1_AI_Data_Path != "":
            self.LoadPlayerModel(self.Settings.Player1_AI_Data_Path, self.Player1AI)
        
        # If KI-Player is selected, load the path:
        if self.Settings.Player2_Human == False and self.Settings.Player2_AI_Data_Path != "":
            self.LoadPlayerModel(self.Settings.Player2_AI_Data_Path, self.Player2AI)
        
    def LoadSettings(self):
        """
        Try to load the settings from the settings file
        """
        try:
            with open("settings.dat", "rb") as file:
                self.Settings = pickle.load(file)
            # Send the settings to the GUI:    
            self.DataQueGUIOut.put(["LoadSettings", self.Settings])
                
            # If KI-Player is selected, load the path:
            if self.Settings.Player1_Human == False and self.Settings.Player1_AI_Data_Path != "":
                self.LoadPlayerModel(self.Settings.Player1_AI_Data_Path, self.Player1AI)
            
            # If KI-Player is selected, load the path:
            if self.Settings.Player2_Human == False and self.Settings.Player2_AI_Data_Path != "":
                self.LoadPlayerModel(self.Settings.Player2_AI_Data_Path, self.Player2AI)
                        
        except:            
            # Show error: (Pop-Up need to be created in the GUI)    
            self.DataQueGUIOut.put(["ShowError", "Einstellungen konnten nicht geladen werden!"])
    
    
    
    def LoadPlayerModel(self, FilePath, PlayerClass):
        """
        Function if a model should be loaded
        All mdels are stored in one folder
        """
        # Return value, if the loading was successfull
        bOk = True 
        
        # Try to load the models
        try:
            loaded_model1 = tf.keras.models.load_model(f"{FilePath}/model1.h5")
            loaded_model2 = tf.keras.models.load_model(f"{FilePath}/model2.h5")
            loaded_model3 = tf.keras.models.load_model(f"{FilePath}/model3.h5")
       
            Model = [loaded_model1, loaded_model2, loaded_model3]
                 
            # try to load the Model to the DQN: 
            if not PlayerClass.setModels(Model):              
                #self.DataQueOut.put(["LoadTrainingModel_Success", FilePath])
                self.DataQueGUIOut.put(["ShowError", "Modell konnte nicht geladen werden!"])
                bOk = False
        except:            
            # Show error: (Pop-Up needs to be created in the GUI)    
            self.DataQueGUIOut.put(["ShowError", "Modell konnte nicht geöffnet werden!"])
            bOk = False
        
        return bOk
    
    """---------------------------------------------------------------------"""
    """ Functions for the training: """
    """---------------------------------------------------------------------"""
    def StartTraining(self, data):
        """
        Function to start the training
        """
        self.TrainingIsRunning = True
        self.DataQueDQNOut.put(data)

    def StopTraining(self, data):
        """
        Function to stop the training
        """
        self.TrainingIsRunning = False
        self.DataQueDQNOut.put(data)
        
    def TrainingRunning(self, data):
        """
        Function to connect the current status of the training to the GUI
        """
        # If Output-queue is empty, put the new data in the output
        # Necessary, because GUI is slower than the training.
        if self.DataQueGUIOut.empty():
            self.DataQueGUIOut.put(data)
    
    def TrainingFinished(self, data):
        """
        Function to connect the finished event to the GUI.
        """
        self.TrainingIsRunning = False
        self.DataQueGUIOut.put(data)

    """---------------------------------------------------------------------"""
    """ Functions for the validation: """
    """---------------------------------------------------------------------"""
    def StartValidation(self, data):
        """
        Function to start the Validation
        """
        self.ValidationIsRunning = True
        self.DataQueDQNOut.put(data)

    def StopValidation(self, data):
        """
        Function to stop the Validation
        """
        self.ValidationIsRunning = False
        self.DataQueDQNOut.put(data)
        
    def ValidationRunning(self, data):
        """
        Function to connect the current status of the Validation to the GUI
        """
        if self.DataQueGUIOut.empty():
            self.DataQueGUIOut.put(data)
    
    def ValidationFinished(self, data):
        """
        Function to connect the finished event to the GUI.
        """
        self.ValidationIsRunning = False
        self.DataQueGUIOut.put(data)
        
    """---------------------------------------------------------------------"""
    """ Main loop:"""
    """---------------------------------------------------------------------""" 
    def run(self):
                
        while True:
            # Reset data:
            data = None
            
            # Try getting data from GUI queue:
            try: 
                # Timeout is used, so while training, the queue from the DQN can also be processed
                # Best way to reduce CPU load for the logic while training is running 
                # (because while training is running, the DQN send data throug the queue)
                data = self.DataQueGUIIn.get(timeout=0.01)
            except:
                pass
        
            # If data available:
            if data:
                print(f"Logic-GUI:{data}")
            
                # If process should be terminated:
                if data[0] == "Terminate":
                    # Signals other process the termination:
                    self.DataQueDQNOut.put(["Terminate"])
                    break
            
                # If data comes from game Position clicked:
                elif data[0] == "GamePosClicked":                    
                    self.GamePosClicked(data)
                    
                # If the game should be reseted:
                elif data[0] == "ResetGame":
                    self.RestartGame(data)
                    
                # If the training should be started: 
                elif data[0] == "StartTraining":
                    self.StartTraining(data)
                    
                # If the training should be stopped: 
                elif data[0] == "StopTraining":
                    self.StopTraining(data)
                    
                # Set The Settings of the Players:
                elif data[0] == "SaveSettings":
                    self.SaveSettings(data)
                
                # If a Model for the training should be laoded:
                elif data[0] == "LoadTrainingModel":
                    self.DataQueDQNOut.put(data)
                
                # If a Model for the training should be saved:
                elif data[0] == "SaveTrainingModel":
                    self.DataQueDQNOut.put(data)
                    
                # If model should be resetet:
                elif data[0] == "ResetTrainingModel":
                    self.DataQueDQNOut.put(data)
                    
                # If a AI move should be calculated:
                elif data[0] == "AIPlayerHandler":
                    self.AIPlayerHandler()
                
                # If the Validation should be started: 
                elif data[0] == "StartValidation":
                    self.StartValidation(data)
                    
                # If the Validation should be stopped: 
                elif data[0] == "StopValidation":
                    self.StopValidation(data)
                                    
                # If a Model for the Validation should be laoded:
                elif data[0] == "LoadValidationModel":
                    self.DataQueDQNOut.put(data)                        
                                                        
                # If a reference model for the validation should be laoded:
                elif data[0] == "LoadReferenceModel":
                    self.DataQueDQNOut.put(data)    
                
            # Reset data:
            data = None
            
            # Try getting data from DQN queue:
            try:            
                data = self.DataQueDQNIn.get_nowait()
            except:
                pass
            
            if data:
                print(f"Logic-DQN:{data}")
                
                # If the training is running:
                if data[0] == "TrainingRunning":
                    self.TrainingRunning(data)
                
                # If the training finished:
                elif data[0] == "TrainingFinished":
                    self.TrainingFinished(data)
                    
                # If an error occured: (Pop-Up needs to be shown in GUI)    
                elif data[0] == "ShowError":
                    self.DataQueGUIOut.put(data)
                    
                # If the model loaded successfully:    
                elif data[0] == "LoadTrainingModel_Success":
                    self.DataQueGUIOut.put(data)

                # If the Validation is running:
                if data[0] == "ValidationRunning":
                    self.ValidationRunning(data)
                
                # If the Validation finished:
                elif data[0] == "ValidationFinished":
                    self.ValidationFinished(data)
                    
                # If the model loaded successfully:    
                elif data[0] == "LoadValidationModel_Success":
                    self.DataQueGUIOut.put(data)                      
                    
                # If the model loaded successfully:    
                elif data[0] == "LoadReferenceModel_Success":
                    self.DataQueGUIOut.put(data) 

"""-------------------------------------------------------------------------"""
""" class Process3_DQN()"""
"""-------------------------------------------------------------------------"""                
class Process3_DQN():
    """
    Class to handle the Thread for the DQN.  As class, so functions can better
    be used.
    """
    
    def __init__(self, DataQueIn, DataQueOut):
        """
        Thread 2 for the logic.
        Parameters
        ----------
        DataQueIn : Queue
            Queue for input data. Outside->Logic
        DataQueGUIOut : Queue
            Queue for output data. Outside<-Logic
        """
         
        # Class for DQN Training:
        self.trainingDQN = DQNTraining()
        
        # Class for validation:
        self.ValidationDQN = DQNValidation()
        
        self.FilePathTrainingModel = ""    # File path of the model which should be trained
        self.FilePathValidationModel = ""    # File path of the model which should be validated
        self.FilePathReferenceModel = ""    # File path of the model which should be used as reference
        
        # Queues for communication between threads:
        self.DataQueIn = DataQueIn  
        self.DataQueOut = DataQueOut
        
        # Local variables:
        self.TrainingIsRunning = False
        self.TrainingGameDone = False
        self.TrainingActualNumberGames = 0
        
        self.ValidationIsRunning = False
        self.ValidationGameDone = False
        self.ValidationActualNumberGames = 0

        # Create Variable with the training parameter:
        self.TrainingsParameter = Trainingsparameter()

        # Create Variable with the training parameter:
        self.ValidationParameter = Validationparameter()
        
        # Start main function:
        self.run()
        
    
    """---------------------------------------------------------------------"""
    """ Functions for the training: """
    """---------------------------------------------------------------------"""
    def StartTraining(self, data):
        """
        Starts the training with the given parameters
        """        
        self.TrainingsParameter = data[1]
        
        self.TrainingActualNumberGames = 1
                
        # If no model is loaded:
        if self.FilePathTrainingModel == "":
            self.trainingDQN = DQNTraining(bCreateModels = True, 
                                           numInstances = self.TrainingsParameter.NumTrainingInstances,
                                           replayMemSize = self.TrainingsParameter.ReplayMemorySize,
                                           learningRate = self.TrainingsParameter.LearningRate)
        
        else:
            self.trainingDQN = DQNTraining(bCreateModels = False, 
                                           numInstances = self.TrainingsParameter.NumTrainingInstances,
                                           replayMemSize = self.TrainingsParameter.ReplayMemorySize,
                                           learningRate = self.TrainingsParameter.LearningRate)
            
            bOk = self.LoadTrainingModel([0, self.FilePathTrainingModel])
            
            # if loading was not successfull: (Error-message comes out of the loading function)
            if bOk == False:
                return
            
        self.trainingDQN.selectModelsToTrain(bSet = self.TrainingsParameter.TrainPhase1, 
                                             bShift = self.TrainingsParameter.TrainPhase2, 
                                             bJump = self.TrainingsParameter.TrainPhase3)
        
        self.trainingDQN.configureTraining(epsilonStart = self.TrainingsParameter.Epsilon, 
                                           epsilonDecay = self.TrainingsParameter.EpsilonDecay, 
                                           epsilonMin = self.TrainingsParameter.EpsilonMin, 
                                           maxSteps = self.TrainingsParameter.MaxMoves,
                                           discount = self.TrainingsParameter.Discount, 
                                           minReplayMemSize = self.TrainingsParameter.MinReplayMemorySize, 
                                           minibatchSize = self.TrainingsParameter.MiniBatchSize, 
                                           trainEveryXSteps = self.TrainingsParameter.TrainEveryXSteps, 
                                           updateTargetEvery = self.TrainingsParameter.UpdateTargetEvery)
        
        self.trainingDQN.initializeGame()
        
        self.TrainingIsRunning = True
        
    def StopTraining(self):
        """
        Stops the training
        """
        self.TrainingIsRunning = False
        #self.trainingDQN.finishGame()
        #self.TrainingFinished()
    
    def TrainingFinished(self):
        """
        Is called, when the training is finished.
        Signals other threads, that the training is finished.
        """
        self.TrainingIsRunning = False
        ep_rewards_1, ep_rewards_2, ep_steps = self.trainingDQN.getGameStats(bFull=True)
        self.DataQueOut.put(["TrainingFinished", ep_rewards_1, ep_rewards_2, ep_steps])
    
    def RunTraining(self):  
        """
        Handles the training.
        Every call, calculate one episode
        """
        # if there are still games left:
        if self.TrainingActualNumberGames < self.TrainingsParameter.NumberGames:
            finishedGames = self.trainingDQN.stepGame()
            self.TrainingActualNumberGames += finishedGames
            
            # get steps for the diagram:
            #_, _, ep_steps = self.trainingDQN.getGameStats(bFull=True)
            
            # Get the mean of the Weigths for the diagram:
            #weights_mean = trainingDQN.getModelWeights(bTotalMean=True)
            
            # Get the mse of the Q-values for the diagram:
            ep_mse = self.trainingDQN.getMeanSquaredErrors(bFull=True)
            
            
            self.DataQueOut.put(["TrainingRunning", self.TrainingActualNumberGames, ep_mse])
            
        # If no games left:
        else:
            self.TrainingFinished()    

    def ResetTrainingModel(self, data):
        """
        Resets the current training model
        """     
        # Reset File-path, so it signals, no model should be loaded
        self.FilePathTrainingModel = ""
                

    def LoadTrainingModel(self, data):
        """
        Function if a model should be loaded
        All mdels are stored in one folder
        """
        FilePath = data[1]
        
        # Return value, if the loading was successfull
        bOk = True
                
        try:
            loaded_model1 = tf.keras.models.load_model(f"{FilePath}/model1.h5")
            loaded_model2 = tf.keras.models.load_model(f"{FilePath}/model2.h5")
            loaded_model3 = tf.keras.models.load_model(f"{FilePath}/model3.h5")
    
            Model = (loaded_model1, loaded_model2, loaded_model3)
                        
            # try to load the Model to the DQN: 
            if self.trainingDQN.setModels(Model):
                # If successfull, save the Filepath and report sucess back to GUI
                self.FilePathTrainingModel = FilePath               
                self.DataQueOut.put(["LoadTrainingModel_Success", FilePath])
            else:
                self.DataQueOut.put(["ShowError", "Modell konnte nicht geladen werden!"])
                bOk = False
        except:            
            # Show error: (Pop-Up needs to be created in the GUI)    
            self.DataQueOut.put(["ShowError", "Modell konnte nicht geöffnet werden!"])
            bOk = False
        
        return bOk


    def SaveTrainingModel(self, data):
        """
        Function if a model should be loaded
        """
        FilePath = data[1]
        
        # Save path for the start:
        self.FilePathTrainingModel = FilePath
        
        # Get the model(s) from the DQN
        model1, model2, model3 = self.trainingDQN.getModels()
          
        # Save it to the Filepath:
        model1.save(f"{FilePath}/model1.h5")
        model2.save(f"{FilePath}/model2.h5")
        model3.save(f"{FilePath}/model3.h5")

    """---------------------------------------------------------------------"""
    """ Functions for the validation: """
    """---------------------------------------------------------------------"""        
    def StartValidation(self, data):
        """
        Starts the Validation with the given parameters
        """        
        self.ValidationParameter = data[1]
      
        self.ValidationActualNumberGames = 1
                
        # Init the validation class with the validation models:
        self.ValidationDQN = DQNValidation(bCreateRandomReference = self.ValidationParameter.RandomReferenceModel, 
                                       numInstances = self.ValidationParameter.NumValidationInstances)
        
        bOk = self.LoadValidationModel([0, self.FilePathValidationModel])
        
        # if loading was not sucessfull: (Error-message comes out of the loading function)
        if bOk == False:
            return
            
        # if the reference should be use random moves, load the reference model:
        if not self.ValidationParameter.ReferenceMovesRandom:
                  
            # If a loaded Reference model should be used:
            if not self.ValidationParameter.ReferenceMovesRandom:
                bOk = self.LoadReferenceModel([0, self.FilePathReferenceModel])
            
                # if loading was not sucessfull: (Error-message comes out of the loading function)
                if bOk == False:
                    return
  
        self.ValidationDQN.configureValidation(maxSteps=self.ValidationParameter.MaxMoves,
                                               bFirstMoveRandom=self.ValidationParameter.FirstMoveRandom,
                                               bRandomReferencePlayer=self.ValidationParameter.ReferenceMovesRandom)
        
        self.ValidationDQN.initializeGame()
        
        self.ValidationIsRunning = True
        
    def StopValidation(self):
        """
        Stops the Validation
        """
        self.ValidationIsRunning = False

    
    def ValidationFinished(self):
        """
        Is called, when the Validation is finished.
        Signals other threads, that the Validation is finished.
        """
        self.ValidationIsRunning = False
        #ep_rewards_1, ep_rewards_2, ep_steps = self.ValidationDQN.getGameStats(bFull=True)
        n_wins, n_losses, n_draws = self.ValidationDQN.getGameResults()
        self.DataQueOut.put(["ValidationFinished", n_wins, n_losses, n_draws])
    
    def RunValidation(self):  
        """
        Handles the Validation.
        Every call, calculate one episode
        """
        # if there are still games left:
        if self.ValidationActualNumberGames < self.ValidationParameter.NumberGames:
            finishedGames = self.ValidationDQN.stepGame(bSwitchPlayers=self.ValidationParameter.SwitchPlayer)
            self.ValidationActualNumberGames += finishedGames
            
            n_wins, n_losses, n_draws = self.ValidationDQN.getGameResults()
            
            self.DataQueOut.put(["ValidationRunning", self.ValidationActualNumberGames, n_wins, n_losses, n_draws])
        # If no games left:
        else:
            self.ValidationFinished()    

    def LoadValidationModel(self, data):
        """
        Function if a model should be loaded
        All models are stored in one folder
        """
        FilePath = data[1]
        
        # Return value, if the loading was successfull
        bOk = True
                
        try:
            loaded_model1 = tf.keras.models.load_model(f"{FilePath}/model1.h5")
            loaded_model2 = tf.keras.models.load_model(f"{FilePath}/model2.h5")
            loaded_model3 = tf.keras.models.load_model(f"{FilePath}/model3.h5")
    
            Model = (loaded_model1, loaded_model2, loaded_model3)
                        
            # try to load the Model to the DQN: 
            if self.ValidationDQN.setModels(agent=AGENT_TRAINED, models = Model):
                # If successfull, save the Filepath and report sucess back to GUI
                self.FilePathValidationModel = FilePath             
                self.ShowValidationModelWeights(Model)             
                self.DataQueOut.put(["LoadValidationModel_Success", FilePath])
            else:
                self.DataQueOut.put(["ShowError", "Modell konnte nicht geladen werden!"])
                bOk = False
        except:            
            # Show error: (Pop-Up needs to be created in the GUI)    
            self.DataQueOut.put(["ShowError", "Modell konnte nicht geöffnet werden!"])
            bOk = False
        
        return bOk       

    def LoadReferenceModel(self, data):
        """
        Function if a model should be loaded
        All models are stored in one folder
        """
        FilePath = data[1]
        
        # Return value, if the loading was successfull
        bOk = True
                
        #try:
        loaded_model1 = tf.keras.models.load_model(f"{FilePath}/model1.h5")
        loaded_model2 = tf.keras.models.load_model(f"{FilePath}/model2.h5")
        loaded_model3 = tf.keras.models.load_model(f"{FilePath}/model3.h5")

        Model = [loaded_model1, loaded_model2, loaded_model3]
                    
        # try to load the Model to the DQN: 
        try:
            if self.ValidationDQN.setModels(agent=AGENT_REFERENCE, models = Model):
                # If successfull, save the Filepath and report sucess back to GUI
                self.FilePathReferenceModel = FilePath    
                self.DataQueOut.put(["LoadReferenceModel_Success", FilePath])
            else:
                self.DataQueOut.put(["ShowError", "Modell konnte nicht geladen werden!"])
                bOk = False
        except:            
            # Show error: (Pop-Up needs to be created in the GUI)    
            self.DataQueOut.put(["ShowError", "Modell konnte nicht geöffnet werden!"])
            bOk = False
        
        return bOk  
    
    def ShowValidationModelWeights(self, Models):
        """
        Function to show some weights of the loaded vaildation model.
        At the moment simpel function, maybe later show also in GUI
        """
        
        # At the moment the data is not used in the othere processes
        # So this function is only used to show the data in the consol
        
        model1, model2, model3 = Models
        
        model1_weights = model1.get_weights()
        model2_weights = model2.get_weights()
        model3_weights = model3.get_weights()
        
        self.DataQueOut.put(["ShowValidationModelWeights", model1_weights[0], model2_weights[0], model3_weights[0]])
        
    
    """---------------------------------------------------------------------"""
    """ Main loop:"""
    """---------------------------------------------------------------------""" 
    def run(self):        
        while True:
            # If training or validation is running, don't wait until data are available:
            # This is used to reduce CPU load, when training is not running 
            if self.TrainingIsRunning or self.ValidationIsRunning:                
                data = None                
                # Try getting data from queue:
                try:            
                    data = self.DataQueIn.get_nowait()
                except:
                    pass
            else:
                # Wait until data available:
                data = self.DataQueIn.get()
            
            # If data available:
            if data:
                print(f"DQN:{data}")
                         
                # If process should be terminated:
                if data[0] == "Terminate":
                    break
            
                # If Training should be started:
                elif data[0] == "StartTraining":                    
                    self.StartTraining(data)
                    
                # If Training should be started:
                elif data[0] == "StopTraining":     
                    self.StopTraining()
                    
                # If a Model for the training should be laoded:
                elif data[0] == "LoadTrainingModel":
                    self.LoadTrainingModel(data)    
                    
                # If a Model for the training should be saved:
                elif data[0] == "SaveTrainingModel":
                    self.SaveTrainingModel(data)  
                    
                # If a model should be reseted:
                elif data[0] == "ResetTrainingModel":
                    self.ResetTrainingModel(data)
            
                # If Validation should be started:
                elif data[0] == "StartValidation":                    
                    self.StartValidation(data)
                    
                # If Validation should be started:
                elif data[0] == "StopValidation":     
                    self.StopValidation()
                    
                # If a Model for the Validation should be loaded:
                elif data[0] == "LoadValidationModel":
                    self.LoadValidationModel(data)    
                    
                # If a reference model should be loaded for the validation:
                elif data[0] == "LoadReferenceModel":
                    self.LoadReferenceModel(data)                     
       
            # If training is active:
            if self.TrainingIsRunning:
                self.RunTraining()
       
            # If validation is active:
            if self.ValidationIsRunning:
                self.RunValidation()
            

                    
def main():
    """
    Entry point of the program
    """
    
    # Create queues for communication between process:   
    DataQueGUIIn = multiprocessing.Queue()
    DataQueGUIOut = multiprocessing.Queue()

    DataQueDQIn = multiprocessing.Queue()
    DataQueDQNOut = multiprocessing.Queue()
    
    
    # Create process with the appropriate functions and pass queues:
    p2 = multiprocessing.Process(target=Process2_Logic, args=(DataQueGUIOut, DataQueGUIIn, DataQueDQNOut, DataQueDQIn, ), name="Logic")
    p3 = multiprocessing.Process(target=Process3_DQN, args=(DataQueDQIn, DataQueDQNOut, ), name="DQN")

    #p2.daemon = True

    # Start threads/processes: 
    p2.start()
    p3.start()
    
    # GUI need to be in the main process
    # Function returns, when GUI is closed
    Process1_GUI(DataQueGUIIn, DataQueGUIOut)    
    print("Ende1")
    # Wait until both processes are completely executed:
    p2.join()
    print("Ende2")
    p3.join()
    print("Ende3")
    

if __name__== '__main__':  
    """
    If started in IDE, start main()
    """
    main()
