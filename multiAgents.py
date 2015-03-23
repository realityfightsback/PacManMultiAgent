# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        print 'Evaluating position'
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        
        if(random.random() > .2):
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        else:
            bestIndices = [index for index in range(len(scores)) if scores[index] > -5000]
        # probably bad and can fail, but don't want to ever select a ghost position, might run out of options get array out of bounds
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        ghostPositions = [generatePossibleGhostMoves(ghostState.configuration.pos) for ghostState in newGhostStates]
        # flatten to single list
        ghostPositions = [item for sublist in ghostPositions for item in sublist]  # fuck. Google found this to flatten. Try to learn later
        

        # consider ghost possible moves
        minDistanceToFood = findNearestFood(successorGameState, newPos)
        

        "*** YOUR CODE HERE ***"
        # If we would move onto a ghost this move should be given the worst possible score
        
        # closer to food is better
        score = successorGameState.getScore() - minDistanceToFood
        
        if(newPos in ghostPositions):
            print 'Would overlap ghost and pacman!'
            score += -10000
        
        
        
#         print '    Looking at %s where Pacman would be at %s with minFoodDistance of %s' % (action, newPos, minDistanceToFood)        
#         print '        Returned score for move %s' % score                                                                           
        
        return score

def findNearestFood(successorGameState, pacManPosition):
    x, y = pacManPosition
    
    fx = 0
    fy = 0
    
    minDistance = 10000
    
    foodGrid = successorGameState.data.food
    
    while fx < foodGrid.width:
        while fy < foodGrid.height:
            if(foodGrid[fx][fy] or successorGameState.data._win):
                tempMin = manhattanDistance((fx, fy), (x, y))
                if(tempMin < minDistance):
                    minDistance = tempMin
            fy += 1
        fy = 0
        fx += 1
    
    return minDistance
# min(iterable)
    
    # expand out in a radius
    
    successorGameState.food

def generatePossibleGhostMoves(ghostPosition):
    t = []
    t.append(ghostPosition)
    
    t.append((ghostPosition[0] + 1, ghostPosition[1]))
    t.append((ghostPosition[0] - 1, ghostPosition[1]))
    t.append((ghostPosition[0], ghostPosition[1] + 1))
    t.append((ghostPosition[0], ghostPosition[1] - 1))

    return t

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
#         return self.maxValue(1, gameState)[1]
        return self.miniMax(gameState, gameState.getNumAgents() * self.depth, 0)[1]
        
    def miniMax(self, gameState, depth, agentIndex):
        if(depth == 0 or gameState.getLegalActions(agentIndex) == []):
            return (self.evaluationFunction(gameState), 'NA')
            
        if(agentIndex == 0):  # Maximizer
            bestValue = (-10000000, "N/A")

            for x in gameState.getLegalActions(0):
                # pass processing to next min node
                tempValue = self.miniMax(gameState.generateSuccessor(0, x), depth - 1, 1)
                
                if(tempValue[0] >= bestValue[0]):
                    bestValue = (tempValue[0], x)   
    
        else:  # Minimizer
            
            bestValue = (10000000, "N/A")

            numOfGhosts = gameState.getNumAgents() - 1

            if(numOfGhosts > agentIndex):  # More ghosts
                for x in gameState.getLegalActions(agentIndex):
                    tempValue = self.miniMax(gameState.generateSuccessor(agentIndex, x), depth - 1, agentIndex + 1)
                    if(tempValue[0] <= bestValue[0]):
                        bestValue = (tempValue[0], x)   
    
            else:  # Back to PacMan
                for x in gameState.getLegalActions(agentIndex):
                    tempValue = self.miniMax(gameState.generateSuccessor(agentIndex, x), depth - 1, 0)
                    if(tempValue[0] <= bestValue[0]):
                        bestValue = (tempValue[0], x)   
    
        return bestValue
        
    def maxValue(self, level, gameState):
        bestValue = (-999999999, 'None')
#         if(terminalTest(gameState)):
#             return self.evaluationFunction(gameState)
#       
        numOfAgents = gameState.getNumAgents()
        
        
        if (numOfAgents == 1):
            # Just pacman, maximize!
            for x in gameState.getLegalActions(0):
                tempValue = self.evaluationFunction(gameState)
                
                if(tempValue >= bestValue[0]):
                    bestValue = (tempValue, x)        
        else:
            # We got ghosts, do calculations
            
            if(gameState.getLegalActions(0) == []):
                return (self.evaluationFunction(gameState), "N/A")
            else:
                for x in gameState.getLegalActions(0):
                    self.printWithTab(level, 'Exploring MAX')
                    tempValue = self.minValue(level, 0, gameState.generateSuccessor(0, x))
                    if(tempValue[0] >= bestValue[0]):
                        bestValue = (tempValue[0], x)
        
        self.printWithTab(level, 'Found max value of ' + str(bestValue))
        
        return bestValue
            
    def minValue(self, level, index, gameState):
        bestValue = (999999999, 'None')
#         if(terminalTest(gameState)):
#             return self.evaluationFunction(gameState)
#       
        numOfGhosts = gameState.getNumAgents() - 1
        
        index = index + 1

        self.printWithTab(level, 'Exploring MIN')
        
        if(index == numOfGhosts):  # completing a ply
            level = level + 1 
            if(level == self.depth):  # Hit the terminal state (final ghost decision)
                bestValue = (self.evaluationFunction(gameState), 'N/A')
                # final ghost, final ply, generates an actual value
                    
            else:  # more plies to come
                for x in gameState.getLegalActions(index):
                    # reset for MAX
                    tempValue = self.maxValue(level, gameState.generateSuccessor(index, x))
                    if(tempValue[0] <= bestValue[0]):
                        bestValue = (tempValue[0], x)
        else:  # more ghosts to come
            for x in gameState.getLegalActions(index):
                    tempValue = self.minValue(level, index, gameState.generateSuccessor(index, x))
                    if(tempValue[0] <= bestValue[0]):
                        bestValue = (tempValue, x)
        
        self.printWithTab(level, 'Found min value of ' + str(bestValue))
        
        return bestValue        
        
    def printWithTab(self, level, string):
            tab = '  '
            result = ''
            x = 0
            while(x < level):
                result = result + tab 
                x = x + 1
            
            print tab + string + " at level " + str(level)
            
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.miniMaxWithAlphaBeta(gameState, gameState.getNumAgents() * self.depth, 0, -float("Inf"), float("Inf"))[1]
    
    def miniMaxWithAlphaBeta(self, gameState, depth, agentIndex, alpha, beta):
        if(depth == 0 or gameState.getLegalActions(agentIndex) == []):
            return (self.evaluationFunction(gameState), 'NA')
            
        if(agentIndex == 0):  # Maximizer
            bestValue = (-float("inf"), "N/A")

            for x in gameState.getLegalActions(0):
                # pass processing to next min node
                tempValue = self.miniMaxWithAlphaBeta(gameState.generateSuccessor(0, x), depth - 1, 1, alpha, beta)
                
                if(tempValue[0] >= bestValue[0]):
                    bestValue = (tempValue[0], x)   
    
                if(tempValue[0] > beta):
                    return bestValue
                
                if(tempValue[0] > alpha):
                    alpha = tempValue[0]
                
        else:  # Minimizer
            
            bestValue = (float("inf"), "N/A")

            numOfGhosts = gameState.getNumAgents() - 1

            for x in gameState.getLegalActions(agentIndex):
                if(numOfGhosts > agentIndex):  # More ghosts
                    tempValue = self.miniMaxWithAlphaBeta(gameState.generateSuccessor(agentIndex, x), depth - 1, agentIndex + 1, alpha, beta)
                else:
                    tempValue = self.miniMaxWithAlphaBeta(gameState.generateSuccessor(agentIndex, x), depth - 1, 0, alpha, beta)
                
                if(tempValue[0] <= bestValue[0]):
                    bestValue = (tempValue[0], x)
                    
                if(tempValue[0] < alpha):
                    return bestValue
    
                if(tempValue[0] < beta):
                    beta = tempValue[0]    
                    
        return bestValue

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        return self.expectiMax(gameState, gameState.getNumAgents() * self.depth, 0)[1]
    

    def expectiMax(self, gameState, depth, agentIndex):
        if(depth == 0 or gameState.getLegalActions(agentIndex) == []):
            return (self.evaluationFunction(gameState), 'NA')
            
        if(agentIndex == 0):  # Maximizer
            bestValue = (-float("inf"), "N/A")

            for x in gameState.getLegalActions(0):
                # pass processing to next min node
                tempValue = self.expectiMax(gameState.generateSuccessor(0, x), depth - 1, 1)
                
                if(tempValue[0] >= bestValue[0]):
                    bestValue = (tempValue[0], x)   
                
        else:  # Expected Value
            numOfGhosts = gameState.getNumAgents() - 1

            numOfGameStates = float(len(gameState.getLegalActions(agentIndex)))
            runningTotal = 0
            
            for x in gameState.getLegalActions(agentIndex):
                if(numOfGhosts > agentIndex):  # More ghosts
                    tempValue = self.expectiMax(gameState.generateSuccessor(agentIndex, x), depth - 1, agentIndex + 1)
                else:
                    tempValue = self.expectiMax(gameState.generateSuccessor(agentIndex, x), depth - 1, 0)
                
                runningTotal = runningTotal + tempValue[0]
                
            return (runningTotal/numOfGameStates, "N/A")
        return bestValue

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    closestFood = findNearestFood(currentGameState, currentGameState.data.agentStates[0].configuration.pos)
    
    return currentGameState.getScore() - closestFood
      

def findNearestGhostDistance(currentGameState):
    
    closestGhostDistance = float("inf")
    
    firstRun = true
    for x in (currentGameState.data.agentStates):
        if(firstRun):#this is pacman
            firstRun = false
            continue
    
        temp = manhattanDistance(x.configuration.pos, currentGameState.data.agentStates[0].configuration.pos)

        if(temp < closestGhostDistance and x.scaredTimer > 0):
            closestGhostDistance = temp

    return closestGhostDistance
# Abbreviation
better = betterEvaluationFunction

