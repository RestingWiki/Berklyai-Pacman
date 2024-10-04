# search.py
# ---------
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

#TODO: remove nguyenpanda
"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import *
# from searchAgents import PositionSearchProblem
from nguyenpanda.swan import color
class Node:
    """
    A class to represent a node in the search problem.

    Attributes:
        prev (Node): The previous node in the path.
        current (tuple): A tuple (x, y) representing the current position of the node in the maze.
        direction (str): The direction taken to move from the previous node to the current node.
                         Possible values could be 'North', 'South', 'East', or 'West'.

    Example:
        prev_node = Node(None, (1, 1), None)
        
        current_node = Node(prev_node, (1, 2), 'North')
        
        the prev_node(1, 1) goes 'North' to reach current_node(1, 2)
    """
    def __init__(self, prev , current, direction):
        self.prev: Node = prev
        self.current = current  # (int, int) the position of a cell/neighbour in the maze
        self.direction  = direction # South | West |...
        
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    
    Args:
        problem (PositionSearchProblem): The search problem instance, for Q1 it is an instance of PositionSearchProblem

    Returns:
        list: A list of directions that lead to the goal ["North", "South", "East", "West"].
    """
    "*** YOUR CODE HERE ***"
    # TODO: Problem 1
    print (color["b"] + "Search function: Depth First Search")
    print (color["r"] + "Start:", problem.getStartState())
    print (color["r"] +"Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print (color["r"] + "Start's successors:", problem.getSuccessors(problem.getStartState()))
    
    
    # This basically just create a reversed LinkedList
    # class Node:
    #   self.prev = AnotherNode
    #   self.current = (x, y)
    #   self.direction = "North"
    #
    start_state = problem.getStartState()  # (int, int) 
    startNode = Node(prev=None, current=start_state, direction=None)
    neighbours = problem.getSuccessors(start_state)  
        
    # For the initial neighbour
    stack = Stack() # util.py
    stack.push(startNode)
    goal = None
    
    visited = set()  # Contains tupple of visited positions
    # Perform DFS
    while not stack.isEmpty():
        currentNode: Node = stack.pop()
        
        current_pos = currentNode.current # (int , int)
        
        if current_pos in visited:
            continue
        
        if problem.isGoalState(current_pos):
            goal = currentNode
            break
        
        visited.add(current_pos)
        
        neighbours = problem.getSuccessors(current_pos)
        for n in neighbours:
            pos, dir, _ = n # position, direction, cost(ignored)
            node = Node(current=pos, direction=dir, prev=currentNode)
            stack.push(node) 
        

    # Rebuilding the path
    path = []
    while goal.prev != None:
        path.append(goal.direction)
        goal= goal.prev
    path.reverse()
    
    # TODO: delete nguyenpdanda
    print(color.reset)
    print(color["g"])   
    return path 
    
    
def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    Args:
        problem (PositionSearchProblem): The search problem instance, for Q1 it is an instance of PositionSearchProblem

    Returns:
        list: A list of directions that lead to the goal ["North", "South", "East", "West"].
    """
    "*** YOUR CODE HERE ***"
    # TODO: Problem 2
    print (color["b"] + "Search function: Breadth First Search")
    print (color["r"] + "Start:", problem.getStartState())
    print (color["r"] +"Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print (color["r"] + "Start's successors:", problem.getSuccessors(problem.getStartState()))
    
    start_state = problem.getStartState()
    startNode = Node(None, start_state, None)
    neighbours = problem.getSuccessors(start_state)  
    
    # A custom made class
    # This basically just create a reversed LinkedList
    # class Node:
    #   self.prev = AnotherNode
    #   self.current = (x, y)
    #   self.direction = "North"
    #
    
    # For the initial neighbour
    queue = Queue() # <--------- replaced Stack with Queue
    queue.push(startNode)
    goal = None  
    
    visited = set()  # Contains tupple of visited positions on the board
    # Perform DFS
    while not queue.isEmpty():
        currentNode: Node = queue.pop()
        
        current_state = currentNode.current
        
        if current_state in visited:
            continue
        
        if problem.isGoalState(current_state):
            goal = currentNode
            break
        
        visited.add(current_state)
        
        neighbours = problem.getSuccessors(current_state)
        for n in neighbours:
            pos, dir, _ = n
            node = Node(current=pos, direction=dir, prev=currentNode)
            queue.push(node) 
        

    # Rebuilding the path
    path = []
    while goal.prev != None:
        path.append(goal.direction)
        goal= goal.prev
    path.reverse()
    print(color.reset)
    print(color["g"])   
    return path 

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
