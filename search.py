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

class PriorityNode(Node):
    def __init__(self, prev, current, direction, cost=-1):
        super().__init__(prev, current, direction)
        self.cost: int = cost
        
class CornerProblemNode(Node):
    def __init__(self, prev, current, direction, visited_corners):
        super().__init__(prev, current, direction)
        self.visited_corners: set = visited_corners     
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
    ret_node = goal
    while goal.prev != None:
        path.append(goal.direction)
        goal= goal.prev
    path.reverse()
    
    # TODO: delete ret_node
    print(color.reset)
    print(color["g"])   
    return path, ret_node
    
    
def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    Args:
        problem (PositionSearchProblem): The search problem instance, for Q2 it is an instance of PositionSearchProblem

    Returns:
        list: A list of directions that lead to the goal ["North", "South", "East", "West"].
    """
    "*** YOUR CODE HERE ***"
    # TODO: Problem 2,5
    from searchAgents import CornersProblem
    from searchAgents import PositionSearchProblem
    import copy
    # I have no idea if defining a function within another function is a good idea or not
    # But type annotation helps alot when coding soooo 
    # Here we go
    def BFS_corner_problem(problem:CornersProblem):
        # Warning reduntdant code inbound!
        start_state = problem.getStartState()
        startNode = CornerProblemNode(None, start_state[0], None, None)
        
        queue = Queue()
        queue.push(startNode)
        goal = None
        visited = set()
        while not queue.isEmpty():
            currentNode:CornerProblemNode = queue.pop()
            
            current_pos, cur_visited_corners = currentNode.current, currentNode.visited_corners
            
            # Use BOTH the position AND the set of corners that pacman visited as a key to mark  
            if (current_pos, cur_visited_corners) in visited:
                continue
            
            if problem.isGoalState(cur_visited_corners):
                goal = currentNode
                return goal
            
            # Use deepcopy to be safe
            neighbours, new_visited_corners  = problem.getSuccessors(state=current_pos,
                                                                     visited_corner=copy.deepcopy(cur_visited_corners))
            for n in neighbours:
                pos, dir, _ = n
                node = CornerProblemNode(prev=currentNode,
                                         current=pos,
                                         direction=dir,
                                         visited_corners=copy.deepcopy(new_visited_corners)) 
                


        return goal
       
       
        
    if isinstance(problem, PositionSearchProblem):
        goal = BFS_position_problem(problem=problem)
    elif isinstance(problem, CornersProblem):
        goal = BFS_corner_problem(problem=problem)
    else:
        # This works for eightpuzzle.py but I had to that file a bit
        # Since I'm returning path, and a node for deubgging
        # eightpuzzle.py only need the path
        goal = BFS_position_problem(problem=problem)
        
    # Rebuilding the path
    path = []
    ret_node = goal
    while goal.prev != None:
        path.append(goal.direction)
        goal= goal.prev
    path.reverse()
    print(color.reset)
    print(color["g"])   
    # TODO: delete ret_node
    return path, ret_node 

def BFS_position_problem(problem):
    print (color["b"] + "Search function: Breadth First Search")
    print (color["r"] + "Start:", problem.getStartState())
    print (color["r"] +"Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print (color["r"] + "Start's successors:", problem.getSuccessors(problem.getStartState()))
    start_state = problem.getStartState()
    startNode = Node(None, start_state, None)
    
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
    # Perform BFS
    while not queue.isEmpty():
        currentNode: Node = queue.pop()
        
        current_pos = currentNode.current
        
        if current_pos in visited:
            continue
        
        if problem.isGoalState(current_pos):
            goal = currentNode
            break
        
        visited.add(current_pos)
        
        neighbours = problem.getSuccessors(current_pos)
        for n in neighbours:
            pos, dir, _ = n
            node = Node(current=pos, direction=dir, prev=currentNode)
            queue.push(node) 
        

    # # Rebuilding the path
    # path = []
    # ret_node = goal
    # while goal.prev != None:
    #     path.append(goal.direction)
    #     goal= goal.prev
    # path.reverse()
    # print(color.reset)
    # print(color["g"])   
    # # TODO: delete ret_node
    return goal

    
def uniformCostSearch(problem):
    "*** YOUR CODE HERE ***"
    # TODO: Problem 3
    print (color["r"] + "Search function: Uniform Cost Search")
    print (color["r"] + "Start:", problem.getStartState())
    print (color["r"] +"Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print (color["r"] + "Start's successors:", problem.getSuccessors(problem.getStartState()))
    
    start_state = problem.getStartState()
    startNode = PriorityNode(None, start_state, None)
    
    # A custom made class
    # This basically just create a reversed LinkedList
    # class Node:
    #   self.prev = AnotherNode
    #   self.current = (x, y)
    #   self.direction = "North"
    #
    
    # For the initial neighbour
    queue = PriorityQueue()
    queue.push(item=startNode, priority=startNode.cost)
    goal = None  
    
    visited = set()  # Contains tupple of visited positions on the board
    # Perform UCS
    while not queue.isEmpty():
        currentNode: PriorityNode = queue.pop()
        
        current_pos = currentNode.current
        
        if current_pos in visited:
            continue
        
        if problem.isGoalState(current_pos):
            goal = currentNode
            break
        
        visited.add(current_pos)
        
        neighbours = problem.getSuccessors(current_pos)
        for n in neighbours:
            pos, dir, cost = n  # <-------------- Cost is used here compared to BFS and DFS
            node = PriorityNode(current=pos, direction=dir, prev=currentNode,cost=cost)
            queue.push(item=node, priority=cost) 
        

    # Rebuilding the path
    path = []
    ret_node = goal
    while goal.prev != None:
        path.append(goal.direction)
        goal= goal.prev
    path.reverse()
    print(color.reset)
    print(color["g"])   
    # TODO: delete ret_node
    return path, ret_node 

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """
        Search the node that has the lowest combined cost and heuristic first.
        
    """
    "*** YOUR CODE HERE ***"
    
    # TODO: Problem 4
    print (color["r"] + "Search function: aStar Search")
    print (color["r"] + "Start:", problem.getStartState())
    print (color["r"] +"Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print (color["r"] + "Start's successors:", problem.getSuccessors(problem.getStartState()))
    
    from searchAgents import manhattanHeuristic
    from searchAgents import euclideanHeuristic
    
    if heuristic is manhattanHeuristic:
        # print(h_func(problem.getStartState(), problem))
        heuristic = euclideanHeuristic
        print("Manhattan")
    elif heuristic is euclideanHeuristic:
        print("Euclidian")  
        
    start_state = problem.getStartState()
    startNode = PriorityNode(None, start_state, None)
    
    # A custom made class
    # This basically just create a reversed LinkedList
    # class Node:
    #   self.prev = AnotherNode
    #   self.current = (x, y)
    #   self.direction = "North"
    #
    
    # For the initial neighbour
    queue = PriorityQueue()
    queue.push(item=startNode, priority=startNode.cost)
    goal = None  
    
    visited = set()  # Contains tupple of visited positions on the board
    # Perform A*
    while not queue.isEmpty():
        currentNode: PriorityNode = queue.pop()
        
        current_pos = currentNode.current
        
        if current_pos in visited:
            continue
        
        if problem.isGoalState(current_pos):
            goal = currentNode
            break
        
        visited.add(current_pos)
        
        neighbours = problem.getSuccessors(current_pos)
        for n in neighbours:
            pos, dir, cost = n  # <-------------- Cost is used here compared to BFS and DFS
            h_value = heuristic(pos, problem)
            node = PriorityNode(current=pos, direction=dir, prev=currentNode,cost=cost + h_value)
            queue.push(item=node, priority=cost) 
        

    # Rebuilding the path
    path = []
    ret_node = goal
    while goal.prev != None:
        path.append(goal.direction)
        goal= goal.prev
    path.reverse()
    print(color.reset)
    print(color["g"])   
    # TODO: delete ret_node
    return path, ret_node 

    


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
