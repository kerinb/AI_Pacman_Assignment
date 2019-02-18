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


from __future__ import print_function

"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


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
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    # initialise an empty stack where we can push/pop or paths from
    stack = util.Stack()

    # push the root onto the stack in the following format:
    # [(state, action, cost)]
    stack.push([(problem.getStartState(), "Stop", 0)])

    # initialise a list for the visited nodes as an empty list
    visited_nodes = []

    # while the stack is no empty; i.e. there are still elements to be searched and we haven't found a solution
    while not stack.isEmpty():
        # get the path returned by the stack
        path = stack.pop()
        # get the last element in the list
        current_state = path[-1][0]

        # if a solution is found
        if problem.isGoalState(current_state):
            # return the list of path elements in path; i.e. (4,5), (5,4), (5,3) .. etc
            return [x[1] for x in path][1:]

        # if the node we are currently on hasn't yet been visited
        if current_state not in visited_nodes:
            # add current node to the list
            visited_nodes.append(current_state)

            # for every other element the hasnt yet been visited that is connect to current_node...
            for next_node in problem.getSuccessors(current_state):
                # copy parent nodes path
                next_node_path = path[:]
                # append our nodes path to the path list
                next_node_path.append(next_node)
                # push the list onto the stack
                stack.push(next_node_path)

    return False


def breadthFirstSearch(problem):
    # initialise an empty queue where we can push/pop or paths from
    queue = util.Queue()

    # push the root onto the stack in the following format:
    # [(state, action, cost)]
    queue.push([(problem.getStartState(), "Stop", 0)])

    # initialise a list for the visited nodes as an empty list
    visited_nodes = []

    # while the stack is no empty; i.e. there are still elements to be searched and we haven't found a solution
    while not queue.isEmpty():
        # get the path returned by the stack
        path = queue.pop()
        # get the last element in the list
        current_state = path[-1][0]

        # if a solution is found
        if problem.isGoalState(current_state):
            # return the list of path elements in path; i.e. (4,5), (5,4), (5,3) .. etc
            return [x[1] for x in path][1:]

        # if the node we are currently on hasn't yet been visited
        if current_state not in visited_nodes:
            # add current node to the list
            visited_nodes.append(current_state)

            # for every other element the hasnt yet been visited that is connect to current_node...
            for next_node in problem.getSuccessors(current_state):
                # copy parent nodes path
                next_node_path = path[:]
                # append our nodes path to the path list
                next_node_path.append(next_node)
                # push the list onto the stack
                queue.push(next_node_path)

    return False


def uniformCostSearch(problem):
    # initialise an empty priority queue where we can push/pop or paths from
    cost_function = lambda path: problem.getCostOfActions([x[1] for x in path][1:])

    queue = util.PriorityQueueWithFunction(cost_function)

    # push the root onto the stack in the following format:
    # [(state, action, cost)]
    queue.push([(problem.getStartState(), "Stop", 0)])

    # initialise a list for the visited nodes as an empty list
    visited_nodes = []

    # while the stack is no empty; i.e. there are still elements to be searched and we haven't found a solution
    while not queue.isEmpty():
        # get the path returned by the stack
        path = queue.pop()
        # get the last element in the list
        current_state = path[-1][0]

        # if a solution is found
        if problem.isGoalState(current_state):
            # return the list of path elements in path; i.e. (4,5), (5,4), (5,3) .. etc
            return [x[1] for x in path][1:]

        # if the node we are currently on hasn't yet been visited
        if current_state not in visited_nodes:
            # add current node to the list
            visited_nodes.append(current_state)

            # for every other element the hasnt yet been visited that is connect to current_node...
            for next_node in problem.getSuccessors(current_state):
                # copy parent nodes path
                next_node_path = path[:]
                # append our nodes path to the path list
                next_node_path.append(next_node)
                # push the list onto the stack
                queue.push(next_node_path)

    return False


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    # initialise an empty priority queue where we can push/pop or paths from
    cost_function = lambda path: problem.getCostOfActions([x[1] for x in path][1:]) + heuristic(path[-1][0], problem)

    queue = util.PriorityQueueWithFunction(cost_function)

    # push the root onto the stack in the following format:
    # [(state, action, cost)]
    queue.push([(problem.getStartState(), "Stop", 0)])

    # initialise a list for the visited nodes as an empty list
    visited_nodes = []

    # while the stack is no empty; i.e. there are still elements to be searched and we haven't found a solution
    while not queue.isEmpty():
        # get the path returned by the stack
        path = queue.pop()
        # get the last element in the list
        current_state = path[-1][0]

        # if a solution is found
        if problem.isGoalState(current_state):
            # return the list of path elements in path; i.e. (4,5), (5,4), (5,3) .. etc
            return [x[1] for x in path][1:]

        # if the node we are currently on hasn't yet been visited
        if current_state not in visited_nodes:
            # add current node to the list
            visited_nodes.append(current_state)

            # for every other element the hasnt yet been visited that is connect to current_node...
            for next_node in problem.getSuccessors(current_state):
                # copy parent nodes path
                next_node_path = path[:]
                # append our nodes path to the path list
                next_node_path.append(next_node)
                # push the list onto the stack
                queue.push(next_node_path)

    return False


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
