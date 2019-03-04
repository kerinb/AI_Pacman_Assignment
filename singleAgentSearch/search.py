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
    stack = util.Stack()
    stack.push([(problem.getStartState(), "Stop", 0)])
    visited_nodes = []

    while not stack.isEmpty():
        path = stack.pop()
        current_state = path[-1][0]

        if problem.isGoalState(current_state):
            return [x[1] for x in path][1:]

        if current_state not in visited_nodes:
            visited_nodes.append(current_state)

            for next_node in problem.getSuccessors(current_state):
                next_node_path = path[:]
                next_node_path.append(next_node)
                stack.push(next_node_path)

    return False


def breadthFirstSearch(problem):
    queue = util.Queue()
    queue.push([(problem.getStartState(), "Stop", 0)])
    visited_nodes = []

    while not queue.isEmpty():
        path = queue.pop()
        current_state = path[-1][0]

        if problem.isGoalState(current_state):
            return [x[1] for x in path][1:]

        if current_state not in visited_nodes:
            visited_nodes.append(current_state)

            for next_node in problem.getSuccessors(current_state):
                next_node_path = path[:]
                next_node_path.append(next_node)
                queue.push(next_node_path)

    return False


def uniformCostSearch(problem):
    cost_function = lambda path: problem.getCostOfActions([x[1] for x in path][1:])
    queue = util.PriorityQueueWithFunction(cost_function)

    queue.push([(problem.getStartState(), "Stop", 0)])
    visited_nodes = []

    while not queue.isEmpty():
        path = queue.pop()
        current_state = path[-1][0]

        if problem.isGoalState(current_state):
            return [x[1] for x in path][1:]

        if current_state not in visited_nodes:
            visited_nodes.append(current_state)

            for next_node in problem.getSuccessors(current_state):
                next_node_path = path[:]
                next_node_path.append(next_node)
                queue.push(next_node_path)

    return False


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    cost_function = lambda path: problem.getCostOfActions([x[1] for x in path][1:]) + heuristic(path[-1][0], problem)
    queue = util.PriorityQueueWithFunction(cost_function)
    queue.push([(problem.getStartState(), "Stop", 0)])
    visited_nodes = []

    while not queue.isEmpty():
        path = queue.pop()
        current_state = path[-1][0]

        if problem.isGoalState(current_state):
            return [x[1] for x in path][1:]

        if current_state not in visited_nodes:
            visited_nodes.append(current_state)

            for next_node in problem.getSuccessors(current_state):
                next_node_path = path[:]
                next_node_path.append(next_node)
                queue.push(next_node_path)

    return False


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
