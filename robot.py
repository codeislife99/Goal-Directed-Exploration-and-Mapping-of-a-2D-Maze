import numpy as np
from numpy import inf
import random
import time
import networkx as nx
import turtle
from heapq import heappush, heappop
from itertools import count


dim = 12
wn = turtle.Screen()
wn.bgcolor("white")        # set the window background color

subir = turtle.Turtle()    # Turtle for first run
ritwik = turtle.Turtle()   # Turtle for second run
ritwik.color("red")
ritwik.pensize(3)
subir.color("blue")
subir.pensize(1)


def bidirectional_dijkstra(G, source, target, weight=1):

    if source == target:
        return [source]
    push = heappush
    pop = heappop
    # Init:   Forward             Backward
    dists  = [{},                {}]  # dictionary of final distances
    paths  = [{source: [source]}, {target: [target]}]  # dictionary of paths
    fringe = [[],                []]  # heap of (distance, node) tuples for
                                      # extracting next node to expand
    seen   = [{source: 0},        {target: 0}]  # dictionary of distances to
                                                # nodes seen
    c = count()
    # initialize fringe heap
    push(fringe[0], (0, next(c), source))
    push(fringe[1], (0, next(c), target))
    # neighs for extracting correct neighbor information
    if G.is_directed():
        neighs = [G.successors_iter, G.predecessors_iter]
    else:
        neighs = [G.neighbors_iter, G.neighbors_iter]
    # variables to hold shortest discovered path
    #finaldist = 1e30000
    finalpath = []
    dir = 1
    while fringe[0] and fringe[1]:
        # choose direction
        # dir == 0 is forward direction and dir == 1 is back
        dir = 1 - dir
        # extract closest to expand
        (dist, _, v) = pop(fringe[dir])
        if v in dists[dir]:
            # Shortest path to v has already been found
            continue
        # update distance
        dists[dir][v] = dist  # equal to seen[dir][v]
        if v in dists[1 - dir]:
            # if we have scanned v in both directions we are done
            # we have now discovered the shortest path
            return finalpath

        for w in neighs[dir](v):
            if(dir == 0):  # forward
                if G.is_multigraph():
                    minweight = min((dd.get(weight, 1)
                                     for k, dd in G[v][w].items()))
                else:
                    minweight = G[v][w].get(weight, 1)
                vwLength = dists[dir][v] + minweight  # G[v][w].get(weight,1)
            else:  # back, must remember to change v,w->w,v
                if G.is_multigraph():
                    minweight = min((dd.get(weight, 1)
                                     for k, dd in G[w][v].items()))
                else:
                    minweight = G[w][v].get(weight, 1)
                vwLength = dists[dir][v] + minweight  # G[w][v].get(weight,1)

            if w in dists[dir]:
                if vwLength < dists[dir][w]:
                    raise ValueError(
                        "Contradictory paths found: negative weights?")
            elif w not in seen[dir] or vwLength < seen[dir][w]:
                # relaxing
                seen[dir][w] = vwLength
                push(fringe[dir], (vwLength, next(c), w))
                paths[dir][w] = paths[dir][v] + [w]
                if w in seen[0] and w in seen[1]:
                    # see if this path is better than than the already
                    # discovered shortest path
                    totaldist = seen[0][w] + seen[1][w]
                    if finalpath == [] or finaldist > totaldist:
                        finaldist = totaldist
                        revpath = paths[1][w][:]
                        revpath.reverse()
                        finalpath = paths[0][w] + revpath[1:]
    raise nx.NetworkXNoPath("No path between %s and %s." % (source, target))

class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)


def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)


def steer(heading, rotation):
    if rotation == 90:
        if heading == 'up':
            return 'right'
        if heading == 'down':
            return 'left'
        if heading == 'right':
            return 'down'
        if heading == 'left':
            return 'up'
    if rotation == -90:
        if heading == 'up':
            return 'left'
        if heading == 'down':
            return 'right'
        if heading == 'right':
            return 'up'
        if heading == 'left':
            return 'down'
    return heading


def move(heading, location, dist):
    location = list(location)
    if heading == 'up':
        location[0] = location[0] + dist
    if heading == 'down':
        location[0] = location[0] - dist
    if heading == 'right':
        location[1] = location[1] + dist
    if heading == 'left':
        location[1] = location[1] - dist
    return location


def isGoal(location, dim):
    return (dim / 2 - 1) <= location[0] and location[0] <= (dim / 2) and (dim / 2 - 1) <= location[1] and location[
                                                                                                              1] <= (
                                                                                                              dim / 2)


def hcost(location, dim):
    return np.amin([heuristic(location, [dim / 2, dim / 2]), heuristic(location, [dim / 2 - 1, dim / 2]),
                    heuristic(location, [dim / 2, dim / 2 - 1]), heuristic(location, [dim / 2 - 1, dim / 2 - 1])])


def traverse(current_loc, desired_loc, heading):
    if (current_loc[1] - desired_loc[1] < 0 and heading == 'left') or (
                current_loc[1] - desired_loc[1] > 0 and heading == 'right') or (
                current_loc[0] - desired_loc[0] < 0 and heading == 'down') or (
                current_loc[0] - desired_loc[0] > 0 and heading == 'up'):
        if heading == 'left' or heading == 'right':
            return 0, -abs(current_loc[1] - desired_loc[1])
        else:
            return 0, -abs(current_loc[0] - desired_loc[0])
    else:
        if current_loc[0] == desired_loc[0]:
            movement = abs(current_loc[1] - desired_loc[1])
            if heading == 'left' or heading == 'right':
                rotation = 0
            elif heading == 'up':
                if current_loc[1] - desired_loc[1] < 0:
                    rotation = 90
                else:
                    rotation = -90
            else:
                if current_loc[1] - desired_loc[1] < 0:
                    rotation = -90
                else:
                    rotation = 90
        if current_loc[1] == desired_loc[1]:
            movement = abs(current_loc[0] - desired_loc[0])
            if heading == 'up' or heading == 'down':
                rotation = 0
            elif heading == 'left':
                if current_loc[0] - desired_loc[0] < 0:
                    rotation = 90
                else:
                    rotation = -90
            else:
                if current_loc[0] - desired_loc[0] < 0:
                    rotation = -90
                else:
                    rotation = 90
    return rotation, movement


def addtograph(G, location, heading, sensors):
    loc = location[1] * dim + location[0]

    if heading == 'up':
        if sensors[0] > 0:
            for i in range(loc - dim, loc - dim * sensors[0] - 1, -dim):
                G.add_edge(i, loc)
                loc = i
        loc = location[1] * dim + location[0]
        if sensors[1] > 0:
            for i in range(loc + 1, loc + sensors[1] + 1, 1):
                G.add_edge(i, loc)
                loc = i
        loc = location[1] * dim + location[0]
        if sensors[2] > 0:
            for i in range(loc + dim, loc + dim * sensors[2] + 1, dim):
                G.add_edge(i, loc)
                loc = i
    elif heading == 'down':
        if sensors[0] > 0:
            for i in range(loc + dim, loc + dim * sensors[0] + 1, dim):
                G.add_edge(i, loc)
                loc = i
        loc = location[1] * dim + location[0]
        if sensors[1] > 0:
            for i in range(loc - 1, loc - sensors[1] - 1, -1):
                G.add_edge(i, loc)
                loc = i
        loc = location[1] * dim + location[0]
        if sensors[2] > 0:
            for i in range(loc - dim, loc - dim * sensors[2] - 1, -dim):
                G.add_edge(i, loc)
                loc = i
    elif heading == 'right':
        if sensors[0] > 0:
            for i in range(loc + 1, loc + sensors[0] + 1, 1):
                G.add_edge(i, loc)
                loc = i
        loc = location[1] * dim + location[0]
        if sensors[1] > 0:
            for i in range(loc + dim, loc + dim * sensors[1] + 1, dim):
                G.add_edge(i, loc)
                loc = i
        loc = location[1] * dim + location[0]
        if sensors[2] > 0:
            for i in range(loc - 1, loc - sensors[2] - 1, -1):
                G.add_edge(i, loc)
                loc = i
    else:
        if sensors[0] > 0:
            for i in range(loc - 1, loc - sensors[0] - 1, -1):
                G.add_edge(i, loc)
                loc = i
        loc = location[1] * dim + location[0]
        if sensors[1] > 0:
            for i in range(loc - dim, loc - dim * sensors[1] - 1, -dim):
                G.add_edge(i, loc)
                loc = i
        loc = location[1] * dim + location[0]
        if sensors[2] > 0:
            for i in range(loc + 1, loc + sensors[2] + 1, 1):
                G.add_edge(i, loc)
                loc = i


def getcoord(loc):
    return loc % dim, loc / dim


def travel(start, end, heading):
    current_loc = getcoord(start)
    desired_loc = getcoord(end)
    return traverse(current_loc, desired_loc, heading)


def astar_path(G, start, end):
    def dist(c, b):
        return abs(c / dim - b / dim) + abs((c / dim) % dim - (b / dim) % dim)

    return nx.astar_path(G, start, end, dist)


def cleanup(path):
    n = len(path)
    for i in range(0, n):
        if i < n - 3:
            if path[i + 1] == path[i] + 1:
                count = 0
                while path[i + 2] == path[i + 1] + 1 and count < 2:
                    count += 1
                    del path[i + 1]
                    n -= 1
            if path[i + 1] == path[i] + dim:
                count = 0
                while path[i + 2] == path[i + 1] + dim and count < 2:
                    count += 1
                    del path[i + 1]
                    n -= 1
            if path[i + 1] == path[i] - dim:
                count = 0
                while path[i + 2] == path[i + 1] - dim and count < 2:
                    count += 1
                    del path[i + 1]
                    n -= 1
            if path[i + 1] == path[i] - 1:
                count = 0
                while path[i + 2] == path[i + 1] - 1 and count < 2:
                    count += 1
                    del path[i + 1]
                    n -= 1


def goodneighbor(G, visited, location):
    currentloc = location[1] * dim + location[0]
    minimum = inf
    for neighbor in G.neighbors(currentloc):
        if minimum == visited[neighbor % dim][neighbor / dim]:
            if heurist_val > hcost([neighbor % dim, neighbor / dim], dim):
                minimum = visited[neighbor % dim][neighbor / dim]
                heurist_val = hcost([neighbor % dim, neighbor / dim], dim)
                optimumloc = neighbor
        if minimum > visited[neighbor % dim][neighbor / dim]:
            minimum = visited[neighbor % dim][neighbor / dim]
            heurist_val = hcost([neighbor % dim, neighbor / dim], dim)
            optimumloc = neighbor
    return getcoord(optimumloc)


def badneighbor(G, located, location):
    currentloc = location[1] * dim + location[0]
    minimum = inf
    for neighbor in G.neighbors(currentloc):
        if minimum == located[neighbor % dim][neighbor / dim]:
            if heurist_val < hcost([neighbor % dim, neighbor / dim], dim):
                minimum = located[neighbor % dim][neighbor / dim]
                heurist_val = hcost([neighbor % dim, neighbor / dim], dim)
                optimumloc = neighbor
        if minimum > located[neighbor % dim][neighbor / dim]:
            minimum = located[neighbor % dim][neighbor / dim]
            heurist_val = hcost([neighbor % dim, neighbor / dim], dim)
            optimumloc = neighbor
    return getcoord(optimumloc)


def cleanup_graph(G):
    for node in G.nodes():
        for neighbor in G.neighbors(node):
            for child in G.neighbors(neighbor):
                if neighbor == node+1 and child == neighbor+1:
                    G.add_edge(node, child)
                    for grandchild in G.neighbors(child):
                        if grandchild == child+1:
                            G.add_edge(node, grandchild)
                if neighbor == node+dim and child == neighbor+dim:
                    G.add_edge(node, child)
                    for grandchild in G.neighbors(child):
                        if grandchild == child+dim:
                            G.add_edge(node, grandchild)


def addtolocated(located, location, heading, sensors, metric):
    loc = location[1] * dim + location[0]

    if heading == 'up':
        if sensors[0] > 0:
            for i in range(loc - dim, loc - dim * sensors[0] - 1, -dim):
                x, y = getcoord(i)
                if located[x][y] == 0:
                    located[x][y] += metric
        if sensors[1] > 0:
            for i in range(loc + 1, loc + sensors[1] + 1, 1):
                x, y = getcoord(i)
                if located[x][y] == 0:
                    located[x][y] += metric
        if sensors[2] > 0:
            for i in range(loc + dim, loc + dim * sensors[2] + 1, dim):
                x, y = getcoord(i)
                if located[x][y] == 0:
                    located[x][y] += metric
    elif heading == 'down':
        if sensors[0] > 0:
            for i in range(loc + dim, loc + dim * sensors[0] + 1, dim):
                x, y = getcoord(i)
                if located[x][y] == 0:
                    located[x][y] += metric
        if sensors[1] > 0:
            for i in range(loc - 1, loc - sensors[1] - 1, -1):
                x, y = getcoord(i)
                if located[x][y] == 0:
                    located[x][y] += metric
        if sensors[2] > 0:
            for i in range(loc - dim, loc - dim * sensors[2] - 1, -dim):
                x, y = getcoord(i)
                if located[x][y] == 0:
                    located[x][y] += metric
    elif heading == 'right':
        if sensors[0] > 0:
            for i in range(loc + 1, loc + sensors[0] + 1, 1):
                x, y = getcoord(i)
                if located[x][y] == 0:
                    located[x][y] += metric
        if sensors[1] > 0:
            for i in range(loc + dim, loc + dim * sensors[1] + 1, dim):
                x, y = getcoord(i)
                if located[x][y] == 0:
                    located[x][y] += metric
        if sensors[2] > 0:
            for i in range(loc - 1, loc - sensors[2] - 1, -1):
                x, y = getcoord(i)
                if located[x][y] == 0:
                    located[x][y] += metric
    else:
        if sensors[0] > 0:
            for i in range(loc - 1, loc - sensors[0] - 1, -1):
                x, y = getcoord(i)
                if located[x][y] == 0:
                    located[x][y] += metric
        if sensors[1] > 0:
            for i in range(loc - dim, loc - dim * sensors[1] - 1, -dim):
                x, y = getcoord(i)
                if located[x][y] == 0:
                    located[x][y] += metric
        if sensors[2] > 0:
            for i in range(loc + 1, loc + sensors[2] + 1, 1):
                x, y = getcoord(i)
                if located[x][y] == 0:
                    located[x][y] += metric


def showcoverage(located):
    coverage = 0
    for i in range(dim):
        for j in range(dim):
            if located[i][j] == 0.5:
                coverage += 0.5
            if located[i][j] >= 1:
                coverage += 1
    return (coverage*100.0)/(dim*dim)


def backtracker(s, heading, vis2, G):
    currentloc = s.peek()
    vis2[currentloc] = 1
    heurist_val = inf
    flag = 0
    for neighbor in G.neighbors(currentloc):
        if vis2[neighbor] == 0:
            flag = 1
            if heurist_val > hcost([neighbor % dim, neighbor / dim], dim):
                desiredloc = neighbor
                heurist_val = hcost([neighbor % dim, neighbor / dim], dim)
    for neighbor in G.neighbors(currentloc):
        if vis2[neighbor] == 1 and flag == 0:
            flag = 1
            s.pop()
            desiredloc = s.peek()
            vis2[currentloc] = -1
    print vis2
    return travel(currentloc, desiredloc, heading)


def badbacktracker(s, heading, vis2, G):
    currentloc = s.peek()
    vis2[currentloc] = 1
    heurist_val = -inf
    flag = 0
    for neighbor in G.neighbors(currentloc):
        if vis2[neighbor] == 0:
            flag = 1
            if heurist_val < hcost([neighbor % dim, neighbor / dim], dim):
                desiredloc = neighbor
                heurist_val = hcost([neighbor % dim, neighbor / dim], dim)
    for neighbor in G.neighbors(currentloc):
        if vis2[neighbor] == 1 and flag == 0:
            flag = 1
            s.pop()
            desiredloc = s.peek()
            vis2[currentloc] = -1
    print vis2
    return travel(currentloc, desiredloc, heading)


def locbacktracker(s, heading, vis2, G, located):
    currentloc = s.peek()
    vis2[currentloc] = 1
    heurist_val = -inf
    flag = 0
    minimum = inf
    for neighbor in G.neighbors(currentloc):
        if vis2[neighbor] == 0:
            flag = 1
            if minimum == located[neighbor % dim][neighbor / dim]:
                if heurist_val < hcost([neighbor % dim, neighbor / dim], dim):
                    minimum = located[neighbor % dim][neighbor / dim]
                    heurist_val = hcost([neighbor % dim, neighbor / dim], dim)
                    desiredloc = neighbor
            if minimum > located[neighbor % dim][neighbor / dim]:
                minimum = located[neighbor % dim][neighbor / dim]
                heurist_val = hcost([neighbor % dim, neighbor / dim], dim)
                desiredloc = neighbor
    for neighbor in G.neighbors(currentloc):
        if vis2[neighbor] == 1 and flag == 0:
            flag = 1
            s.pop()
            desiredloc = s.peek()
            vis2[currentloc] = -1
    print vis2
    return travel(currentloc, desiredloc, heading)


class Robot(object):
    def __init__(self, maze_dim):
        self.location = [0, 0]
        self.heading = 'up'
        self.maze_dim = maze_dim
        global dim
        dim = maze_dim
        self.flag = 0
        self.visited = np.zeros((self.maze_dim, self.maze_dim))
        self.located = np.zeros((self.maze_dim, self.maze_dim))
        self.vis2 = np.zeros(self.maze_dim*self.maze_dim)
        self.visited[0][0] = 1
        self.located[0][0] = 1
        self.goal_loc = [0, 0]
        self.count = 1
        G = nx.Graph()
        for i in range(0, dim * dim):
            G.add_node(i)
        self.graph = G
        self.path = []
        self.gn1 = 1        # 1-> Implements Good Neighbor Algorithm for Pre-GoalReach Stint
                            # 0-> Implements Backtracking Algorithm for Pre-GoalReach Stint
        self.gn2 = 0        # 1-> Implements Good Neighbor Algorithm for Post-GoalReach Stint
                            # 0-> Implements Backtracking Algorithm for Post-GoalReach Stint
        self.s = Stack()
        subir.penup()
        subir.setpos(-10*dim+10, -10*dim+10)
        subir.pendown()
        subir.left(90)
        ritwik.penup()
        ritwik.setpos(-10 * dim + 10, -10 * dim + 10)
        ritwik.pendown()
        ritwik.left(90)

    def next_move(self, sensors):
        if self.flag == 0 and self.gn1 == 1:
            addtograph(self.graph, self.location, self.heading, sensors)
            cleanup_graph(self.graph)
            addtolocated(self.located, self.location, self.heading, sensors, 0.5)
            print goodneighbor(self.graph, self.visited, self.location)
            rotation, movement = traverse(self.location, goodneighbor(self.graph, self.visited, self.location),
                                          self.heading)
            if np.amax(sensors) == 0 or movement == -1:
                self.visited[self.location[0]][self.location[1]] += 1
                self.located[self.location[0]][self.location[1]] += 1
            subir.right(rotation)
            subir.forward(movement*20)
            self.heading = steer(self.heading, rotation)
            self.location = move(self.heading, self.location, movement)
            self.visited[self.location[0]][self.location[1]] += 1
            self.located[self.location[0]][self.location[1]] += 1
            print self.visited
            print self.located
            print showcoverage(self.located)

            if isGoal(self.location, self.maze_dim):
                self.flag = 1
                self.goal_loc = self.location
                self.count = 0
                time.sleep(0.5)
                print "GOAL IS REACHED YOHOO!!!"
                time.sleep(1)
        elif self.flag == 0 and self.gn1 == 0:
            addtograph(self.graph, self.location, self.heading, sensors)
            cleanup_graph(self.graph)
            addtolocated(self.located, self.location, self.heading, sensors, 0.5)
            if self.vis2[self.location[1] * dim + self.location[0]] == 0:
                self.vis2[self.location[1] * dim + self.location[0]] = 1
                self.s.push(self.location[1] * dim + self.location[0])
            rotation, movement = backtracker(self.s, self.heading, self.vis2, self.graph)
            if np.amax(sensors) == 0 or movement < 0:
                self.located[self.location[0]][self.location[1]] += 1
            subir.right(rotation)
            subir.forward(movement * 20)
            self.heading = steer(self.heading, rotation)
            self.location = move(self.heading, self.location, movement)
            self.located[self.location[0]][self.location[1]] += 1
            print showcoverage(self.located)
            if isGoal(self.location, self.maze_dim):
                self.flag = 1
                self.goal_loc = self.location
                self.count = 0
                time.sleep(0.5)
                print "GOAL IS REACHED YOHOO!!!"
                time.sleep(1)
        elif self.flag == 1 and self.count == 0 and showcoverage(self.located) < 76 and self.gn2 == 1:
            addtograph(self.graph, self.location, self.heading, sensors)
            cleanup_graph(self.graph)
            addtolocated(self.located, self.location, self.heading, sensors, 0.5)
            rotation, movement = traverse(self.location,
                                          badneighbor(self.graph, self.located, self.location), self.heading)
            if np.amax(sensors) == 0 or movement == -1:
                self.located[self.location[0]][self.location[1]] += 1
            subir.right(rotation)
            subir.forward(movement * 20)
            self.heading = steer(self.heading, rotation)
            self.location = move(self.heading, self.location, movement)
            self.located[self.location[0]][self.location[1]] += 1
            print showcoverage(self.located)
            print self.located
        elif self.flag == 1 and self.count == 0 and showcoverage(self.located) < 76 and self.gn2 == 0:
            addtograph(self.graph, self.location, self.heading, sensors)
            cleanup_graph(self.graph)
            addtolocated(self.located, self.location, self.heading, sensors, 0.5)
            if self.vis2[self.location[1] * dim + self.location[0]] == 0:
                self.vis2[self.location[1] * dim + self.location[0]] = 1
                self.s.push(self.location[1] * dim + self.location[0])
            rotation, movement = locbacktracker(self.s, self.heading, self.vis2, self.graph, self.located)
            if np.amax(sensors) == 0 or movement < 0:
                self.located[self.location[0]][self.location[1]] += 1
            subir.right(rotation)
            subir.forward(movement * 20)
            self.heading = steer(self.heading, rotation)
            self.location = move(self.heading, self.location, movement)
            self.located[self.location[0]][self.location[1]] += 1
            print showcoverage(self.located)
            print self.located
        else:
            if self.count == 0:
                self.count = 1
                cleanup_graph(self.graph)
                self.path = bidirectional_dijkstra(self.graph, 0, self.goal_loc[1] * dim + self.goal_loc[0])
                print self.path, len(self.path)
                self.location = [0, 0]
                self.heading = 'up'
                return 'Reset', 'Reset'
            else:
                rotation, movement = travel(self.location[1] * dim + self.location[0], self.path[self.count],
                                            self.heading)
                ritwik.right(rotation)
                ritwik.forward(movement * 20)
                self.heading = steer(self.heading, rotation)
                self.location = move(self.heading, self.location, movement)
                self.count += 1
                print self.location
        return rotation, movement
