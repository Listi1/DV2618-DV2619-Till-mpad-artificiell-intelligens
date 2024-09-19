#Viktor Listi
#vili22@student.bth.se

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from collections import deque
import heapq
import math



# Created with help by chatGPT but configured manually
def visualize_maze(maze, start, goal, path, visited_order):
    # Convert maze to a numpy array
    maze_array = np.array(maze)

    # Create a color map
    cmap = mcolors.ListedColormap(['black', 'white', 'green', 'red', 'blue'])
    norm = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5], cmap.N)

    # Initialize display maze
    display_maze = np.copy(maze_array)

    # Mark start, goal, and path
    display_maze[start[1], start[0]] = 2
    display_maze[goal[1], goal[0]] = 3
    for x, y in path:
        if (x, y) != start and (x, y) != goal:
            display_maze[y, x] = 4

    # Create the plot
    fig, ax = plt.subplots()
    cax = ax.imshow(display_maze, cmap=cmap, norm=norm)

    # Plot the order of visited nodes
    for i, (x, y) in enumerate(visited_order):
        ax.text(x, y, str(i + 1), color='black', ha='center', va='center', fontsize=8)

    # Turn off the x and y axis
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()

def get_neighbors(node, maze):
    x, y = node
    neighbors = []

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    #find coordinates for all neighbors
    for dir_x, dir_y in directions:
        new_x = x + dir_x
        new_y = y + dir_y

        #Check if neighbor nodes are valid
        if 0 <= new_x < len(maze[0]) and 0 <= new_y < len(maze) and maze[new_y][new_x] == 1:
            neighbors.append((new_x, new_y))

    return neighbors

def bfs(maze, start, goal):
    visited_order = [] #Order of visited nodes (not final path)
    queue = deque([(start, [start])]) #Initlialze queue with start node and path to it
    visited = [start]

    while queue:
        current_node, path = queue.popleft() #Dequeue node and its path
        visited_order.append(current_node)
        if current_node == goal: #Check if goal is reached
            return path, visited_order
        for neighbor in get_neighbors(current_node, maze): #Explore all neighbors of the current node
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append((neighbor, path + [neighbor])) #Queue neighbor with updated path

    return -1 #No path found

def dfs(maze, start, goal):
    visited_order = [] #Order of visited nodes (not final path)
    stack = [(start, [start])] #Initialize stack with start node and path to it
    visited = [start]

    while stack:
        current_node, path = stack.pop() #Pop node and path from stack
        visited_order.append(current_node)
        if current_node == goal: #Check if goal is reached
            return path, visited_order
        for neighbor in get_neighbors(current_node, maze): #Explore neighbors of current node
            if neighbor not in visited:
                visited.append(neighbor)
                stack.append((neighbor, path + [neighbor])) #pust neighbor with updated path to stakc

    return -1 #No path found

# Manhattan distance
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

#Euclidean distance (not used for this implementation of A*)
def euclidean_heuristic(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def a_star(maze, start, goal):
    open_list = []
    visited_order = [] #Order of visited nodes (not final path)
    heapq.heappush(open_list,(0, 0, start, [start])) #Initialize priority queue with start node
    visited = []

    while open_list:
        bad_variable, cost_so_far, current_node, path = heapq.heappop(open_list) #dequeue node with lowest cost
        visited_order.append(current_node)
        if current_node == goal:  #Check if goal is reached
            return path + [goal],visited_order
        for neighbor in get_neighbors(current_node, maze): #Explore neighbors of current node
            if neighbor not in visited:
                visited.append(neighbor)
                new_cost = cost_so_far + 1 #update cost of neighbors
                heapq.heappush(open_list, ((new_cost + heuristic(neighbor, goal)), new_cost, neighbor, path + [current_node])) #Enqueue neighbor with updated vost

    return -1 #No path found


this_maze = [
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],
    [0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,1],
    [0,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,0,0,0,1],
    [0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,0],
    [0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,1,0,1,0],
    [0,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0]
] #1 = node

def main():

    start = (1,0)
    goal = (18,9)


    path_bfs, visited_order_bfs = bfs(this_maze, start, goal)
    print("BFS: ", path_bfs)
    print("BFS Order: ", visited_order_bfs)
    visualize_maze(this_maze, start, goal, path_bfs, visited_order_bfs)

    path_dfs, visited_order_dfs = dfs(this_maze, start, goal)
    print("DFS: ", path_dfs)
    print("DFS Order: ", visited_order_dfs)
    visualize_maze(this_maze, start, goal, path_dfs, visited_order_dfs)

    path_a_star, visited_order_a_star = a_star(this_maze, start, goal)
    print("A*: ", path_a_star)
    print("A* Order: ", visited_order_a_star)
    visualize_maze(this_maze, start, goal, path_a_star, visited_order_a_star)

    return 0

if __name__=="__main__":
    main()

