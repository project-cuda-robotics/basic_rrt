import math
import numpy as np
from numba import cuda
import matplotlib.pyplot as plt


def euc_distance_2d(x1,y1,x2,y2):
    return np.sqrt((x2-x1)**2+(y2-y1)**2)

def nearest_neighbor_2d(x,y,V,nov):
    distance = np.zeros(nov)
    for i in range(0,nov):
        distance[i] = euc_distance_2d(x,y,V[0,i],V[1,i])
    ind_min = np.argmin(distance)
    min_dis = distance[ind_min]
    return [min_dis,ind_min]

def collision_check(x,y,obstacle_coordinates,obstacle_radii,epsilon):
    coord_diam = obstacle_coordinates.shape[1]
    num_steps = 101
    allowable_radii = obstacle_radii*2/np.sqrt(3)
    noo = obstacle_radii.size # no. of obstacles
    flag = 1
    #print("Printing the coordinates: ", obstacle_coordinates)
    #plt.plot(obstacle_coordinates[0,:], obstacle_coordinates[1, :])
    for i in range(1, coord_diam):
        x_line = np.linspace(obstacle_coordinates[0, i-1], obstacle_coordinates[0, i], num_steps)
        m_slope = (obstacle_coordinates[1, i]-obstacle_coordinates[1, i-1])/(obstacle_coordinates[0, i]-obstacle_coordinates[0,i-1])
        y_line = m_slope*(x_line - obstacle_coordinates[0, i-1]) + obstacle_coordinates[1, i-1]
        for i in range(0,noo):
            if euc_distance_2d(x,y,x_line[i],y_line[i])<allowable_radii[i]:
                flag = 0
                break
        #plt.plot(x_line, y_line)
        #plt.show()
    #print("line lie in coordinates are: ", y_line)
    """
    for i in range(0,noo):
        if euc_distance_2d(x,y,obstacle_coordinates[0,i],obstacle_coordinates[1,i])<allowable_radii[i]:
            flag = 0
            break
            """
    return flag

def draw_circle(xc,yc,r):
    t = np.arange(0,2*np.pi,.05)
    x = xc+r*np.sin(t)
    y = yc+r*np.cos(t)
    plt.plot(x,y,c='blue')

def main(num_tree):

    max_iter = 6000
    epsilon = 2 # step size

    x = np.zeros(max_iter+1)
    y = np.zeros(max_iter+1)
    vertices = np.zeros([2,max_iter+1])
    A = -np.ones([max_iter+1,max_iter+1])

    flag = 0 # for finding a connectivity path

    # initial and goal points/states
    x0 = 10
    y0 = 10
    x_goal = 90
    y_goal = 90
    plt.figure(figsize=[10,10])
    plt.scatter([x0,x_goal],[y0,y_goal],c='r',marker="P")
    # if euc_distance_2d(x0,y0,x_goal,y_goal)<epsilon:
    #     flag = 1

    # obstacle info
    noo = 16 # no. of obstacles
    radius = np.sqrt(3)/2*epsilon
    obs_radii = radius*np.ones(noo)
    obs_coors = 100*np.random.rand(2,noo) # position of obstacles
    for i in range(0,noo):
        if ((obs_coors[0, i]+obs_radii[i]<=x_goal) and (obs_coors[1, i] +obs_radii[i] <= y_goal )) or ((obs_coors[0, i]+obs_radii[i]<=x0 and obs_coors[1, i]+obs_radii[i]<=y0)):
            obs_coors[0, i] = 100*np.random.rand()
            obs_coors[1, i] = 100*np.random.rand()
            draw_circle(obs_coors[0,i],obs_coors[1,i],obs_radii[i])
        else:
            draw_circle(obs_coors[0,i],obs_coors[1,i],obs_radii[i])

    x[0] = x0;
    y[0] = y0;
    vertices[0,0] = x[0]
    vertices[1,0] = y[0]
    nov = 0 # no. of vertices except the initial one
    A[0,0] = 0

    for i in range(1, max_iter+1):
        x_rand= 100*np.random.rand(1)
        y_rand= 100*np.random.rand(1)
        [min_dis,p_near] = nearest_neighbor_2d(x_rand,y_rand,vertices,nov+1)
        if min_dis<epsilon:
            x_new = x_rand
            y_new = y_rand
        else: # interpolate
            r = epsilon/min_dis # ratio
            x_new = vertices[0,p_near]+r*(x_rand-vertices[0,p_near])
            y_new = vertices[1,p_near]+r*(y_rand-vertices[1,p_near])
        if collision_check(x_new,y_new,obs_coors,obs_radii,epsilon):
            nov = nov+1
            vertices[0,nov] = x_new
            vertices[1,nov] = y_new
            plt.scatter(x_new,y_new,c='g')
            plt.plot([vertices[0,p_near],x_new],[vertices[1,p_near],y_new],c='black')
            A[nov,:] = A[p_near,:]
            A[nov,nov] = nov
            if euc_distance_2d(x_new,y_new,x_goal,y_goal)<epsilon:
                nov = nov+1
                A[nov,:] = A[nov-1,:]
                A[nov,nov] = nov
                vertices[0,nov] = x_goal
                vertices[1,nov] = y_goal
                plt.plot([x_new,x_goal],[y_new,y_goal],c='black')
                flag = 1
                break

    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.axis('scaled')
    #plt.show()

    if flag ==1:
        B = np.zeros(nov)
        nov_path =0 # no. of vertices on the connectivity path
        for i in range(0,nov+1):
            if A[nov,i]>-1:
                B[nov_path]=A[nov,i]
                nov_path += 1
        B = B[0:nov_path]
        #print(B)
        for i in range(0, B.size-1):
            plt.plot([vertices[0,int(B[i])],vertices[0,int(B[i+1])]],[vertices[1,int(B[i])],vertices[1,int(B[i+1])]],c='yellow',linewidth=7,alpha=0.5)
    else:
        print('Failure.')
        
    #print(nov)
    plt.savefig("parralel_tree"+str(num_tree+1)+".png")
    #plt.show()

import time
for n in range(10):
    start = time.time()
    main(n)
    tot_time = (time.time()-start)
    print("Time required for calculating tree : ", n+1, "is: ", tot_time, " sec")

