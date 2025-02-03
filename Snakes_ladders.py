# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 21:09:18 2019

@author: Gabriele Pinna
"""

'''A statistical analysis of the popular board game Snakes & Ladders.
Three methods are used:
    - Monte Carlo simulation
    - Markov matrices
    - Absorbing Markov matrices
To call the various functions use the main statement at the end of the code.
The following abbreviations are used in some function names/parameters:
    sl: snakes and ladders
    mcm: Monte Carlo simulation (method 1)
    mkm: Markov matrix (method 2)
    abs: Absorbing Markov matrix (method 3)
'''

import numpy as np
import matplotlib.pyplot as plt
import statistics
from scipy import stats
import pandas as pd

pd.options.display.max_columns = 10 #avoid truncation of DataFrame table


sl = {1:38, 4:14, 9:31, 16:6, 21:42, 28:84, 36:44,
                  47:26, 49:11, 51:67, 56:53, 62:19, 64:60,
                  71:91, 80:100, 87:24, 93:73, 95:75, 98:78} #common snakes and ladders locations


'Monte Carlo simulation'

def simulate(board_size, max_roll, p_0, version, sl):
    '''Simulation of the game.
    Parameters:
    board_size: number of squares (should be a square of an integer)
    max_roll: the spinner range [1,max_roll]
    p_0: starting position
    version: three winning coditions (1,2,3)
    sl: dictionary that encodes the starting:ending positions of snakes/ladders
    '''
    
    turn = 0
    position = sl.get(p_0, p_0)
    while position<board_size:
        turn+=1
        roll = np.random.randint(1,max_roll+1)
        if position + roll > board_size: #do not move
            if version == 1:
                continue # do not move 
            
            elif version ==2:
                position = board_size # win
            
            elif version==3:
                position = 2*board_size - (position+roll) #bounce back        
        else:
            position += roll
            
        position = sl.get(position, position)
        
    return turn
    


def mcs(board_size, N, max_roll, p_0, version, sl):
    '''Monte Carlo simulation (mcs) of the games with N iterations.
    Parameters:
    board_size: number of squares (should be a square of an integer)
    N: number of times the simulation is run
    max_roll: the spinner range [1,max_roll]
    p_0: starting position
    version: three winning conditions (1,2,3)
    sl: dictionary that encodes the starting:ending positions of snakes/ladders
    '''
    
    result = [simulate(board_size, max_roll, p_0, version, sl) for i in range(N+1)]
    
    return result



def plot_pdf_mcs(board_size, N, max_roll, version, sl):
    '''Generate a plot of the pdf, using the Monte Carlo simulation.
    Parameters:
    board_size: number of squares (should be a square of an integer)
    N: number of times the simulation is run
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary that encodes the starting:ending positions of snakes/ladders
    '''
    
    data = mcs(board_size, N, max_roll, 0, version, sl)
    fig = plt.figure()
    plt.hist(data, bins=range(200), density = True) #for older versions of numpy use normed instead of density
    plt.xlabel('Number of turns', fontsize = 12)
    plt.ylabel('Fraction of games', fontsize = 12)
    plt.title('Probability density function')
    mean= np.mean(data)
    mode = statistics.mode(data)
    median = np.median(data)
    var = statistics.variance(data)
    skw = stats.skew(data)
    k = stats.kurtosis(data, fisher =False) #using Pearson's definition
    
    data_dic = {'Mean': [mean], 'Mode': [mode], 'Median': [median], 'Variance': [var], 
                'Skewness': [skw], 'Kurtosis': [k]}
    table = pd.DataFrame(data_dic)
    
    print(table)
    
    return fig



def plot_cdf_mcs(board_size, N, max_roll,  version, sl):
    '''Generate a plot of the cdf, using Monte Carlo simulation.
    Parameters:
    board_size: number of squares (should be the square of an integer)
    N: number of times the simulation is run
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders
    '''
    
    data = mcs(board_size, N, max_roll, 0, version, sl)
    fig = plt.figure()
    counts, bins = np.histogram(data, bins = range(len(data)), density = True)
    area = np.cumsum(np.diff(bins)*counts)
    t = np.cumsum(np.ones_like(area))
    plt.plot(t, area)
    plt.xlabel('Number of turns', fontsize = 12)
    plt.ylabel('Cumulative probability', fontsize = 12)
    plt.title('Cumulative distribution function')
    plt.xlim(0,1000)
    
    
    return fig
    


def order_mcs(board_size, N, max_roll, version, sl):
    '''Order the squares depending on the average turns, using Monte Carlo.
    Parameters:
    board_size: number of squares (should be a square of an integer)
    N: number of times the simulation is run
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders
    '''
    l= np.arange(0, board_size+1, 1)
    b  = [np.mean(mcs(board_size, N, max_roll, value, version, sl)) for value in l]
    order = np.argsort(b)
    order = [x for x in order if x not in sl.keys()]
    # the starting points of a snake (or ladder) are deleted, 
    # because they are inaccessible
    
    return order





'Markov matrix'

def sl_mkm(board_size, max_roll, version, sl, roll_first = True):
    '''
    Create the Markov matrix.
    Parameters:
    board_size: number of squares (should be a square of an integer)
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders
    '''
    mat = np.zeros((board_size+1, board_size+1))
    
    #dice matrix
    for i in range(board_size+1):
        mat[i, i + 1: i + 1 + max_roll] = np.float64(1 / max_roll)
        
    if version == 1:
        np.fill_diagonal(mat, np.float64(1 - np.sum(mat, axis = 1)))
    
    elif version ==2:
        mat[:, -1] += np.float64(1 - np.sum(mat, axis = 1))
        
        
    elif version ==3: #max_roll should not exceed board_size
        for i in range(board_size-max_roll+1, board_size):
            a = 1/max_roll*np.ones(i+max_roll-board_size)
            a.resize(board_size)
            b = a[::-1]
            b = np.append(b, 0)
            mat[i, :] += b
        mat[board_size,board_size] = 1    
                                
    #snakes and ladders matrix
    sl_mat = np.zeros((board_size+1, board_size+1))
    ind = [sl.get(i, i) for i in range(board_size+1)]
    sl_mat[range(board_size+1), ind] = 1
    
    if roll_first == True:
        return np.matmul(mat, sl_mat)
    
    else:
        return np.matmul(sl_mat, mat)
        
    
    
def sl_probability(n, v, mat):
    '''Compute the state vector after n turns.
    Parameters:
    n: number of turns
    v: initial state vector
    mat: appropriate Markov matrix
    '''
    
    return np.matmul(v, np.linalg.matrix_power(mat, n))



def mean_mkm(board_size, v, n, max_roll, version, sl):
    '''Compute the average of the pdf.
    Parameters:
    board_size: number of squares (should be a square of an integer)
    v: initial state vector
    n: number of times the simulation is run
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders
    '''
    
    mat = sl_mkm(board_size, max_roll, version, sl) 
    probs = [sl_probability(i, v, mat)[-1] for i in range(n)]
    mean = np.dot(np.diff(probs), np.arange(1, n))
    
    return mean



def var_mkm(board_size, v, n, max_roll, version, sl):
    '''Compute the variance of the pdf.
    Parameters:
    board_size: number of squares (should be a square of an integer)
    n: number of times the simulation is run
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders
    '''
    mat = sl_mkm(board_size, max_roll, version, sl) 
    probs = [sl_probability(i, v, mat)[-1] for i in range(n)]
    pdf = np.diff(probs)
    turns = np.arange(1, n, dtype = np.float64)
    mean = mean_mkm(board_size, v, n, max_roll, version, sl)
    mean_2 = np.dot(pdf , np.square(turns))
    var = mean_2-np.square(mean)
          
    return var



def skw_mkm(board_size, v, n, max_roll, version, sl):
    '''Compute the skewness of the pdf.
    Parameters:
    board_size: number of squares (should be a square of an integer)
    n: number of times the simulation is run
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders
    '''
    mat = sl_mkm(board_size, max_roll, version, sl) 
    probs = [sl_probability(i, v, mat)[-1] for i in range(n)]
    pdf = np.diff(probs)
    turns = np.arange(1, n, dtype = np.float64)
    mean = mean_mkm(board_size, v, n, max_roll, version, sl)
    mean_2 = np.dot(pdf , np.square(turns))
    var = mean_2-np.square(mean)
    mean_3 = np.dot(pdf, (turns)**3)
    skw = (mean_3 - 3*mean*var -mean**3)/(var)**(3/2)
      
    return skw



def k_mkm(board_size, v, n, max_roll, version, sl):
    '''Compute the kurtosis of the pdf.
    Parameters:
    board_size: number of squares (should be a square of an integer)
    n: number of times the simulation is run
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders
    '''
    mat = sl_mkm(board_size, max_roll, version, sl) 
    probs = [sl_probability(i, v, mat)[-1] for i in range(n)]
    pdf = np.diff(probs)
    turns = np.arange(1, n, dtype = np.float64)
    mean = mean_mkm(board_size, v, n, max_roll, version, sl)
    mean_2 = np.dot(pdf , np.square(turns))
    var = mean_2-np.square(mean)
    mean_3 = np.dot(pdf, (turns)**3)
    mean_4 = np.dot(pdf, (turns**4))
    k =  (mean_4 - 4*mean_3*mean -3*mean**4 + 6*mean_2*np.square(mean))/np.square(var)  
    
    return k



def plot_pdf_mkm(board_size, n, max_roll, version, sl):
    '''Generate a plot of the probability density, using Markov matrices.
    Parameters:
    board_size: number of squares (should be a square of an integer)
    n: number of turns
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders
    '''
    
    v_0 = [1, *np.zeros(board_size)]
    
    mat = sl_mkm(board_size, max_roll, version, sl)      
    probs = [sl_probability(i, v_0, mat)[-1] for i in range(n)] #-1 indicate prob(x=final square)
    pdf = np.diff(probs)
    turns = np.arange(1, n, dtype=np.float64)
    
    fig = plt.figure()
    plt.plot(turns, pdf , color='red')
    plt.title('Probability density function')
    plt.xlabel('Number of turns', fontsize = 12)
    plt.ylabel('Fraction of games', fontsize = 12)
    plt.xlim(0,200)
    plt.ylim(np.interp(0, turns, pdf))
    mean = mean_mkm(board_size,v_0, n, max_roll, version, sl)
    mode = np.argmax(pdf) + 1
    var = var_mkm(board_size,v_0, n, max_roll, version, sl)
    median = np.interp(1/2, probs, np.arange(n))
    skw = skw_mkm(board_size, v_0, n, max_roll, version, sl)
    k =  k_mkm(board_size,v_0, n, max_roll, version, sl)
    
    mean_y = np.interp(mean, turns, pdf) # y-value of the mean
    mode_y = np.interp(mode, turns, pdf) # y-value of the mode
    median_y =  np.interp(median, turns, pdf) # y-value of the median
    
    
    plt.vlines(mean, 0, mean_y, color ='k' ,label=f'mean = {mean:.3f}',linewidth=2)
    plt.vlines(mode, 0, mode_y, color = 'b', label=f'mode = {mode}',linewidth=2)
    plt.vlines(median, 0, median_y, color = 'green', label=f'median = {median:.3f}',linewidth=2)
    plt.legend()
    
    data_dic = {'Mean': [mean], 'Mode': [mode], 'Median': [median], 'Variance': [var], 
                'Skewness': [skw], 'Kurtosis': [k]}
    table = pd.DataFrame(data_dic) #table showing statistical measures
    print(table)  

    return fig



def plot_cdf_mkm(board_size, n, max_roll, version, sl):
    '''Generate a plot of the cumulative probability, using Markov matrices.
    Parameters:
    board_size: number of squares (should be a square of an integer)
    n: number of turns
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders
    '''
    
    v_0 = [1, *np.zeros(board_size)]
    mat = sl_mkm(board_size, max_roll, version, sl) 
    probs = [sl_probability(i, v_0, mat)[-1] for i in range(n)]
    
    fig = plt.figure()
    plt.plot(np.arange(n), probs, color='red')
    plt.title('Cumulative distribution function')
    plt.xlabel('Number of turns', fontsize = 12)
    plt.ylabel('Cumulative probability', fontsize = 12)
    plt.xlim(0,1000)
    
    print(f'Shortest game = {np.nonzero(probs)[0][0]} turns')
    
    return fig

    

def order_mkm(board_size, n, max_roll, version, sl):
    '''Order the squares depending on the average turns, using Markov matrix (mkm).
    Parameters:
    board_size: number of squares (should be a square of an integer)
    n: number of turns
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders
    '''
    
    I = np.identity(board_size+1)
    m = [mean_mkm(board_size, v, n, max_roll, version, sl) for v in I]
    order = np.argsort(m)
    order = [x for x in order if x not in sl.keys()]
    # the starting points of a snake (or ladder) are deleted, 
    # because they are inaccessible
    
    return order
   
    



'Absorbing markov matrix'

def order_abs(board_size, max_roll, version, sl):
    '''Order the squares depending on the average turns, using absorbing Markov matrix.
    Parameters:
    board_size: number of squares (should be a square of an integer)
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders
    '''
    
    mat = sl_mkm(board_size, max_roll, version, sl)  
    Q = mat[:board_size,:board_size]
    N = np.linalg.matrix_power(np.identity(board_size)-Q, -1)
    one = np.ones(board_size)  
    order = np.argsort(np.matmul(N, one))
    order = [x for x in order if x not in sl.keys()]
    # the starting points of a snake (or ladder) are deleted, 
    # because they are inaccessible
    
    return order



def mean_abs(board_size, max_roll, version, sl):
    '''Compute the average duration, using absorbing Markov matrix.
    Parameters:
    board_size: number of squares (should be a square of an integer)
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders
    '''
    
    mat = sl_mkm(board_size, max_roll, version, sl)    
    Q = mat[:board_size,:board_size]
    N = np.linalg.matrix_power(np.identity(board_size)-Q, -1)
    one = np.ones(board_size)
    mean = np.matmul(N, one)
    
    return mean



def var_abs(board_size, max_roll, version, sl):
    '''Compute the variance, using absorbing Markov matrix.
    Parameters:
    board_size: number of squares (should be a square of an integer)
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders
    '''
    
    mat = sl_mkm(board_size, max_roll, version, sl)    
    Q = mat[:board_size,:board_size]
    N = np.linalg.matrix_power(np.identity(board_size)-Q, -1)
    one = np.ones(board_size)
    mean = np.matmul(N, one)
    var = (2*N-np.identity(board_size)) @ mean - np.square(mean)
    
    return var



def skw_abs(board_size, max_roll, version, sl):
    '''Compute the skewness, using absorbing Markov matrix.
    Parameters:
    board_size: number of squares (should be a square of an integer)
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders
    '''
    mat = sl_mkm(board_size, max_roll, version, sl)    
    Q = mat[:board_size,:board_size]
    N = np.linalg.matrix_power(np.identity(board_size)-Q, -1)
    one = np.ones(board_size)
    tau = np.matmul(N, one)
    I = np.identity(board_size)
    var = (2*N-I) @ tau - np.square(tau)
    skw = ((6*N@(N-I)+I)@tau-3*tau*var-tau**3)/(var**(3/2))
    
    return skw
 
    

def k_abs(board_size, max_roll, version, sl):
    '''Compute the kurtosis, using absorbing Markov matrix.
    Parameters:
    board_size: number of squares (should be a square of an integer)
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders
    '''
    mat = sl_mkm(board_size, max_roll, version, sl)    
    Q = mat[:board_size,:board_size]
    N = np.linalg.matrix_power(np.identity(board_size)-Q, -1)
    one = np.ones(board_size)
    tau = np.matmul(N, one)
    I = np.identity(board_size)
    var = (2*N-I) @ tau - np.square(tau)
    tau_2 = (2*N-I)@tau
    tau_3 = tau + 3*(N-I)@(tau_2 + tau)
    tau_4 = tau + (N-I)@(4*tau_3 + 6*tau_2 + 4*tau)
    k =  (tau_4 - 4*tau_3*tau -3*tau**4 +6*tau_2*tau**2)/np.square(var)
    
    return k   



def most_least_visited(board_size, max_roll, version, sl):
    '''Compute the most/least visited state, using absorbing Markov matrix.
    The starting points of ladders/snakes are not included, as they are
    virtually inaccessible.
    Parameters:
    board_size: number of squares (should be a square of an integer)
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders
    '''
    
    mat = sl_mkm(board_size, max_roll, version, sl)
    Q = mat[:board_size,:board_size]
    N = np.linalg.matrix_power(np.identity(board_size)-Q, -1)
    a = N[0,:] #starting from square 0
    a[a==0] = np.nan
    
    return ([np.nanargmax(a), np.nanmax(a)], [np.nanargmin(a), np.nanmin(a)])



def variance_visits(board_size, max_roll, version, sl):
    ''' Variance of the number of visits to a transient state j 
    starting from a transient state i (before being absorbed)
    Parameters:
    board_size: number of squares (should be a square of an integer)
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders
    '''
    
    mat = sl_mkm(board_size, max_roll, version, sl)
    Q = mat[:board_size,:board_size]
    N = np.linalg.matrix_power(np.identity(board_size)-Q, -1)
    N_diag = np.diag(np.diag(N))
    N_2 = N @ (2*N_diag - np.identity(board_size)) -np.square(N)
    
    return N_2
   


def table_abs(board_size, max_roll, version, sl):
    ''' Table that summarieses the results obtained using absorbing matrices
    Parameters:
    board_size: number of squares (should be a square of an integer)
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders
    '''
    mean = mean_abs(board_size, max_roll, version, sl)[0]
    var = var_abs(board_size, max_roll, version, sl)[0]
    skw = skw_abs(board_size, max_roll, version, sl)[0]
    k = k_abs(board_size, max_roll, version, sl)[0]
    m = most_least_visited(board_size, max_roll, version, sl)[0][0]
    l = most_least_visited(board_size, max_roll, version, sl)[1][0]
    data_dic = {'Mean': [mean], 'Variance': [var], 
                'Skewness': [skw], 'Kurtosis': [k],
                'Most visited square': [m], 'Least visited square': [l]}
    table = pd.DataFrame(data_dic) #table showing statistical measures
    np.set_printoptions(precision = 4, threshold = 1000)
    
    return table
    


def random_sl(board_size, n):
    '''Generate random snakes/ladders locations.
    Parameters:
    board_size: number of squares (should be a square of an integer)
    n: number of snakes and ladders per 100 squares
    '''
    
    sl = np.random.choice(np.arange(1,board_size,1), int(np.floor(n*board_size/100)*2), replace=False)
    sl = np.split(sl, int(np.floor(n*board_size/100)))
    sl = {l:m for l,m in sl}

    return sl



def plot_board(board_size, sl):
    '''Plot a generic board.
    Parameters:
    board_size: number of squares (should be a square of an integer)
    sl:  dictionary with the starting:ending positions of snakes/ladders
    '''
    
    n = np.sqrt(board_size)    
    fig,ax = plt.subplots(1, figsize=(4,4))
    plt.plot()
    ax.set_xlim(1, n+1)
    ax.set_ylim(1, n+1)
    plt.grid(color='k', linestyle='-', linewidth=2)
    ax.set_xticks(np.arange(1,n+1,1))
    ax.set_yticks(np.arange(1,n+1,1))
    for i in range(1,board_size+1):
        if np.ceil(i/n)%2 == 0:
            k =  2*n*np.floor((i-1)/n) + n + 1-i
        else:
            k = i
        plt.text((k-n*(np.ceil(k/n)-1))+1/20, np.ceil(k/n)+1/20,int(i), fontsize = 10)
    x = [(k,v) for k,v in sl.items()]
    for i,l in x:
        if np.ceil(i/n)%2 == 0:
            i =  2*n*np.floor((i-1)/n) + n + 1-i
            
        if np.ceil(l/n)%2 ==0:
            l =  2*n*np.floor((l-1)/n) + n + 1-l
        plt.plot((i-n*(np.ceil(i/n)-1)) + 1/2, np.ceil(i/n) +1/2, marker='o',
                 markersize=5, color="red")
        plt.arrow((i-n*(np.ceil(i/n)-1)) + 1/2, np.ceil(i/n) +1/2 ,
     (l-n*(np.ceil(l/n)-1)) -(i-n*(np.ceil(i/n)-1)),np.ceil(l/n) -np.ceil(i/n),
     head_width=0.3, head_length=0.3 )

    return fig


 
def plot_roll():
    '''Plot the average number of turns as the spinner range 
    increases for the three different versions.
    '''
    
    roll = np.arange(2, 100,1)
    average1 = np.array([mean_abs(100, item, 1, sl)[0] for item in roll])
    average2 = np.array([mean_abs(100, item, 2, sl)[0] for item in roll])
    average3 = np.array([mean_abs(100, item, 3, sl)[0] for item in roll])
    fig = plt.figure()
    plt.xlabel('Spinner range', fontsize = 12)
    plt.ylabel('Average duration (turns)', fontsize = 12)
    plt.plot(roll, average1, label = 'Classic')
    plt.plot(roll, average2, label = 'Fast')
    plt.plot(roll, average3, label = 'Bounce back')
    plt.title('Average duration as a function of spinner range')
    r =  np.argwhere(np.diff(np.sign(average3 - average1)))[1] + 2 #integer near intersection 
    # of curve 1 and 3
    
    print(f'The fastest game (version 1) corresponds to max_roll = {np.argmin(average1)+2}')
    print(f'The fastest game (version 2) corresponds to max_roll = {np.argmin(average2)+2}')
    print(f'The fastest game (version 3) corresponds to max_roll = {np.argmin(average3)+2}')
    print(f'The bounce back version is faster than the classic '
          f'for a spinner range less than {r[0]}')
    
    plt.legend()
    plt.show()
    
    return fig





'Error analysis'

def plot_pdf():
    '''Plot Markov pdf and Monte Carlo pdf on the same graph'''
    
    r = mcs(100, 10**5, 6, 0, 1, sl)
    fig = plt.figure()
    plt.hist(r, bins=range(200), density = True, label = 'Monte Carlo')
    plt.xlabel('Duration (turns)', fontsize = 12)
    plt.ylabel('Fraction of games', fontsize = 12)
    plt.title('PDF', fontsize = 14)
    v_0 = [1, *np.zeros(100)]
    
    mat = sl_mkm(100, 6, 1, sl)      
    probs = [sl_probability(i, v_0, mat)[-1] for i in range(10**3)]
    plt.plot(np.arange(1, 10**3), np.diff(probs), color='red', label = 'Markov')
    plt.xlim(0,200)
    plt.legend()
    
    return fig
    
    

def plot_cdf():
    '''Plot Markov cdf and Monte Carlo cdf on the same graph'''
    
    fig = plt.figure()
    r = mcs(100, 10**5, 6, 0, 1, sl)
    counts, bins = np.histogram(r, bins = range(len(r)), density = True)
    area = np.cumsum(np.diff(bins)*counts)
    t = np.cumsum(np.ones_like(area))
    plt.plot(t, area, label = 'Monte Carlo')
    plt.xlim(0,100)
    v_0 = [1, *np.zeros(100)]
    mat = sl_mkm(100, 6, 1, sl) 
    probs = [sl_probability(i, v_0, mat)[-1] for i in range(10**3)]
    plt.plot(np.arange(10**3), probs, color='red', label = 'Markov' )
    plt.xlabel('Turns', fontsize = 12)
    plt.ylabel('Cumulative probability', fontsize = 12)
    plt.title('CDF', fontsize = 14)
    plt.legend()
    plt.show()
    
    return fig


    
def error_mcs(N, board_size, max_roll, version, sl):
    '''Relative error on the average duration, using the Monte Carlo simulation
    Parameters:
    board_size: number of squares (should be a square of an integer)
    N: number of times the simulation is run
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders
    '''
    
    I = mean_abs(board_size, max_roll, version, sl)[0]
    mean  = np.mean(mcs(board_size, N, max_roll, 0, version, sl))
    error = np.abs(1-mean/I)
    
    return error



def plot_error_mcs(board_size, max_roll, version, sl):
    '''Plot of the relative error on the average duration
    as a function of simulations (Monte Carlo)
    Parameters:
    board_size: number of squares (should be a square of an integer)
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders
    '''
    
    fig = plt.figure()
    l = np.logspace(1,4,50, dtype = np.int64)
    error = [error_mcs(N, board_size, max_roll, version, sl) for N in l]
    plt.loglog(l, error)
    plt.xlabel('Simulations', fontsize =12)
    plt.ylabel('Relative error', fontsize = 12)
    plt.title('Relative error on the mean, using Monte Carlo simulation')
    
    return fig



def error2_mcs(N, board_size, max_roll, version, sl):
    '''Relative error on the variance, using the Monte Carlo simulation
    Parameters:
    board_size: number of squares (should be a square of an integer)
    N: number of times the simulation is run
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders'''
    
    I = var_abs(board_size, max_roll, version, sl)[0]
    data = mcs(board_size, N, max_roll, 0, version, sl)
    var = statistics.variance(data)
    error = np.abs(1-var/I)
    
    return error



def plot_error2_mcs(board_size, max_roll, version, sl):
    '''Plot of the relative error on the variance
    as a function of simulations (Monte Carlo)
    Parameters:
    board_size: number of squares (should be a square of an integer)
    N: number of times the simulation is run
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders'''
    
    fig = plt.figure()
    l = np.logspace(1,4,50, dtype = np.int64)
    error = [error2_mcs(N, board_size, max_roll, version, sl) for N in l]
    plt.loglog(l, error)
    plt.xlabel('Iterations', fontsize =12)
    plt.ylabel('Relative error', fontsize = 12)
    plt.title('Relative error on the variance, using Monte Carlo simulation')
    
    return fig



def error3_mcs(N, board_size, max_roll, version, sl):
    '''Relative error on the skewness, using the Monte Carlo simulation
    Parameters:
    board_size: number of squares (should be a square of an integer)
    N: number of times the simulation is run
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders'''
    
    I = skw_abs(board_size, max_roll, version, sl)[0]
    data = mcs(board_size, N, max_roll, 0, version, sl)
    skw = stats.skew(data)
    error = np.abs(1-skw/I)
    
    return error



def plot_error3_mcs(board_size, max_roll, version, sl):
    '''Plot of the relative error on the skewness
    as a function of simulations (Monte Carlo)
    Parameters:
    board_size: number of squares (should be a square of an integer)
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders
    '''
    
    fig = plt.figure()
    l = np.logspace(1,4,50, dtype = np.int64)
    error = [error3_mcs(N, board_size, max_roll, version, sl) for N in l]
    plt.loglog(l, error)
    plt.xlabel('Iterations', fontsize =12)
    plt.ylabel('Relative error', fontsize = 12)
    plt.title('Relative error on the skewness, using Monte Carlo simulation')
    
    return fig



def error4_mcs(N, board_size, max_roll, version, sl):
    '''Relative error on the kurtosis, using the Monte Carlo simulation
    Parameters:
    board_size: number of squares (should be a square of an integer)
    N: number of times the simulation is run
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders
    '''
    
    I = k_abs(board_size, max_roll, version, sl)[0]
    data = mcs(board_size, N, max_roll, 0, version, sl)
    k = stats.kurtosis(data, fisher =False)
    error = np.abs(1-k/I)
    
    return error



def plot_error4_mcs(board_size, max_roll, version, sl):
    '''Plot of the relative error on the kurtosis
    as a function of simulations (Monte Carlo)
    Parameters:
    board_size: number of squares (should be a square of an integer)
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders
    '''
    
    fig = plt.figure()
    l = np.logspace(1,4,50, dtype = np.int64)
    error = [error4_mcs(N, board_size, max_roll, version, sl) for N in l]
    plt.loglog(l, error)
    plt.xlabel('Iterations', fontsize =12)
    plt.ylabel('Relative error', fontsize = 12)
    plt.title('Relative error on the kurtosis, using Monte Carlo simulation')
    
    return fig  



def error_mkm(N, board_size, max_roll, version, sl):
    '''Relative error on the average using the Markov matrix method
    Parameters:
    board_size: number of squares (should be a square of an integer)
    N: number of times the simulation is run
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders
    '''
    
    I = mean_abs(board_size, max_roll, version, sl)[0]
    v = np.identity(101)[0]
    mean  = mean_mkm(board_size, v, N, max_roll, version, sl)
    error = np.abs(1-mean/I)
    
    return error



def plot_error_mkm(board_size, max_roll, version, sl):
    '''Plot of the relative error on the average duration 
    as a function of the matrix power (turns) (Markov matrix)
    Parameters:
    board_size: number of squares (should be a square of an integer)
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders'''
    
    fig = plt.figure()
    l = np.logspace(1,3,25, dtype = np.int64)
    error = [error_mkm(n, board_size, max_roll, version, sl) for n in l]
    plt.loglog(l, error)
    plt.xlabel('Matrix power', fontsize = 12 )
    plt.ylabel('Error', fontsize = 12)
    plt.title('Relative error on the mean, using Markov matrix')
    
    return fig



def error2_mkm(N, board_size, max_roll, version, sl):
    '''Relative error on the variance, using the Moarkov matrix method
    Parameters:
    board_size: number of squares (should be a square of an integer)
    N: number of times the simulation is run
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders'''
    
    I = var_abs(board_size, max_roll, version, sl)[0]
    v = np.identity(101)[0]
    var  = var_mkm(board_size, v, N, max_roll, version, sl)
    error = np.abs(1-var/I)
    
    return error



def plot_error2_mkm(board_size, max_roll, version, sl):
    '''Plot of the relative error on the variance 
    as a function of the matrix power (Markov matrix)
    Parameters:
    board_size: number of squares (should be a square of an integer)
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders'''
    
    fig = plt.figure()
    l = np.logspace(1,3,25, dtype = np.int64)
    error = [error2_mkm(n, board_size, max_roll, version, sl) for n in l]
    plt.loglog(l, error)
    plt.xlabel('Iterations', fontsize = 12 )
    plt.ylabel('Relative error', fontsize = 12)
    plt.title('Relative error on the variance, using Markov matrix')
    
    return fig



def error3_mkm(N, board_size, max_roll, version, sl):
    '''Relative error on the skewness, using the Markov matrix method
    Parameters:
    board_size: number of squares (should be a square of an integer)
    N: number of times the simulation is run
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders'''
    
    I = skw_abs(board_size, max_roll, version, sl)[0]
    v = np.identity(101)[0]
    skw  = skw_mkm(board_size, v, N, max_roll, version, sl)
    error = np.abs(1-skw/I)
    
    return error



def plot_error3_mkm(board_size, max_roll, version, sl):
    '''Plot of the relative error on the skewness 
    as a function of the matrix power (Markov matrix)
    Parameters:
    board_size: number of squares (should be a square of an integer)
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders'''
    
    fig = plt.figure()
    l = np.logspace(1,3,25, dtype = np.int64)
    error = [error3_mkm(n, board_size, max_roll, version, sl) for n in l]
    plt.loglog(l, error)
    plt.xlabel('Iterations', fontsize = 12 )
    plt.ylabel('Relative error', fontsize = 12)
    plt.title('Relative error on the skewness, using Markov matrix')
    
    return fig



def error4_mkm(N, board_size, max_roll, version, sl):
    '''Relative error on the kurtosis, using the Markov matrix method
    Parameters:
    board_size: number of squares (should be a square of an integer)
    N: number of times the simulation is run
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders'''
    
    I = k_abs(board_size, max_roll, version, sl)[0]
    v = np.identity(101)[0]
    b  = k_mkm(board_size, v, N, max_roll, version, sl)
    error = np.abs(1-b/I)
    
    return error



def plot_error4_mkm(board_size, max_roll, version, sl):
    '''Plot of the relative error on the kurtosis
    as a function of the matrix power (Markov matrix)
    Parameters:
    board_size: number of squares (should be a square of an integer)
    max_roll: the spinner range [1,max_roll]
    version: three winning conditions (1,2,3)
    sl: dictionary with the starting:ending positions of snakes/ladders'''
    
    fig = plt.figure()
    l = np.logspace(1,3,25, dtype = np.int64)
    error = [error4_mkm(n, board_size, max_roll, version, sl) for n in l]
    plt.loglog(l, error)
    plt.xlabel('Iterations', fontsize = 12 )
    plt.ylabel('Relative error', fontsize = 12)
    plt.title('Relative error on the kurtosis, using Markov matrix')
    
    return fig



#Toggle the functions

if __name__ == "__main__":
#    plot_pdf_mcs(100, 10**4, 6, 1, sl)
#    plot_cdf_mcs(100, 10**4, 6, 1, sl)
#    print(order_mcs(100, 10**3, 6, 1, sl))
#    plot_pdf_mkm(100, 10**3,6,1, sl)
#    plot_cdf_mkm(100, 10**3,6,1, sl)
#    print(order_mkm(100, 10**3, 6, 1, sl))
#    print(order_abs(100, 6, 1, sl))
#    print(variance_visits(100, 6, 1, sl)[0, 10])
#    print(table_abs(100, 6, 1, sl))    
#    r_sl = random_sl(49, 15) #run this and the next two lines
#    np.random.seed(0) #freeze the random snakes and ladders
#    plot_board(49, r_sl)
    plot_roll()
#    plot_pdf()
#    plot_cdf()
#    plot_error_mcs(100, 6, 1, sl)
#    plot_error2_mcs(100, 6, 1, sl)
#    plot_error3_mcs(100, 6, 1, sl)
#    plot_error4_mcs(100, 6, 1, sl)
#    plot_error_mkm(100, 6, 1, sl)
#    plot_error2_mkm(100, 6, 1, sl)
#    plot_error3_mkm(100, 6, 1, sl)
#    plot_error4_mkm(100, 6, 1, sl)
    
    
    