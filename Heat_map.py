# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 17:00:43 2020

@author: Gabriele Pinna
"""
'''This file generates an animation (heat map), showing how
 the probability of finishing the game changes as the number of turns
increases. There is a major drawback: the quality of the immage generated
via plot_board() does not correspond to the same quality in the gif file.'''



from Snakes_ladders import *
from matplotlib import colors
import imageio
from io import BytesIO


'''ANIMATION' (Heat map)'''
    


def redbar():
    '''# Make a red colorbar with increasing opacity'''
    c = np.zeros((100, 4))
    c[:, -1] = np.linspace(0, 1, 100)  # transparency gradient
    c[:, 0] = 0.8  # make the map red
    TransparencyMap = colors.ListedColormap(c)
    
    return TransparencyMap


def show_board(turn):
    '''Generate the heat map for different turns'''
    fig, ax = plt.subplots()
    board = plt.imread('board.tif') #board file_name
    
    # Compute the state vector
    v_0 = np.identity(101)[0]
    mat = sl_mkm(100, 6, 1, sl)
    prob = sl_probability(turn,v_0, mat)

    prob = prob[1:].reshape(10, 10)[::-1]
    prob[::2] = prob[::2, ::-1]
    
    # Heat map
    ax.imshow(board, alpha=0.9, extent =[-129, 920, 920, -110])     
    im = ax.imshow(prob, extent =[10, 800, 810, 10],               
                   norm=colors.LogNorm(vmin=1E-3, vmax=1),
                   cmap=redbar())
    fig.colorbar(im, ax=ax, label='Fraction of games')
    ax.axis('off')
    ax.set_title(f"Turn {turn}")
    
    return fig




def gif(figures, filename, fps=10, **kwargs):
    '''Create the heat map simulation'''
    images = []
    for fig in figures:
        output = BytesIO()
        fig.savefig(output)
        plt.close(fig)
        output.seek(0)
        images.append(imageio.imread(output))
    imageio.mimsave(filename, images, fps=fps, **kwargs)


frames = [*range(15), *range(15, 50, 5), *range(50, 101, 10)]
sims = (show_board(i) for i in frames) #generate the heat map for differen turns

duration = np.zeros(len(frames))
duration[:8] = 0.5
duration[-1] = 1.0



if __name__ == "__main__":
    plot_board(100, sl)
    #plt.savefig('board.tiff', dpi=1800) #save the board generated using plot_board(100, sl)
    gif(sims, 'Heatmap.gif', fps=10, duration=list(duration))
    
    
    
    
    
    