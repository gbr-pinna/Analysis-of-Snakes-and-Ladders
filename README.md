# Analysis-of-Snakes-and-Ladders

This project conducts an analysis of the classic **Snakes and Ladders** game using a 10x10 board. The goal is to calculate the average duration of the game under three different winning conditions. The project uses different methods, including numerical simulations, Markov chains, and absorbing Markov chains, to investigate the expected duration and other related statistical metrics.

The three winning conditions explored are:
1. **Classic**: To win, a player must land exactly on square 100; if they overshoot, they remain in place.
2. **Fast**: Overshooting is allowed, meaning the player can win even if they exceed square 100.
3. **Bounce Back**: If a player overshoots, they "bounce back" to a lower square, which may extend the game duration.

### Files in this Project

1. **`Snakes_Ladders.py`**: Contains the core numerical analysis for simulating the game and calculating the expected duration under each of the winning conditions. It also implements the methods for the random simulation, Markov chain, and absorbing Markov chain approaches.

2. **`Heatmap.py`**: Generates an animated heatmap that visualizes the probability of being in any given square of the board at each turn. This heatmap animation is useful for observing the convergence of the game as it progresses over time.

3. **`A Statistical Analysis of Snakes and Ladders.pdf`**: A PDF document that explains the methodology used in the analysis, outlines the findings, and discusses the results of each approach. It provides a detailed breakdown of the statistical models and their implications.

## Results

The analysis focuses on the average number of turns required to win the game under the three winning conditions:

- **Classic**: The average number of turns for the Classic rule is **39.225** turns.
- **Fast**:  The average number of turns for the Fast rule is **35.835** turns.
- **Bounce Back**: The average number of turns for the Bounce Back rule is **43.325** turns.

These results are summarized and discussed in **`A Statistical Analysis of Snakes and Ladders.pdf`**, where the methodology is explained in more detail.

## Methods Used

### 1. **Random Number Simulation**:
   - This method simulates multiple games using random dice rolls. The game progresses by moving the player along the board, encountering snakes and ladders at each step. The number of turns taken to reach the winning square (100) is recorded across many simulations, and histograms are generated to estimate the probability distribution of the game's duration.

### 2. **Markov Chain**:
   - The game can be modeled as a Markov chain, where the states represent the positions on the board and the transitions are governed by dice rolls and board effects (snakes and ladders). We construct a transition matrix **M** and compute the powers of this matrix to update the probabilities of being in each square as the game progresses.
   - The expected duration is derived by analyzing the probability distribution of states at each turn.

### 3. **Absorbing Markov Chain**:
   - The game contains an absorbing state: square 100, where the game ends. By treating the game as an absorbing Markov chain, we express the transition matrix in a canonical form to simplify the calculation of the expected duration to absorption (i.e., how many moves it takes to reach square 100).
