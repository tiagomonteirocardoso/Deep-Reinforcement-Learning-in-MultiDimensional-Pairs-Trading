import datetime as dt
import numpy as np
from simulator import Simulator

"""
        
        Code based on previous code written by Yichen Shen and Yiding Zhao 
        ( <https://github.com/shenyichen105/Deep-Reinforcement-Learning-in-Stock-Trading> )
        
"""

def main():
    actions = ["buyA", "sellA", "buyB", "sellB","buyC", "sellC", "hold"]
    env_train = Simulator(['ITSA4', 'ITUB3', 'ITUB4'], dt.datetime(2017, 1, 1), dt.datetime(2018, 1, 1))

    
    while env_train.has_more():
        action = np.random.randint(7)
        action = actions[action]
        reward, state, day, boundary = env_train.step(action)
        

if __name__ == '__main__':
    main()
