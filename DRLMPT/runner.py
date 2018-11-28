import datetime as dt
import numpy as np
from simulator import Simulator
from actor_critic_agents import ActorAgent

"""
        
        Code based on previous code written by Yichen Shen and Yiding Zhao 
        ( <https://github.com/shenyichen105/Deep-Reinforcement-Learning-in-Stock-Trading> )
        
"""

def main():
    actions = ["buyA", "sellA", "buyB", "sellB","buyC", "sellC", "hold"]

    environment = Simulator(['ITSA4', 'ITUB3', 'ITUB4'], dt.datetime(2007, 1, 1), dt.datetime(2018, 1, 1))

    # Creates instance of agent
    agent = ActorAgent(lookback=environment.init_state())
    
    # Choice of first action
    action = agent.init_query()

    while environment.has_more():

            # Maps action from id to name
            action = actions[action]

            # Simulation step (trading day). Apart from reward and state, the simulator provides the agent with the current day 
            # and information concerning restrictions (boundary). The agent uses the first one in the computation of a kind of
            # epsilon-greedy policy; the second piece of information serves to limit the ability of the agent to open or close
            # positions in certain scenarios 
            reward, state, day, boundary = environment.step(action)

            # Agent takes action
            action = agent.query(state, reward, day, boundary)

if __name__ == '__main__':
    main()


