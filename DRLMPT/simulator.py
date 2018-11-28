import numpy as np
import pandas as pd
import util
import datetime as dt
import csv

        

"""
Code based on previous code written by Yichen Shen and Yiding Zhao 
( <https://github.com/shenyichen105/Deep-Reinforcement-Learning-in-Stock-Trading> )

"""
# Simulates the market. It accept the agent's choice of action, executes the respective operations (buying, selling), 
# atualizes the portfolio composition and value and returns a reward for the action and the new state, dependent on
# next day's stock prices. 

class Simulator(object):

    def __init__(self, symbols,
        start_date=dt.datetime(2008,1,1),           
        end_date= dt.datetime(2009,1,1)):

        self.dates_range = pd.date_range(start_date, end_date)
        self.check = 0
        self.cum_reward = 0

        # initialize portfolio's cash
        self.init_cash = 100000
        
        # value of each position (long or short)
        self.buy_volume = 8000
        
        self.cum_operations = 0
        self.cum_cash = self.init_cash
        self.date_start = 1

        #for visualization
        self.data_out = []

        # preprocessing time series
        # stock symbol data
        stock_symbols = symbols[:]

        # price data
        prices_all = util.get_prices(symbols, start_date, end_date)

        self.stock_A = stock_symbols[0]
        self.stock_B = stock_symbols[1]
        self.stock_C = stock_symbols[2]

        # first trading day
        self.dateIdx = 0
        self.date = prices_all.index[0]
        self.start_date = start_date
        self.end_date = end_date

        self.prices = prices_all[stock_symbols]
        self.prices_ibovespa = prices_all['ibovespa']

        # keep track of portfolio value as a series.
        #########################
        # the variables 'longA', 'longB' and 'longC' keep track of the number of opened positions on each stock. When a stock is 
        # bought, the variables are increased 1. When a stock is sold, the variables are decreased 1. When the variables' values are
        # zero, no positions are opened.
        #########################
        # variable 'a_vol' keeps track of the number of A shares traded (either bought or sold) in the form of a list. 
        # Whenever a position is opened, it is feeded. Whenever a position is closed, the respective register is deleted. The
        # positions are closed beginning with the last ones.
        #########################
        # variable 'a_price' is a list that keeps track of the price of the stock A when a trade occurs. Likewise, Whenever a 
        # position is opened, it is feeded. Whenever a position is closed, the respective register is deleted.
        #########################
        # The same observations apply to 'b_vol', 'c_vol', 'b_price' and 'c_price'. All those variables are needed for the 
        # computation of returns when the positions are closed.
        self.portfolio = {'cash': self.init_cash, 'a_vol': [], 'a_price': [], 'b_vol': [], 'b_price': [], 'c_vol': [], 'c_price': [], 'longA': 0, 'longB': 0, 'longC': 0}
        self.port_val = self.port_value()
        self.port_val_market = self.port_value_for_output()



    def init_state(self, lookback=50):
        """
        return initial states of the market
        """
        states = []
        states.append(self.get_state(self.date))
        for _ in range(lookback-1):
            
            self.dateIdx += 1
            self.date = self.prices.index[self.dateIdx]
            states.append(self.get_state(self.date))

        return states

    def step(self, action):
        
        """
        Take an action and and move the date forward.
        
        There are 7 actions: buyA, sellA, buyB, sellB, buyC, sellC and hold

        returns reward, next day's market status (the "state"), the day and the restrictions ("boundary")
        """

        buy_volume = self.buy_volume
        abs_return_A = 0
        pct_return_A = 0
        abs_return_B = 0
        pct_return_B = 0
        abs_return_C = 0
        pct_return_C = 0
        A_cost = 0
        B_cost = 0
        C_cost = 0
        A_return = 0
        B_return = 0
        C_return = 0
        temp = 0

        # This parameter has been used on a former version of the reward function. On the present version of the program, it only
        # serves to help calculate one of the program's output for informative purposes (percentage returns).
        cost_reward = 1.0

        if (action == 'buyA'):

            if (self.portfolio['longA'] >= 0): # i.e., the agent wants to buy A and it is already bought on A. Thus it will open a new position.
                # Whenever a position is OPENED, costs are computed
                A_cost = 2 * buy_volume
                self.portfolio['a_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_A])
                self.portfolio['a_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_A])
                self.portfolio['a_price'].append(self.prices.ix[self.date, self.stock_A])
                self.portfolio['a_price'].append(self.prices.ix[self.date, self.stock_A])
                self.portfolio['longA'] += 2
                pct_return_A = -2.0 * cost_reward
                self.cum_operations += 2
                

            else: #longA < 0, i.e., the agent wants to buy A but it is sold on A. Thus it will close a (short) position.
                # Whenever a position is CLOSED, returns are computed
                short_initial_1 = buy_volume
                abs_return_A = buy_volume - self.portfolio['a_vol'][-1] * self.prices.ix[self.date, self.stock_A]
                A_return = buy_volume
                A_return += (buy_volume - self.portfolio['a_vol'].pop() * self.prices.ix[self.date, self.stock_A])
                self.portfolio['a_price'].pop()
                pct_return_A = float(abs_return_A)/short_initial_1
                self.portfolio['longA'] += 1

                if (self.portfolio['longA'] >= 0):

                    A_cost = buy_volume
                    self.portfolio['a_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_A])
                    self.portfolio['a_price'].append(self.prices.ix[self.date, self.stock_A])
                    self.portfolio['longA'] += 1
                    pct_return_A += -1.0 * cost_reward
                    self.cum_operations += 1
                    

                else: #longA < 0

                    short_initial_2 = buy_volume
                    abs_return_A = (buy_volume - self.portfolio['a_vol'][-1] * self.prices.ix[self.date, self.stock_A])
                    temp = buy_volume
                    temp += (buy_volume - self.portfolio['a_vol'].pop() * self.prices.ix[self.date, self.stock_A])
                    A_return += temp
                    self.portfolio['a_price'].pop()
                    #pct_return_A = float(abs_return_A)/(short_initial_1 + short_initial_2)
                    pct_return_A += float(abs_return_A)/(short_initial_2)
                    self.portfolio['longA'] += 1

            if (self.portfolio['longB'] > 0):

                long_initial = buy_volume
                B_return = self.portfolio['b_vol'].pop() * self.prices.ix[self.date, self.stock_B]
                abs_return_B = B_return - long_initial
                pct_return_B = float(abs_return_B)/long_initial
                self.portfolio['b_price'].pop()
                self.portfolio['longB'] -= 1

            else: #longB <= 0

                B_cost = buy_volume
                self.portfolio['b_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_B])
                self.portfolio['b_price'].append(self.prices.ix[self.date, self.stock_B])
                self.portfolio['longB'] -= 1
                pct_return_B = -1.0 * cost_reward
                self.cum_operations += 1
                

            if (self.portfolio['longC'] > 0):

                long_initial = buy_volume
                C_return = self.portfolio['c_vol'].pop() * self.prices.ix[self.date, self.stock_C]
                abs_return_C = C_return - long_initial
                pct_return_C = float(abs_return_C)/long_initial
                self.portfolio['c_price'].pop()
                self.portfolio['longC'] -= 1

            else: #longC <= 0

                C_cost = buy_volume
                self.portfolio['c_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_C])
                self.portfolio['c_price'].append(self.prices.ix[self.date, self.stock_C])
                self.portfolio['longC'] -= 1
                pct_return_C = -1.0 * cost_reward
                self.cum_operations += 1
                

        elif (action == 'sellA'):

            if (self.portfolio['longA'] > 0):

                long_initial_1 = buy_volume
                A_return = self.portfolio['a_vol'].pop() * self.prices.ix[self.date, self.stock_A]
                abs_return_A = A_return - long_initial_1
                pct_return_A = float(abs_return_A)/long_initial_1
                self.portfolio['a_price'].pop()
                self.portfolio['longA'] -= 1

                if (self.portfolio['longA'] > 0):

                    long_initial_2 = buy_volume
                    temp = self.portfolio['a_vol'].pop() * self.prices.ix[self.date, self.stock_A]
                    A_return += temp
                    abs_return_A = (A_return - long_initial_2)
                    #pct_return_A = float(abs_return_A)/(long_initial_1 + long_initial_2)
                    pct_return_A += float(abs_return_A)/(long_initial_2)
                    self.portfolio['a_price'].pop()
                    self.portfolio['longA'] -= 1

                else: #longA <= 0

                    A_cost = buy_volume
                    self.portfolio['a_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_A])
                    self.portfolio['a_price'].append(self.prices.ix[self.date, self.stock_A])
                    self.portfolio['longA'] -= 1
                    pct_return_A += -1.0 * cost_reward
                    self.cum_operations += 1
                    

            else: #longA <= 0

                A_cost = 2 * buy_volume
                self.portfolio['a_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_A])
                self.portfolio['a_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_A])
                self.portfolio['a_price'].append(self.prices.ix[self.date, self.stock_A])
                self.portfolio['a_price'].append(self.prices.ix[self.date, self.stock_A])
                self.portfolio['longA'] -= 2
                pct_return_A = -2.0 * cost_reward
                self.cum_operations += 2
                

            
            if (self.portfolio['longB'] >= 0):

                B_cost = buy_volume
                self.portfolio['b_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_B])
                self.portfolio['b_price'].append(self.prices.ix[self.date, self.stock_B])
                self.portfolio['longB'] += 1
                pct_return_B = -1.0 * cost_reward
                self.cum_operations += 1
                

            else: #longB < 0

                short_initial = buy_volume
                abs_return_B = buy_volume - self.portfolio['b_vol'][-1] * self.prices.ix[self.date, self.stock_B]
                B_return = buy_volume
                B_return += (buy_volume - self.portfolio['b_vol'].pop() * self.prices.ix[self.date, self.stock_B])
                self.portfolio['b_price'].pop()
                pct_return_B = float(abs_return_B)/short_initial
                self.portfolio['longB'] += 1

            if (self.portfolio['longC'] >= 0):

                C_cost = buy_volume
                self.portfolio['c_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_C])
                self.portfolio['c_price'].append(self.prices.ix[self.date, self.stock_C])
                self.portfolio['longC'] += 1
                pct_return_C = -1.0 * cost_reward
                self.cum_operations += 1
                

            else: #longC < 0

                short_initial = buy_volume
                abs_return_C = buy_volume - self.portfolio['c_vol'][-1] * self.prices.ix[self.date, self.stock_C]
                C_return = buy_volume
                C_return += (buy_volume - self.portfolio['c_vol'].pop() * self.prices.ix[self.date, self.stock_C])
                self.portfolio['c_price'].pop()
                pct_return_C = float(abs_return_C)/short_initial
                self.portfolio['longC'] += 1





        elif (action == 'buyB'):

            if (self.portfolio['longB'] >= 0):

                B_cost = 2 * buy_volume
                self.portfolio['b_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_B])
                self.portfolio['b_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_B])
                self.portfolio['b_price'].append(self.prices.ix[self.date, self.stock_B])
                self.portfolio['b_price'].append(self.prices.ix[self.date, self.stock_B])
                self.portfolio['longB'] += 2
                pct_return_B = -2.0 * cost_reward
                self.cum_operations += 2
                

            else: #longB < 0 

                short_initial_1 = buy_volume
                abs_return_B = buy_volume - self.portfolio['b_vol'][-1] * self.prices.ix[self.date, self.stock_B]
                B_return = buy_volume
                B_return += (buy_volume - self.portfolio['b_vol'].pop() * self.prices.ix[self.date, self.stock_B])
                self.portfolio['b_price'].pop()
                pct_return_B = float(abs_return_B)/short_initial_1
                self.portfolio['longB'] += 1

                if (self.portfolio['longB'] >= 0):

                    B_cost = buy_volume
                    self.portfolio['b_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_B])
                    self.portfolio['b_price'].append(self.prices.ix[self.date, self.stock_B])
                    self.portfolio['longB'] += 1
                    pct_return_B += -1.0 * cost_reward
                    self.cum_operations += 1
                    

                else: #longB < 0

                    short_initial_2 = buy_volume
                    abs_return_B = (buy_volume - self.portfolio['b_vol'][-1] * self.prices.ix[self.date, self.stock_B])
                    temp = buy_volume
                    temp += (buy_volume - self.portfolio['b_vol'].pop() * self.prices.ix[self.date, self.stock_B])
                    B_return += temp
                    self.portfolio['b_price'].pop()
                    #pct_return_B = float(abs_return_B)/(short_initial_1 + short_initial_2)
                    pct_return_B += float(abs_return_B)/(short_initial_2)
                    self.portfolio['longB'] += 1

            if (self.portfolio['longA'] > 0):

                long_initial = buy_volume
                A_return = self.portfolio['a_vol'].pop() * self.prices.ix[self.date, self.stock_A]
                abs_return_A = A_return - long_initial
                pct_return_A = float(abs_return_A)/long_initial
                self.portfolio['a_price'].pop()
                self.portfolio['longA'] -= 1

            else: #longA <= 0

                A_cost = buy_volume
                self.portfolio['a_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_A])
                self.portfolio['a_price'].append(self.prices.ix[self.date, self.stock_A])
                self.portfolio['longA'] -= 1
                pct_return_A = -1.0 * cost_reward
                self.cum_operations += 1
                

            if (self.portfolio['longC'] > 0):

                long_initial = buy_volume
                C_return = self.portfolio['c_vol'].pop() * self.prices.ix[self.date, self.stock_C]
                abs_return_C = C_return - long_initial
                pct_return_C = float(abs_return_C)/long_initial
                self.portfolio['c_price'].pop()
                self.portfolio['longC'] -= 1

            else: #longC <= 0

                C_cost = buy_volume
                self.portfolio['c_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_C])
                self.portfolio['c_price'].append(self.prices.ix[self.date, self.stock_C])
                self.portfolio['longC'] -= 1
                pct_return_C = -1.0 * cost_reward
                self.cum_operations += 1
                

        elif (action == 'sellB'):

            if (self.portfolio['longB'] > 0):

                long_initial_1 = buy_volume
                B_return = self.portfolio['b_vol'].pop() * self.prices.ix[self.date, self.stock_B]
                abs_return_B = B_return - long_initial_1
                pct_return_B = float(abs_return_B)/long_initial_1
                self.portfolio['b_price'].pop()
                self.portfolio['longB'] -= 1

                if (self.portfolio['longB'] > 0):

                    long_initial_2 = buy_volume
                    temp = self.portfolio['b_vol'].pop() * self.prices.ix[self.date, self.stock_B]
                    B_return += temp
                    abs_return_B = (B_return - long_initial_2)
                    #pct_return_B = float(abs_return_B)/(long_initial_1 + long_initial_2)
                    pct_return_B += float(abs_return_B)/(long_initial_2)
                    self.portfolio['b_price'].pop()
                    self.portfolio['longB'] -= 1

                else: #longB <= 0

                    B_cost = buy_volume
                    self.portfolio['b_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_B])
                    self.portfolio['b_price'].append(self.prices.ix[self.date, self.stock_B])
                    self.portfolio['longB'] -= 1
                    pct_return_B += -1.0 * cost_reward
                    self.cum_operations += 1
                    

            else: #longB <= 0

                B_cost = 2 * buy_volume
                self.portfolio['b_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_B])
                self.portfolio['b_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_B])
                self.portfolio['b_price'].append(self.prices.ix[self.date, self.stock_B])
                self.portfolio['b_price'].append(self.prices.ix[self.date, self.stock_B])
                self.portfolio['longB'] -= 2
                pct_return_B = -2.0 * cost_reward
                self.cum_operations += 2
                

            
            if (self.portfolio['longA'] >= 0):

                A_cost = buy_volume
                self.portfolio['a_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_A])
                self.portfolio['a_price'].append(self.prices.ix[self.date, self.stock_A])
                self.portfolio['longA'] += 1
                pct_return_A = -1.0 * cost_reward
                self.cum_operations += 1
                

            else: #longA < 0

                short_initial = buy_volume
                abs_return_A = buy_volume - self.portfolio['a_vol'][-1] * self.prices.ix[self.date, self.stock_A]
                A_return = buy_volume
                A_return += (buy_volume - self.portfolio['a_vol'].pop() * self.prices.ix[self.date, self.stock_A])
                self.portfolio['a_price'].pop()
                pct_return_A = float(abs_return_A)/short_initial
                self.portfolio['longA'] += 1

            if (self.portfolio['longC'] >= 0):

                C_cost = buy_volume
                self.portfolio['c_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_C])
                self.portfolio['c_price'].append(self.prices.ix[self.date, self.stock_C])
                self.portfolio['longC'] += 1
                pct_return_C = -1.0 * cost_reward
                self.cum_operations += 1
                

            else: #longC < 0

                short_initial = buy_volume
                abs_return_C = buy_volume - self.portfolio['c_vol'][-1] * self.prices.ix[self.date, self.stock_C]
                C_return = buy_volume
                C_return += (buy_volume - self.portfolio['c_vol'].pop() * self.prices.ix[self.date, self.stock_C])
                self.portfolio['c_price'].pop()
                pct_return_C = float(abs_return_C)/short_initial
                self.portfolio['longC'] += 1






        elif (action == 'buyC'):

            if (self.portfolio['longC'] >= 0):

                C_cost = 2 * buy_volume
                self.portfolio['c_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_C])
                self.portfolio['c_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_C])
                self.portfolio['c_price'].append(self.prices.ix[self.date, self.stock_C])
                self.portfolio['c_price'].append(self.prices.ix[self.date, self.stock_C])
                self.portfolio['longC'] += 2
                pct_return_C = -2.0 * cost_reward
                self.cum_operations += 2
                

            else: #longC < 0 

                short_initial_1 = buy_volume
                abs_return_C = buy_volume - self.portfolio['c_vol'][-1] * self.prices.ix[self.date, self.stock_C]
                C_return = buy_volume
                C_return += (buy_volume - self.portfolio['c_vol'].pop() * self.prices.ix[self.date, self.stock_C])
                self.portfolio['c_price'].pop()
                pct_return_C = float(abs_return_C)/short_initial_1
                self.portfolio['longC'] += 1

                if (self.portfolio['longC'] >= 0):

                    C_cost = buy_volume
                    self.portfolio['c_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_C])
                    self.portfolio['c_price'].append(self.prices.ix[self.date, self.stock_C])
                    self.portfolio['longC'] += 1
                    pct_return_C += -1.0 * cost_reward
                    self.cum_operations += 1
                    

                else: #longC < 0

                    short_initial_2 = buy_volume
                    abs_return_C = (buy_volume - self.portfolio['c_vol'][-1] * self.prices.ix[self.date, self.stock_C])
                    temp = buy_volume
                    temp += (buy_volume - self.portfolio['c_vol'].pop() * self.prices.ix[self.date, self.stock_C])
                    C_return += temp
                    self.portfolio['c_price'].pop()
                    #pct_return_C = float(abs_return_C)/(short_initial_1 + short_initial_2)
                    pct_return_C += float(abs_return_C)/(short_initial_2)
                    self.portfolio['longC'] += 1

            if (self.portfolio['longA'] > 0):

                long_initial = buy_volume
                A_return = self.portfolio['a_vol'].pop() * self.prices.ix[self.date, self.stock_A]
                abs_return_A = A_return - long_initial
                pct_return_A = float(abs_return_A)/long_initial
                self.portfolio['a_price'].pop()
                self.portfolio['longA'] -= 1

            else: #longA <= 0

                A_cost = buy_volume
                self.portfolio['a_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_A])
                self.portfolio['a_price'].append(self.prices.ix[self.date, self.stock_A])
                self.portfolio['longA'] -= 1
                pct_return_A = -1.0 * cost_reward
                self.cum_operations += 1
                

            if (self.portfolio['longB'] > 0):

                long_initial = buy_volume
                B_return = self.portfolio['b_vol'].pop() * self.prices.ix[self.date, self.stock_B]
                abs_return_B = B_return - long_initial
                pct_return_B = float(abs_return_B)/long_initial
                self.portfolio['b_price'].pop()
                self.portfolio['longB'] -= 1

            else: #longB <= 0

                B_cost = buy_volume
                self.portfolio['b_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_B])
                self.portfolio['b_price'].append(self.prices.ix[self.date, self.stock_B])
                self.portfolio['longB'] -= 1
                pct_return_B = -1.0 * cost_reward
                self.cum_operations += 1
                

        elif (action == 'sellC'):

            if (self.portfolio['longC'] > 0):

                long_initial_1 = buy_volume
                C_return = self.portfolio['c_vol'].pop() * self.prices.ix[self.date, self.stock_C]
                abs_return_C = C_return - long_initial_1
                pct_return_C = float(abs_return_C)/long_initial_1
                self.portfolio['c_price'].pop()
                self.portfolio['longC'] -= 1

                if (self.portfolio['longC'] > 0):

                    long_initial_2 = buy_volume
                    temp = self.portfolio['c_vol'].pop() * self.prices.ix[self.date, self.stock_C]
                    C_return += temp
                    abs_return_C = (C_return - long_initial_2)
                    #pct_return_C = float(abs_return_C)/(long_initial_1 + long_initial_2)
                    pct_return_C += float(abs_return_C)/(long_initial_2)
                    self.portfolio['c_price'].pop()
                    self.portfolio['longC'] -= 1

                else: #longC <= 0

                    C_cost = buy_volume
                    self.portfolio['c_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_C])
                    self.portfolio['c_price'].append(self.prices.ix[self.date, self.stock_C])
                    self.portfolio['longC'] -= 1
                    pct_return_C += -1.0 * cost_reward
                    self.cum_operations += 1
                    

            else: #longC <= 0

                C_cost = 2 * buy_volume
                self.portfolio['c_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_C])
                self.portfolio['c_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_C])
                self.portfolio['c_price'].append(self.prices.ix[self.date, self.stock_C])
                self.portfolio['c_price'].append(self.prices.ix[self.date, self.stock_C])
                self.portfolio['longC'] -= 2
                pct_return_C = -2.0 * cost_reward
                self.cum_operations += 2
                

            
            if (self.portfolio['longA'] >= 0):

                A_cost = buy_volume
                self.portfolio['a_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_A])
                self.portfolio['a_price'].append(self.prices.ix[self.date, self.stock_A])
                self.portfolio['longA'] += 1
                pct_return_A = -1.0 * cost_reward
                self.cum_operations += 1
                
            else: #longA < 0

                short_initial = buy_volume
                abs_return_A = buy_volume - self.portfolio['a_vol'][-1] * self.prices.ix[self.date, self.stock_A]
                A_return = buy_volume
                A_return += (buy_volume - self.portfolio['a_vol'].pop() * self.prices.ix[self.date, self.stock_A])
                self.portfolio['a_price'].pop()
                pct_return_A = float(abs_return_A)/short_initial
                self.portfolio['longA'] += 1
        
            if (self.portfolio['longB'] >= 0):

                B_cost = buy_volume
                self.portfolio['b_vol'].append(buy_volume/self.prices.ix[self.date, self.stock_B])
                self.portfolio['b_price'].append(self.prices.ix[self.date, self.stock_B])
                self.portfolio['longB'] += 1
                pct_return_B = -1.0 * cost_reward
                self.cum_operations += 1
                

            else: #longB < 0

                short_initial = buy_volume
                abs_return_B = buy_volume - self.portfolio['b_vol'][-1] * self.prices.ix[self.date, self.stock_B]
                B_return = buy_volume
                B_return += (buy_volume - self.portfolio['b_vol'].pop() * self.prices.ix[self.date, self.stock_B])
                self.portfolio['b_price'].pop()
                pct_return_B = float(abs_return_B)/short_initial
                self.portfolio['longB'] += 1


        # The portfolio cash receives the returns of closed positions and pays for the newly opened ones.
        self.portfolio['cash'] = self.portfolio['cash'] + A_return + B_return + C_return - A_cost - B_cost - C_cost

        # This variable accumulates the daily value of the portfolio's cash. In the end of the program's execution it will be used
        # to calculate the average daily value of the cash account
        self.cum_cash += self.portfolio['cash']

        old_port_val = self.port_val

        # This is the portfolio evaluation method chosen for the calculation of the reward function. The stocks in the portfolio
        # are evaluated by the prices they had when the positions were opened. This precludes the simulator of rewarding the agent
        # for an increase in the market value of a specific asset. Otherwise the agent would be reinforced to accumulate 
        # well-performing stocks instead of opening and closing positions pursuant to the pairs trading strategy
        self.port_val = self.port_value()  

        # This is an alternate portfolio evaluation method, based on the assets' current prices. I include it in the output files
        # only for comparison purposes
        self.port_val_market = self.port_value_for_output()

        # The reward function        
        reward = np.tanh( 100*(self.port_val - old_port_val)/self.init_cash   )

        self.cum_reward += reward
        
        self.data_out.append(self.date.isoformat()[0:10] + ',' + str(self.portfolio['cash']) + ',' + str(self.prices.ix[self.date, self.stock_A]) + ',' + str(self.prices.ix[self.date, self.stock_B]) + ',' + str(self.prices.ix[self.date, self.stock_C]) + ',' + action + ',' + str(abs_return_A) + ',' +  str(pct_return_A) + ',' + str(abs_return_B) + ',' + str(pct_return_B) + ',' + str(abs_return_C) + ',' + str(pct_return_C) + ',' + str(self.prices_ibovespa.loc[self.date]) + ',' + str(self.cum_reward/self.dateIdx) + ',' + str(self.port_val) + ',' + str(self.port_val_market) )
        

        self.dateIdx += 1
        if self.dateIdx < len(self.prices.index):
            self.date = self.prices.index[self.dateIdx]

        state = self.get_state(self.date)

        # The following function applies limitations to the closing of certain positions and to the opening of too many ones
        boundary = self.get_boundary()

        # resets the portfolio values when the simulation enters the testing period (in the present version, the 11th year)
        if self.date >= self.dates_range[-1] - dt.timedelta(days=365) and self.check == 0:
            self.portfolio = {'cash': self.init_cash, 'a_vol': [], 'a_price': [], 'b_vol': [], 'b_price': [], 'c_vol': [], 'c_price': [], 'longA': 0, 'longB': 0, 'longC': 0}
            self.check = 1
            self.cum_cash = self.init_cash
            self.date_start = self.dateIdx
            self.cum_operations = 0
        
        
        return (reward, state, self.dateIdx, boundary)

    

    def get_state(self, date):
        """
        returns state of next day's market.
        """
        if date not in self.dates_range:
            if verbose: print('Date was out of bounds.')
            if verbose: print(date)
            exit

        if (date == self.prices.index[-1]):
            file_name = "data_for_vis_%s.csv" % dt.datetime.now().strftime("%H-%M-%S")
            
            file = open(file_name, 'w');
            for line in self.data_out:
                file.write(line);
                file.write('\n')
            file.close()
        # Calculates the mean between stocks A, B and C's normalized prices on the day corresponding to the variable "date".
        prices_mean = np.mean([self.prices.ix[date, self.stock_A]/self.prices.ix[0, self.stock_A] , self.prices.ix[date, self.stock_B]/self.prices.ix[0, self.stock_B] , self.prices.ix[date, self.stock_C]/self.prices.ix[0, self.stock_C]])
        
        # returns state of next day's market, i.e. the difference between each stock's normalized price and the mean of the three
        # normalized prices.
        return [self.prices.ix[date, self.stock_A]/self.prices.ix[0, self.stock_A] - prices_mean,
            self.prices.ix[date, self.stock_B]/self.prices.ix[0, self.stock_B] - prices_mean,
            self.prices.ix[date, self.stock_C]/self.prices.ix[0, self.stock_C] - prices_mean
            ]


    def get_boundary(self):

        boundary = np.array([1,1,1,1,1,1,1])
        default = [0,0]
        max_positions = 6
        min_return = 0

        # Forbids the opening of more than "max_positions" positions in any share and in any direction (long or short). 
        # If max_positions is large enough, no such limits are imposed on the agent.

        if self.portfolio['longA'] >= max_positions:
            boundary[0] = 0
            boundary[3] = 0
            boundary[5] = 0
        if self.portfolio['longA'] <= -max_positions:
            boundary[1] = 0
            boundary[2] = 0
            boundary[4] = 0
        if self.portfolio['longB'] >= max_positions:
            boundary[1] = 0
            boundary[2] = 0
            boundary[5] = 0
        if self.portfolio['longB'] <= -max_positions:
            boundary[0] = 0
            boundary[3] = 0
            boundary[4] = 0
        if self.portfolio['longC'] >= max_positions:
            boundary[1] = 0
            boundary[3] = 0
            boundary[4] = 0
        if self.portfolio['longC'] <= -max_positions:
            boundary[0] = 0
            boundary[2] = 0
            boundary[5] = 0

        a_vol = default + self.portfolio['a_vol']
        b_vol = default + self.portfolio['b_vol']
        c_vol = default + self.portfolio['c_vol']


        # Forbids the closing of operations with returns inferior to the thresold defined by the variable "min_return". In a true
        # life situation the setting of "min_return" may take into consideration the existence of transaction costs. If set to zero
        # it simply precludes the closing of operations with loss.

        if (self.portfolio['longA']<0)*(self.buy_volume - a_vol[-1] * self.prices.ix[self.date, self.stock_A]) + (self.portfolio['longA']+1<0)*(self.buy_volume - a_vol[-2] * self.prices.ix[self.date, self.stock_A]) + (self.portfolio['longB']>0) * (b_vol[-1] * self.prices.ix[self.date, self.stock_B]-self.buy_volume) + (self.portfolio['longC']>0)*(c_vol[-1] * self.prices.ix[self.date, self.stock_C]-self.buy_volume) < min_return:
            boundary[0] = 0

        if (self.portfolio['longA']>0)*(a_vol[-1] * self.prices.ix[self.date, self.stock_A]-self.buy_volume) + (self.portfolio['longA']-1>0)*(a_vol[-2] * self.prices.ix[self.date, self.stock_A]-self.buy_volume) + (self.portfolio['longB']<0)*(self.buy_volume - b_vol[-1] * self.prices.ix[self.date, self.stock_B]) + (self.portfolio['longC']<0)*(self.buy_volume - c_vol[-1] * self.prices.ix[self.date, self.stock_C]) < min_return:
            boundary[1] = 0

        if (self.portfolio['longB']<0)*(self.buy_volume - b_vol[-1] * self.prices.ix[self.date, self.stock_B]) + (self.portfolio['longB']+1<0)*(self.buy_volume - b_vol[-2] * self.prices.ix[self.date, self.stock_B]) + (self.portfolio['longA']>0) * (a_vol[-1] * self.prices.ix[self.date, self.stock_A]-self.buy_volume) + (self.portfolio['longC']>0)*(c_vol[-1] * self.prices.ix[self.date, self.stock_C]-self.buy_volume) < min_return:
            boundary[2] = 0

        if (self.portfolio['longB']>0)*(b_vol[-1] * self.prices.ix[self.date, self.stock_B]-self.buy_volume) + (self.portfolio['longB']-1>0)*(b_vol[-2] * self.prices.ix[self.date, self.stock_B]-self.buy_volume) + (self.portfolio['longA']<0)*(self.buy_volume - a_vol[-1] * self.prices.ix[self.date, self.stock_A]) + (self.portfolio['longC']<0)*(self.buy_volume - c_vol[-1] * self.prices.ix[self.date, self.stock_C]) < min_return:
            boundary[3] = 0

        if (self.portfolio['longC']<0)*(self.buy_volume - c_vol[-1] * self.prices.ix[self.date, self.stock_C]) + (self.portfolio['longC']+1<0)*(self.buy_volume - c_vol[-2] * self.prices.ix[self.date, self.stock_C]) + (self.portfolio['longA']>0)*(a_vol[-1] * self.prices.ix[self.date, self.stock_A]-self.buy_volume) + (self.portfolio['longB']>0)*(b_vol[-1] * self.prices.ix[self.date, self.stock_B]-self.buy_volume) < min_return:
            boundary[4] = 0

        if (self.portfolio['longC']>0)*(c_vol[-1] * self.prices.ix[self.date, self.stock_C]-self.buy_volume) + (self.portfolio['longC']-1>0)*(c_vol[-2] * self.prices.ix[self.date, self.stock_C]-self.buy_volume) + (self.portfolio['longA']<0)*(self.buy_volume - a_vol[-1] * self.prices.ix[self.date, self.stock_A]) + (self.portfolio['longB']<0)*(self.buy_volume - b_vol[-1] * self.prices.ix[self.date, self.stock_B]) < min_return:
            boundary[5] = 0


        return boundary

    # calculates portfolio based on the prices of acquisition
    def port_value(self):
        value = self.portfolio['cash']
        value += self.buy_volume * abs(self.portfolio['longA'])
        value += self.buy_volume * abs(self.portfolio['longB'])
        value += self.buy_volume * abs(self.portfolio['longC'])
        return value

    # calculates portfolio based on current market prices
    def port_value_for_output(self):
        buy_volume = self.buy_volume
        value = self.portfolio['cash']
        
        if (self.portfolio['longA'] > 0):
            for i in range(len(self.portfolio['a_vol'])):
                value += (self.portfolio['a_vol'][i] * self.prices.ix[self.date, self.stock_A])
        
        if (self.portfolio['longA'] < 0):
            for i in range(len(self.portfolio['a_vol'])):
                value += buy_volume
                value += (buy_volume - self.portfolio['a_vol'][i] * self.prices.ix[self.date, self.stock_A])
        
        if (self.portfolio['longB'] > 0):
            for i in range(len(self.portfolio['b_vol'])):
                value += (self.portfolio['b_vol'][i] * self.prices.ix[self.date, self.stock_B])
            
        if (self.portfolio['longB'] < 0):
            for i in range(len(self.portfolio['b_vol'])):
                value += buy_volume
                value += (buy_volume - self.portfolio['b_vol'][i] * self.prices.ix[self.date, self.stock_B])

        if (self.portfolio['longC'] > 0):
            for i in range(len(self.portfolio['c_vol'])):
                value += (self.portfolio['c_vol'][i] * self.prices.ix[self.date, self.stock_C])
        
        if (self.portfolio['longC'] < 0):
            for i in range(len(self.portfolio['c_vol'])):
                value += buy_volume
                value += (buy_volume - self.portfolio['c_vol'][i] * self.prices.ix[self.date, self.stock_C])

        return value


    def has_more(self):
        if ((self.dateIdx < len(self.prices.index)) == False):
            print('\n\n\n*****')
            # Average daily cash account
            print(self.cum_cash/(self.dateIdx - self.date_start + 1) )
            print('*****\n\n\n')
            # Final portfolio value in the testing year
            print(self.port_val)
            print('*****\n\n\n')
            # Number of positions opened in the testing year
            print(self.cum_operations)
            print('*****\n\n\n')
        return self.dateIdx < len(self.prices.index)
