import numpy as np
import theano
import theano.tensor as T
import lasagne
import sys

"""
        
        Code based on previous code written by Yichen Shen and Yiding Zhao 
        ( <https://github.com/shenyichen105/Deep-Reinforcement-Learning-in-Stock-Trading> )
        
"""

class RNN(object):

    def __init__(self, seq_len, n_feature):
        self.Input = lasagne.layers.InputLayer(shape=(None, seq_len, n_feature))
        self.buildNetwork()
        self.output = lasagne.layers.get_output(self.network)
        self.params = lasagne.layers.get_all_params(self.network, trainable=True)
        self.output_fn = theano.function([self.Input.input_var], self.output)

        fx = T.fvector().astype("float64")
        choices = T.ivector()
        px = self.output[T.arange(self.output.shape[0]), choices]
        log_px = T.log(px)
        cost = -fx.dot(log_px)
        updates = lasagne.updates.adagrad(cost, self.params, 0.0008)
        Input = lasagne.layers.InputLayer(shape=(None, seq_len, n_feature))
        self.train_fn = theano.function([self.Input.input_var, choices, fx], [cost, px, log_px], updates=updates)
    
    def buildNetwork(self):
        l_forward = lasagne.layers.RecurrentLayer(
            self.Input, 20,
            grad_clipping = 100,
            W_in_to_hid=lasagne.init.HeUniform(),
            W_hid_to_hid=lasagne.init.HeUniform(),
            nonlinearity=lasagne.nonlinearities.tanh
            #only_return_final=True)
            )

        self.network = lasagne.layers.DenseLayer(
            l_forward, num_units=7, nonlinearity=lasagne.nonlinearities.softmax)
    
    def predict(self, x):
        x = np.array([x]).astype(theano.config.floatX)
        prediction = self.output_fn(x)
        return prediction[0]

    def train(self, Input, choices, rewards):
        if rewards.std() != 0:
            rewards = (rewards - rewards.mean()) / rewards.std()
        cost, px, log_px = self.train_fn(Input, choices, rewards)

    def save(self, path='model.npz'):
        np.savez(path, *lasagne.layers.get_all_param_values(self.network))

    def load(self, path='model.npz'):
        with np.load(path) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(network, param_values)


class LSTM(RNN):
    def buildNetwork(self):
        l_forward = lasagne.layers.LSTMLayer(
            self.Input, 20,
            grad_clipping = 10
            )

        self.network = lasagne.layers.DenseLayer(
            l_forward, num_units=7, nonlinearity=lasagne.nonlinearities.softmax)


class ActorAgent(object):
    def __init__(self, n_batch=5, batch_size=40, update_cycle=3, update_target_cycle=8, lookback=[]):
        self.batch_size = batch_size # number of x, y pairs to fit model
        self.update_cycle = update_cycle # cyle to update actor and critics model, i.e. every 3 days
        self.update_target_cycle = update_target_cycle #cycle for update target model
        self.lookback_size = len(lookback)
        self.n_feature = len(lookback[0])
        self.states = lookback[:]
        self.choices = []
        self.rewards = []
        self.model = LSTM(self.lookback_size, len(lookback[0]))

        self.size_of_replay = self.batch_size*4

        #initialize critic network
        self.critic_agent = CriticAgent(batch_size = self.batch_size, critic_update_cycle=self.update_cycle, target_update_cycle = self.update_target_cycle, lookback = lookback)

    def reset(self, lookback):
        # resetting actor
        self.lookback_size = len(lookback)
        self.states = lookback[:]
        self.choices = []
        self.rewards = []

    def init_query(self):
        probs = self.model.predict(self.states)
        self.action = np.random.choice(7, p=probs)
        return self.action

    def query(self, new_state, reward, day, boundary):

        # updating the critic network
        self.critic_agent.update_history(self.states, self.rewards)

        self.choices.append(self.action)
        self.rewards.append(reward)

        if len(self.choices) % self.update_cycle == 0 and len(self.choices) >= self.batch_size * 1.25:
            # updating the actor
            random_sample = np.random.choice(np.arange(max(0,len(self.rewards) - self.size_of_replay - 1), len(self.rewards) - 1), self.batch_size)
            self.update(random_sample)
            # updating the critic
            self.critic_agent.update_critic(random_sample)

        if len(self.choices) % self.update_target_cycle == 0 and len(self.choices) >= self.batch_size * 1.25:
            # updating the target network
            self.critic_agent.update_target()

        self.states.append(new_state)
        probs = self.model.predict(self.states[-self.lookback_size:])

        if np.isnan(probs[0]):
            sys.exit()

        # application of limitations to the closing of certain positions and to the opening of too many ones 
        probs = probs * boundary


        # Implementation of a progressively deterministic agent
        ####################
        # The present implementation was devised with the view of presenting the agent with 11 years of data. During the first
        # 10 years the agent learns and its behavior is governed by a stochastic policy function. In the 11th year it assumes
        # a predominantly deterministic behavior.
        ###################
        # The function epsilon is so devised to guarantee that the agent should fully explore during 10 years and assume almost
        # complete deterministic behavior during the eleventh year.

        epsilon = 1  /   (  ((day/2400)**(8)) + 1    )
        probs2 = (epsilon/7)*probs
        probs2[np.argmax(probs2)]=np.max(probs)*(1-epsilon+epsilon/7)
        probs = probs2/np.sum(probs2)

        self.action = np.random.choice(7, p=probs)

        # update critic
        self.critic_agent.update_history(self.states, self.rewards)

        return self.action

    def update(self, random_sample):
       
        Input = np.zeros((self.batch_size, self.lookback_size, self.n_feature), dtype = theano.config.floatX)
        Input_next = np.zeros((self.batch_size, self.lookback_size, self.n_feature), dtype = theano.config.floatX)
        choices = np.array(self.choices, dtype = np.int32)[random_sample]

        for i in range(self.batch_size):

            random_id = random_sample[i]
            state_cur = np.array(self.states[random_id: random_id+self.lookback_size], dtype = theano.config.floatX)
            state_next = np.array(self.states[random_id+1: random_id+self.lookback_size+1], dtype = theano.config.floatX)

            Input[i,:,:] = state_cur
            Input_next[i,:,:] = state_next
            #td error estimating the advantage function, to multiply policy gradient
        td_advantage = np.array(self.rewards)[random_sample] + 0.9*self.critic_agent.target_batch_query(Input_next) - self.critic_agent.batch_query(Input)
        self.model.train(Input, choices, td_advantage)


from keras.layers import Dense
from keras.layers import LSTM as lstm
from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adagrad, Adam

class model(object):
    def __init__(self, seq_len, n_feature):

        self.seq_len = seq_len
        self.n_feature = n_feature
        self.model = self.buildnetwork()


    def buildnetwork(self):
        model = Sequential()
        model.add(lstm(20, dropout=0.2,input_shape = (self.seq_len, self.n_feature)))
        model.add(Dense(1, activation=None))
        model.compile(loss='mean_squared_error', optimizer=Adagrad(lr=0.002,clipvalue=10), metrics=['mean_squared_error'])

        return model

    def predict(self, x):
        # x is a tensor having shape (1, seq_len, n_feature)
        return self.model.predict(x)

    def get_weights(self):
        return self.model.get_weights()

    def train(self, x_batch, target_batch):
        # x_batch is a tensor having shape (batch_size, seq_len, n_feature)
        # target batch is a tensor having shape (batch_size, 1)
        self.model.train_on_batch(x_batch, target_batch)

class target_model(model):

    def set_weights(self, weights):
        #weights should be returned by "get_weights" method in Critic network
        self.model.set_weights(weights)


class CriticAgent(object):
    def __init__(self, batch_size=40, critic_update_cycle=3, target_update_cycle=8,lookback=[]):
    
        self.batch_size = batch_size

        self.states = lookback[:]
        self.rewards = []

        self.n_feature = len(lookback[0])
        self.lookback_size = len(lookback)

        self.critic_model = model(self.lookback_size, len(lookback[0]))
        self.target_model = target_model(self.lookback_size, len(lookback[0]))

        self.critic_update_cycle = critic_update_cycle
        self.target_update_cycle = target_update_cycle

        self.gamma = 0.9

        self.count = 0

        self.init_lookback = lookback

    def update_critic(self, random_sample):

        states_batch = np.zeros((self.batch_size, self.lookback_size, self.n_feature), dtype = "float32")
        states_next_batch = np.zeros((self.batch_size, self.lookback_size, self.n_feature),dtype = "float32")

        for i in range(self.batch_size):
            random_id = random_sample[i]
            states_batch[i,:,:] =np.array(self.states[random_id:random_id+self.lookback_size]).astype("float32")
            states_next_batch[i,:,:] =np.array(self.states[random_id + 1:(random_id+self.lookback_size +1)]).astype("float32")

        reward_batch = np.array([self.rewards[i] for i in random_sample]).astype("float32")
        #using target model to predict
        target_value = self.target_model.predict(states_next_batch).flatten()*self.gamma + reward_batch

        self.critic_model.train(states_batch, target_value.reshape(self.batch_size,1))


    def query(self, state):
        #query state for one day (shape: (lookback_size, n_feature))
        value = self.critic_model.predict(state.reshape(1,self.lookback_size, self.n_feature))[0][0]
        return value

    def batch_query(self, states):
        values = self.critic_model.predict(states.reshape(self.batch_size, self.lookback_size, self.n_feature))[0]
        return values


    def target_batch_query(self, states):
        values = self.target_model.predict(states.reshape(self.batch_size, self.lookback_size, self.n_feature))[0]
        return values

    def update_target(self):
        weights = self.critic_model.get_weights()
        self.target_model.set_weights(weights)


    def update_history(self, states, rewards):
        self.states = states
        self.rewards = rewards
