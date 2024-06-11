import random 
import math
from datetime import datetime

from sklearn.preprocessing import normalize
from fxpmath import Fxp
import numpy as np
import numpy.random as rn

from utils.base_models import Models

class Quantizer(object):
    def __init__(self, model_name, max_steps,
                 min_frac_bits, max_frac_bits, max_degradation,
                 alpha, beta, gamma,
                 seed):
        
        self.model_name = model_name
        self.max_steps = max_steps
        self.min_frac_bits = min_frac_bits
        self.max_frac_bits = max_frac_bits
        self.max_degradation = max_degradation
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.seed = seed

        self.factor = 10
        self.test_acc = 0
        self.n_int = 1

        self.last_i = -1
        self.new_cost = -1
        self.lower_bound = -1
        self.weights_cost = []

        self.model = None
        self.x_test = None
        self.y_test = None

        self.time_stamp = None

        self.args = {}
        self.w_dict = {}
        self.final_acc_sim = []
        self.final_avg_sim = []
        self.layers_of_interest = []

        models_instance = Models()

        self.model_dict = {
        'cnn_mnist': models_instance.cnn_mnist,
        'convnet_js': models_instance.convnet_js,
        'custom': models_instance.custom_model
        }


    def build_model(self):

        self.model, self.x_test, self.y_test = self.model_dict[self.model_name](self.model)

        self.w_dict = {layer.name: layer.get_weights() for layer in self.model.layers}

        self.test_acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)[1]
        self.lower_bound = (self.test_acc-(self.max_degradation/100))*self.factor

        self._get_interested_layers()

    def _get_interested_layers(self) -> None:
        self.layers_of_interest = [layer for layer in self.model.layers if len(layer.get_weights()) > 0]

    def f(self, x, alpha, beta, gamma):
        fxp_by_layer = [Fxp(None, signed=True, n_int=self.n_int, n_frac=x_i) for x_i in x]
        for i, layer in enumerate(self.layers_of_interest):
            layer.set_weights([Fxp(self.w_dict[layer.name][0], like=fxp_by_layer[i]),
                               Fxp(self.w_dict[layer.name][1], like=fxp_by_layer[i])])

        actual_acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)[1] * self.factor
        avg_bits = normalize([[sum(x) / len(self.layers_of_interest), max(self.min_frac_bits, self.max_frac_bits)]])[0][0]
        cost = gamma * ((self.lower_bound - actual_acc) ** 2) + beta * avg_bits - alpha * self.lower_bound
        self.weights_cost.append([gamma * ((self.lower_bound - actual_acc) ** 2) / cost,
                                  beta * avg_bits / cost,
                                  alpha * self.lower_bound / cost])
        return cost

    def clip(self, x, i):  # OK
        """ Force x to be in the interval."""
        a, b = (self.min_frac_bits, self.max_frac_bits)
        x[i] = int(max(min(x[i], b), a))
        return x

    def _random_start(self):  # OK
        """ Random point in the interval."""
        a, b = (self.min_frac_bits, self.max_frac_bits)
        start = []
        for _ in range(0, len(self.layers_of_interest)):
            start.append(int(round(a + (b - a) * rn.random_sample())))

        return start

    def _cost_function(self, x, alpha, beta, gamma):
        """ Cost of x = f(x)."""
        return self.f(x, alpha, beta, gamma)

    def random_neighbour(self, x, T, cost, new_cost):
        """Move a little bit x, from the left or the right."""
        amplitude = int(math.ceil((self.max_frac_bits - self.min_frac_bits) * 0.5 * T))

        i = random.randint(0, len(self.layers_of_interest) - 1) if (cost == new_cost or new_cost > cost) else self.last_i

        delta = amplitude * random.choice([-1, 1])
        x[i] += delta
        return self.clip(x, i), i

    def acceptance_probability(self, cost, new_cost, temperature):
        """
        Calculate the acceptance probability for a new solution in the simulated annealing algorithm.    
        This method evaluates whether to accept a new solution based on its cost relative to the current
        solution's cost, and the current temperature of the system. The acceptance probability helps in
        exploring the solution space effectively by allowing occasional uphill moves, which prevents
        the algorithm from getting stuck in local minima.
        """
        if new_cost < cost:
            print(f"    - Acceptance probabilty = 1 as new_cost = {new_cost} < cost = {cost}...")
            return 1
        else:
            p = np.exp(- (new_cost - cost) / temperature)
            print(f"    - Acceptance probabilty = {p:.3g}...")
            return p

    def temperature(self, fraction):
        """ Example of temperature dicreasing as the process goes on."""
        return max(0.01, min(1, 1 - fraction))

    def annealing(self, maxsteps=100, alpha=1, beta=1, gamma=1, debug=True):
        """ Optimize the black-box function 'cost_function' with the simulated annealing algorithm."""
        self.time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        state = self._random_start()
        cost = self._cost_function(state, alpha, beta, gamma)
        costs = [cost]
        states = [state[:]]
        self.weights_cost = []
        for step in range(maxsteps):
            fraction = step / float(maxsteps)
        
            t = self.temperature(fraction)

            new_state, self.last_i = self.random_neighbour(state[:], t, cost, self.new_cost)
            self.new_cost = self._cost_function(new_state, alpha, beta, gamma)

            states.append(state[:])
            costs.append(cost)

            if debug:
                print(
                    f"|Step #{step}/{maxsteps} : T = {t:.3g}, state = {state}, cost = {cost:.3g}, new_state = {new_state}, new_cost = {self.new_cost:.3g}|".ljust(100, "=")
                    )
            if self.acceptance_probability(cost, self.new_cost, t) > rn.random():
                state, cost = new_state, self.new_cost
                print("  ==> Accept it!")
            else:
                print("  ==> Reject it...")
                
        return state, self._cost_function(state, alpha, beta, gamma), states, costs