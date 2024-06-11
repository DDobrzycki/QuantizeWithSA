import argparse
import os
import pickle

from fxpmath import Fxp

from quantizer.quantizer import Quantizer
from utils import set_global_determinism
from utils.representation import (plot_gaussian_accuracy,
                                  plot_gaussian_avg_bits, see_annealing,
                                  see_weights_cost)


def main(args):
    final_acc_sim = []
    final_avg_sim = []

    quantizer = Quantizer(
        args.model_name, args.max_steps,
        args.min_frac_bits, args.max_frac_bits,
        args.max_degradation, args.alpha,
        args.beta, args.gamma,
        args.seed)
    
    output_dir = os.path.join("./output_simulations", f"{args.number_simulations}_sims_max_steps_{args.max_steps}")
    os.makedirs(output_dir, exist_ok=True)

    for _ in range(args.number_simulations):
        quantizer.build_model()

        state, c, states, costs = quantizer.annealing(
            maxsteps=args.max_steps,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            debug=True)
        
        states.append(state.copy())
        states = states[2:]
        costs.append(c)
        costs = costs[2:]

        # Compute accuracy after simulated annealing
        fxp_by_layer = [Fxp(None, signed=True, n_int=quantizer.n_int, n_frac=frac_bits) for frac_bits in state]
        
        for i, layer in enumerate(quantizer.layers_of_interest):
            weights = [Fxp(w, like=fxp_by_layer[i]) for w in quantizer.w_dict[layer.name]]
            quantizer.model.get_layer(layer.name).set_weights(weights)

        actual_acc = quantizer.model.evaluate(quantizer.x_test, quantizer.y_test, verbose=0)[1]


        see_annealing(states, costs, state, args.alpha, args.beta, args.gamma,
                     quantizer.lower_bound, quantizer.factor, actual_acc,
                     args.number_simulations, args.max_steps, 
                     quantizer.time_stamp, c)
        
        see_weights_cost(quantizer.weights_cost, args.number_simulations, args.max_steps, quantizer.time_stamp)

        final_acc_sim.append(actual_acc)
        final_avg_sim.append(sum(state) / len(state))

    with open(os.path.join(output_dir, f'final_acc_sim_{args.number_simulations}_sims_max_steps_{args.max_steps}.pkl'), 'wb') as file:
        pickle.dump(final_acc_sim, file)

    with open(os.path.join(output_dir, f'final_avg_sim_{args.number_simulations}_sims_max_steps_{args.max_steps}.pkl'), 'wb') as file:
        pickle.dump(final_avg_sim, file)

    if len(final_avg_sim) > 1:
        plot_gaussian_avg_bits(
            final_avg_sim, args.alpha, args.beta,
            args.gamma, args.number_simulations, args.max_steps, quantizer.time_stamp
        )
        plot_gaussian_accuracy(
            final_acc_sim, args.alpha, args.beta, args.gamma,
            quantizer.lower_bound, quantizer.factor, args.number_simulations, args.max_steps, quantizer.time_stamp
        )

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Cuantize the desired model employing fixed point precision employing the simulated annealing algorithm')

    parser.add_argument(
        "--model_name", "-m", choices=["cnn_mnist", "convnet_js", "custom"], default="cnn_mnist", required=True,
        help="Model to cuantize. If custom, it must be implemented first")
    parser.add_argument("--number_simulations", "-ns", type=int, default=1,
                        help="Nº of simulations. Set > 1 if want multiple cuantization results.")
    parser.add_argument("--max_steps", "-ms", type=int, default=100,
                        help="Maximum number of steps of Simulated Annealing convergence algorithm.")

    parser.add_argument("--min_frac_bits", "-minb", type=int, default=1,
                        help="Lower search range when simulating the quantification of the fractional part of the parameters.")

    parser.add_argument("--max_frac_bits", "-maxb", type=int, default=15,
                        help="Upper search range when simulating the quantification of the fractional part of the parameters.")

    parser.add_argument("--max_degradation", "-md",
                        type=float, default=5,
                        help="Maximum degradation accepeted on models accuracy")

    # ---------------------------Convergence guidance hyperparameters
    # Cost function=gamma*((lower_bound-actual_acc)**2) + beta*avg_bits -alpha*lower_bound

    parser.add_argument("--alpha", "-a", type=float, default=0,
                        help="Related with the importance of the lower bound")

    parser.add_argument("--beta", "-b", type=float, default=25,
                        help="Related with the importance of  the average nº of bits employed for the cuantization")

    parser.add_argument("--gamma", "-g", type=float, default=1,
                        help="Related with the importance of diference between the cuantized model accuracy")
    # ---------------------------/Convergence guidance hyperparameters
    parser.add_argument("--seed", "-s", type=int, default=None,
                        help="Seed for global determinism. Helps with replicating experiments but may slow down execution. Default is None.")

    args = parser.parse_args()

    if args.seed is not None:
        set_global_determinism(args.seed)

    main(args)

    print("\n Finishing...")
