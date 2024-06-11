import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(context="talk", style="darkgrid", palette="hls", font="sans-serif", font_scale=1.05)
FIGSIZE = (25, 8)  #: Figure size, in inches!
mpl.rcParams['figure.figsize'] = FIGSIZE

def see_annealing(states, costs, state, alpha, beta, gamma, lower_bound, factor, accs, actual_acc, test_acc, simulations, max_steps, time_stamp, c):
    plt.figure()

    plt.suptitle("Evolution of states and costs of the simulated annealing")

    plt.subplot(131)
    plt.plot(np.mean(states,1), 'r')
    plt.title(f"States      Final state: {state} -> Avg bits: {sum(state)/len(state):.3f}")
    plt.xlabel('Step')
    plt.ylabel('Avg nº of bits')

    plt.subplot(132)
    plt.plot(accs, 'k')
    plt.axhline(y=test_acc, color='k', linestyle=':', label=f"float32 accuracy: {test_acc:.3f}")
    plt.title("Accuracy evolution over accepted states")
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(133)
    plt.plot(costs, 'b',label=f"\u03B1:{alpha:.2f}, \u03B2:{beta:.2f}, \u03B3:{gamma:.2f}\nLower bound: {lower_bound/factor:.3f}\nFinal acc: {actual_acc:.3f}\nError: {((lower_bound/factor)-actual_acc):.3f} ")
    plt.title(f"Costs           Final state cost: {c:.3f}")
    plt.xlabel('Step')
    plt.ylabel('Cost')

    plt.legend()
    plt.savefig(f'./output_simulations/{simulations}_sims_max_steps_{max_steps}/sim_{max_steps}_steps_{time_stamp}.pdf', format="pdf", bbox_inches="tight")
    plt.close()

def see_weights_cost(weights_cost, simulations, max_steps, time_stamp):
    alpha_cost=[item[2] for item in weights_cost]
    beta_cost=[item[1] for item in weights_cost]
    gamma_cost=[item[0] for item in weights_cost]

    plt.figure()
    plt.plot(alpha_cost,label="\u03B2*lower_bound")
    plt.plot(beta_cost,label="\u03B1*avg_bits")
    plt.plot(gamma_cost,label="\u03B3*(lower_bound-actual_acc)")
    plt.legend()
    plt.savefig(f'./output_simulations/{simulations}_sims_max_steps_{max_steps}/sim_weights_{max_steps}_steps_{time_stamp}.pdf', format="pdf", bbox_inches="tight")
    plt.close()

def plot_gaussian_avg_bits(final_avg_sim, alpha, beta, gamma, simulations, max_steps, time_stamp):
    mean_avg=sum(final_avg_sim)/len(final_avg_sim)
    std_avg=np.std(final_avg_sim) + 0.0000001 # To avoid div. by 0

    s_avg = np.random.normal(mean_avg, std_avg, 1000)
    _, bins, _ = plt.hist(s_avg, 50, density=True, color='r',alpha=0.3)
    plt.plot(bins, 1/(std_avg * np.sqrt(2 * np.pi)) *
                np.exp( - (bins - mean_avg)**2 / (2 * std_avg**2) ),
            linewidth=2, color='r',label=f"μ:{mean_avg:.3f}, \u03c3:{std_avg:.3f} \n\u03B1:{alpha:.2f}, \u03B2:{beta:.2f}, \u03B3:{gamma:.2f}\n")
    plt.title(f"Distribution of the final avg bits with {simulations} simulations ")
    plt.legend()
    plt.savefig(f'./output_simulations/{simulations}_sims_max_steps_{max_steps}/distribution_avg_{max_steps}_steps_{time_stamp}.pdf', format="pdf", bbox_inches="tight")

    plt.close()

def plot_gaussian_accuracy(final_acc_sim, alpha, beta, gamma, lower_bound, factor, simulations, max_steps, time_stamp):
    mean_acc=sum(final_acc_sim)/len(final_acc_sim)
    std_acc=np.std(final_acc_sim) + 0.0000001 # To avoid div. by 0

    s_acc = np.random.normal(mean_acc, std_acc, 1000)
    _, bins, _ = plt.hist(s_acc, 50, density=True, color='b',alpha=0.3)
    plt.plot(bins, 1/(std_acc * np.sqrt(2 * np.pi)) *
                np.exp( - (bins - mean_acc)**2 / (2 * std_acc**2) ),
            linewidth=2, color='b',label=f"μ:{mean_acc:.3f}, \u03c3:{std_acc:.3f} \n\u03B1:{alpha:.2f}, \u03B2:{beta:.2f}, \u03B3:{gamma:.2f}\n")
    plt.axvline(x=lower_bound/factor,label=f"Lower bound: {lower_bound/factor:.3f}",color='k',linestyle='dashed',linewidth=1)
    plt.title(f"Distribution of the final accuracy with {simulations} simulations ")
    plt.legend()
    plt.savefig(f'./output_simulations/{simulations}_sims_max_steps_{max_steps}/distribution_acc_{max_steps}_steps_{time_stamp}.pdf', format="pdf", bbox_inches="tight")
    plt.close()

