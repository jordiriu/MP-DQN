import logging
import os
import click
import time
import gym
from gym import spaces
import numpy as np

from qibo.gates import CNOT, RZ, RY, M
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import Z
from gym_universal.classical_hamiltonians import MaxCutHamiltonian

from gym.wrappers import Monitor
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


from common import ClickPythonLiteralOption
from common.soccer_domain import SoccerScaledParameterisedActionWrapper, kill_soccer_server
from agents.paddpg import PADDPGAgent

# Define gate set as Ry,Rz,CNOT

# Number of actions will be 2*N + N*(N-1) + 1 for the measure gate, number of parameters is 2*N


def pad_action(act, act_param, qubits):

    if act == 2*qubits + qubits*(qubits-1):
        return M(*(q for q in range(qubits)))

    gate_type = act//qubits

    if gate_type == 0:
        qubit = act % qubits
        action = RY(qubit, theta=act_param[0])

    elif gate_type == 1:
        qubit = act % qubits
        action = RZ(qubit, theta=act_param[0])

    else:
        target_qubits = []
        for i in range(qubits):
            for j in range(qubits):
                if i != j:
                    target_qubits += [j]

        control_qubit = (act-2*qubits)//(qubits-1)
        target_qubit = target_qubits[act-2*qubits]

        action = CNOT(control_qubit, target_qubit)

    return action


def evaluate(env, agent, qubits, episodes=10):
    returns = []
    timesteps = []
    flag = 0
    for _ in range(episodes):
        state = env.reset()
        terminal = False
        t = 0
        total_reward = 0.
        while not terminal:
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, all_actions, all_action_parameters = agent.act(state)
            action = pad_action(act, act_param, qubits)
            state, reward, terminal, info = env.step(action)
            total_reward += reward
        timesteps.append(t)
        if flag == 0:
            print(env.circuit.draw())
            flag = 1
        returns.append(total_reward)

    return np.column_stack((returns, timesteps))


@click.command()
@click.option('--seed', default=0, help='Random seed.', type=int)
@click.option('--episodes', default=30000, help='Number of epsiodes.', type=int)
@click.option('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
@click.option('--update-ratio', default=0.1, help='Ratio of updates to samples.', type=float)
@click.option('--batch-size', default=32, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.99, help='Discount factor.', type=float)
@click.option('--beta', default=0.2, help='Averaging factor for on-policy and off-policy targets.', type=float)
@click.option('--inverting-gradients', default=True,
              help='Use inverting gradients scheme instead of squashing function.', type=bool)
@click.option('--initial-memory-threshold', default=1000, help='Number of transitions required to start learning.',
              type=int)
@click.option('--use-ornstein-noise', default=False,
              help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
@click.option('--replay-memory-size', default=500000, help='Replay memory size in transitions.', type=int)
@click.option('--epsilon-steps', default=1000, help='Number of episodes over which to linearly anneal epsilon.',
              type=int)
@click.option('--epsilon-final', default=0.1, help='Final epsilon value.', type=float)
@click.option('--tau', default=0.001, help='Soft target network update averaging factor.', type=float)
@click.option('--learning-rate-actor', default=0.001, help="Actor network learning rate.", type=float)
@click.option('--learning-rate-critic', default=0.001, help="Critic network learning rate.", type=float)
@click.option('--clip-grad', default=1., help="Gradient clipping.", type=float)  # default 10
@click.option('--n-step-returns', default=True, help="Use n-step returns.", type=bool)
@click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
@click.option('--layers', default="[1024,512,256,128]", help='Duplicate action-parameter inputs.',
              cls=ClickPythonLiteralOption)
@click.option('--save-dir', default="results/MaxCut", help='Output directory.', type=str)
@click.option('--title', default="PADDPG", help="Prefix of output files", type=str)
def run(seed, episodes, batch_size, gamma, beta, use_ornstein_noise, inverting_gradients, initial_memory_threshold,
        replay_memory_size, tau, learning_rate_actor, learning_rate_critic, epsilon_steps, epsilon_final,
        n_step_returns, clip_grad, scale_actions, layers, evaluation_episodes, update_ratio, save_dir, title):
    writer = SummaryWriter()
    qbits = 8

    counter = 0
    non_ordered_clauses = []
    for i in range(qbits):
        if i < qbits-1:
            clause = [i, i+1]
        else:
            clause = [0, i]
        non_ordered_clauses.append(clause)

    observables = [SymbolicHamiltonian(Z(q)) for q in range(qbits)] + \
                  [SymbolicHamiltonian(-Z(clause[0]) * Z(clause[1]))
                   for clause in non_ordered_clauses]
    hamiltonian = MaxCutHamiltonian(non_ordered_clauses)
    env = gym.make('universal-v1', qbits=qbits, shots=1000, allowed_gates={RZ, RY, CNOT, M}, ham=hamiltonian,
                   obs=observables)

    '''
    if scale_actions:
        env = SoccerScaledParameterisedActionWrapper(env)
    '''

    dir = os.path.join(save_dir, title)
    env = Monitor(env, directory=os.path.join(dir, str(seed)), video_callable=False, write_upon_reset=False, force=True)
    # env.seed(seed)
    np.random.seed(seed)

    new_params = [spaces.Box(low=0, high=2*np.pi, shape=(1,), dtype=np.float32) for i in range(1, 2 * qbits + 1)]
    new_params += [spaces.Box(low=0, high=2*np.pi, shape=(0,), dtype=np.float32) for i in range(1, qbits*(qbits-1) + 2)]

    action_space = spaces.Tuple((spaces.Discrete(2 * qbits + qbits * (qbits - 1) + 1), *new_params))

    agent = PADDPGAgent(env.observation_space, action_space,
                        actor_kwargs={'hidden_layers': layers, 'init_type': "kaiming", 'init_std': 0.01,
                                      'activation': 'leaky_relu'},
                        critic_kwargs={'hidden_layers': layers, 'init_type': "kaiming", 'init_std': 0.01,
                                       'activation': 'leaky_relu'},
                        batch_size=batch_size,
                        learning_rate_actor=learning_rate_actor,  # 0.0001
                        learning_rate_critic=learning_rate_critic,  # 0.001
                        gamma=gamma,  # 0.99
                        tau_actor=tau,
                        tau_critic=tau,
                        n_step_returns=n_step_returns,
                        epsilon_steps=epsilon_steps,
                        epsilon_final=epsilon_final,
                        replay_memory_size=replay_memory_size,
                        inverting_gradients=inverting_gradients,
                        initial_memory_threshold=initial_memory_threshold,
                        beta=beta,
                        clip_grad=clip_grad,
                        use_ornstein_noise=use_ornstein_noise,
                        adam_betas=(0.9, 0.999),  # default 0.95,0.999
                        seed=seed)
    print(agent)
    max_steps = 400
    total_reward = 0.
    returns = []
    timesteps = []
    goals = []
    start_time_train = time.time()

    for i in tqdm(range(episodes)):

        info = {'status': "NOT_SET"}
        state = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)

        act, act_param, all_actions, all_action_parameters = agent.act(state)
        action = pad_action(act, act_param, env.circuit.nqubits)

        episode_reward = 0.
        agent.start_episode()
        transitions = []
        for j in range(max_steps):
            ret = env.step(action)
            next_state, reward, terminal, info = ret
            next_state = np.array(next_state, dtype=np.float32, copy=False)

            next_act, next_act_param, next_all_actions, next_all_action_parameters = agent.act(next_state)
            next_action = pad_action(next_act, next_act_param, env.circuit.nqubits)

            # don't add individual steps, so we can calculate n-step returns at the end...
            if n_step_returns:
                transitions.append(
                    [state, np.concatenate((all_actions.data, all_action_parameters.data)).ravel(), reward,
                     next_state, np.concatenate((next_all_actions.data,
                                                 next_all_action_parameters.data)).ravel(), terminal])
            else:
                agent.step(state, (act, act_param, all_actions, all_action_parameters), reward, next_state,
                           (next_act, next_act_param, next_all_actions, next_all_action_parameters), terminal,
                           optimise=False)

            act, act_param, all_actions, all_action_parameters = next_act, next_act_param, next_all_actions, \
                                                                 next_all_action_parameters
            action = next_action
            state = next_state

            episode_reward += reward
            # env.render()

            if terminal:
                writer.add_scalar('Measurement Weight', all_actions[-1], i)
                writer.add_scalar('Episode Steps', j, i)
                break
        agent.end_episode()

        # calculate n-step returns
        if n_step_returns:
            nsreturns = compute_n_step_returns(transitions, gamma)
            for t, nsr in zip(transitions, nsreturns):
                t.append(nsr)
                agent.replay_memory.append(state=t[0], action=t[1], reward=t[2], next_state=t[3], next_action=t[4],
                                           terminal=t[5], time_steps=None, n_step_return=nsr)

        n_updates = int(update_ratio * j)
        for _ in range(n_updates):
            counter +=1
            agent._optimize_td_loss(writer, counter)

        writer.add_scalar('Reward', episode_reward, i)

        returns.append(episode_reward)
        timesteps.append(j)

        total_reward += episode_reward
        if i % 100 == 0:
            print('{0:5s} R:{1:.4f} r:{2:.4f}'.format(str(i + 1), total_reward / (i + 1), episode_reward))
    end_time_train = time.time()

    returns = env.get_episode_rewards()
    np.save(os.path.join(dir, title + "{}".format(str(seed))), np.column_stack((returns, timesteps)))

    if evaluation_episodes > 0:
        print("Evaluating agent over {} episodes".format(evaluation_episodes))
        agent.epsilon_final = 0.
        agent.epsilon = 0.
        agent.noise = None
        agent.actor.eval()
        agent.critic.eval()
        start_time_eval = time.time()
        evaluation_results = evaluate(env, agent, qbits, evaluation_episodes)  # returns, timesteps, goals
        end_time_eval = time.time()
        print("Ave. evaluation return =", sum(evaluation_results[:, 0]) / evaluation_results.shape[0])
        print("Ave. timesteps =", sum(evaluation_results[:, 1]) / evaluation_results.shape[0])
        np.save(os.path.join(dir, title + "{}e".format(str(seed))), evaluation_results)
        print("Evaluation time: %.2f seconds" % (end_time_eval - start_time_eval))
    print("Training time: %.2f seconds" % (end_time_train - start_time_train))

    print(agent)
    env.close()


def compute_n_step_returns(episode_transitions, gamma):
    n = len(episode_transitions)
    n_step_returns = np.zeros((n,))
    n_step_returns[n - 1] = episode_transitions[n - 1][2]  # Q-value is just the final reward
    for i in range(n - 2, 0, -1):
        reward = episode_transitions[i][2]
        target = n_step_returns[i + 1]
        n_step_returns[i] = reward + gamma * target
    return n_step_returns


if __name__ == '__main__':
    run()

