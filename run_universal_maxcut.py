import logging
import os
import click
import time
import gym
from gym import spaces
import numpy as np
from dataclasses import dataclass

from qibo.gates import CNOT, RZ, RY, M
from qibo.hamiltonians import SymbolicHamiltonian, Hamiltonian
from qibo.symbols import Z, X
from gym_universal.classical_hamiltonians import MaxCutHamiltonian

from gym.wrappers import Monitor
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


from common import ClickPythonLiteralOption
from agents.paddpg import PADDPGAgent

# device = torch.device("cpu")
@dataclass
class InstanceConfig:

    seed: int = 50000
    episodes: int = 70000
    evaluation_episodes: int = 1
    update_ratio: float = 0.1
    batch_size: int = 32
    gamma: float = 0.99
    beta: float = 0.2
    inverting_gradients: bool = True
    initial_memory_threshold: int = 1000
    use_ornstein_noise: bool = True
    replay_memory_size: int = 80000
    epsilon_steps: int = 1000
    epsilon_final: float = 0.05
    tau: float = 0.0001
    learning_rate_actor: float = 0.002
    learning_rate_critic: float = 0.002
    clip_grad: float = 1.
    n_step_returns: bool = True
    scale_actions: bool = True
    layers = [256,256,256]
    save_dir: str = 'results/MaxCut'
    title: str = 'PADDPG'
    '''
    edges = [[0, 1], [0, 11], [1, 2], [1, 11], [2, 3], [2, 10], [2, 11],
                           [3, 4], [3, 9], [3, 10], [4, 5], [4, 8], [4, 9], [5, 6], 
                           [5, 7], [5, 8], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11],
                           [6,12], [12,13], [7,12], [7,13]]

    edges =  [[0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,7], [0,7], [0,6], [1,6], [1,5], [2,5], [2,4]]
    '''
    
    edges =  [[0,1], [1,2], [2,3], [3,4], [4,5], [0,5], [0,4], [1,4], [1,3]]
    qbits: int = 6
    max_depth: int = 10
    instance_name: str = 'chordal'
    train_id: str = ''

    @property
    def path(self):
        if self.train_id:
            return 'runs/'+self.instance_name+'/'+str(self.qbits)+'_qubits/'+str(config.seed)+'_seed/'+str(config.train_id)+'/'
        else:
            return 'runs/'+self.instance_name+'/'+str(self.qbits)+'_qubits/'+str(config.seed)+'_seed/'
    


def pad_action(act, act_param, qubits):

    if act == 0:
        qubit = np.argmax(act_param[:-1])
        action = RY(qubit, theta=act_param[-1])
    
    elif act == 1:
        qubit = np.argmax(act_param[:-1])
        action = RZ(qubit, theta=act_param[-1])

    elif act == 2:
        control_qubit, target_qubit = tuple(np.argsort(act_param)[-2:])
        action = CNOT(control_qubit, target_qubit)
    
    else:
        action = M(*(q for q in range(qubits)))

    return action


def evaluate(env, agent, qubits, path, episodes=10):
    returns = []
    timesteps = []
    with open(path+'/results.txt', 'w+') as f:
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
                print('Output Action {} with parameters {}'.format(action,act_param), file=f)
                state, reward, terminal, info = env.step(action)
                total_reward += reward
            timesteps.append(t)
            
            print(env.circuit.draw(), file=f)
                
            returns.append(total_reward)
    
        print(f'Energy: {env.current_energy}', file=f)

    return np.column_stack((returns, timesteps))



def run(config: InstanceConfig):
    
    writer = SummaryWriter(log_dir=config.path)
    observables = [SymbolicHamiltonian(-Z(clause[0]) * Z(clause[1])) for clause in config.edges]

    field = 0.0
    hamiltonian = MaxCutHamiltonian(config.edges, field)
    gs = Hamiltonian.from_symbolic(hamiltonian.symbolic_hamiltonian().form, hamiltonian.symbolic_hamiltonian().symbol_map).eigenvalues()
    print('Max Energy {}'.format(gs[0]))

    print('Maximum Cut Attainable {}'.format(hamiltonian.brute_force_solution()[0]))
    max_steps = config.qbits*config.max_depth
    allowed_gates = {RZ, RY, CNOT, M}
    env = gym.make('universal-v1', qbits=config.qbits, shots=1000, allowed_gates=allowed_gates, ham=hamiltonian,
                   obs=observables, max_depth=config.max_depth)

    dir = os.path.join(config.save_dir, config.title)
    env = Monitor(env, directory=os.path.join(dir, str(config.seed)), video_callable=False, write_upon_reset=False, force=True)
    np.random.seed(config.seed)

    new_params = [spaces.Box(low=0, high=2*np.pi, shape=(config.qbits+1,), dtype=np.float32) for i in range(2)]
    new_params += [spaces.Box(low=0, high=2*np.pi, shape=(config.qbits,), dtype=np.float32)]
    new_params += [spaces.Box(low=0, high=2*np.pi, shape=(0,), dtype=np.float32)]

    action_space = spaces.Tuple((spaces.Discrete(len(allowed_gates)),*new_params))

    agent = PADDPGAgent(env.observation_space, action_space,
                        actor_kwargs={'hidden_layers': config.layers, 'init_type': "kaiming", 'init_std': 0.01,
                                      'activation': 'leaky_relu'},
                        critic_kwargs={'hidden_layers': config.layers, 'init_type': "kaiming", 'init_std': 0.01,
                                       'activation': 'leaky_relu'},
                        batch_size=config.batch_size,
                        learning_rate_actor=config.learning_rate_actor,
                        learning_rate_critic=config.learning_rate_critic,
                        gamma=config.gamma,  # 0.99
                        tau_actor=config.tau,
                        tau_critic=config.tau,
                        n_step_returns=config.n_step_returns,
                        epsilon_steps=config.epsilon_steps,
                        epsilon_final=config.epsilon_final,
                        replay_memory_size=config.replay_memory_size,
                        inverting_gradients=config.inverting_gradients,
                        initial_memory_threshold=config.initial_memory_threshold,
                        beta=config.beta,
                        clip_grad=config.clip_grad,
                        use_ornstein_noise=config.use_ornstein_noise,
                        adam_betas=(0.95, 0.999),  # default 0.95,0.999
                        seed=config.seed)
    print(agent)
    total_reward = 0.
    returns = []
    timesteps = []
    start_time_train = time.time()

    for i in tqdm(range(config.episodes)):

        info = {'status': "NOT_SET"}
        state = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)

        act, act_param, all_actions, all_action_parameters = agent.act(state)
        # writer.add_scalars('first step action', dict([(str(j), all_actions[j]) for j in range(all_actions.shape[0])]), i)
        action = pad_action(act, act_param, env.circuit.nqubits)

        episode_reward = 0.
        agent.start_episode()
        transitions = []
        for j in range(max_steps+1):
            ret = env.step(action)
            next_state, reward, terminal, info = ret
            next_state = np.array(next_state, dtype=np.float32, copy=False)

            next_act, next_act_param, next_all_actions, next_all_action_parameters = agent.act(next_state)
            next_action = pad_action(next_act, next_act_param, env.circuit.nqubits)

            # don't add individual steps, so we can calculate n-step returns at the end...
            if config.n_step_returns:
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
                writer.add_scalar('Episode Steps', j, i)
                writer.add_scalar('Energy', env.env.env.current_energy, i)
                writer.add_scalar('Max_Energy', env.env.env.max_energy, i)
                writer.add_scalar('Minimum Depth', env.env.env.min_depth, i)
                writer.add_scalar('Depth', env.env.env.circuit.depth, i)

                break
        agent.end_episode()

        # calculate n-step returns
        if config.n_step_returns:
            nsreturns = compute_n_step_returns(transitions, config.gamma)
            for t, nsr in zip(transitions, nsreturns):
                t.append(nsr)
                agent.replay_memory.append(state=t[0], action=t[1], reward=t[2], next_state=t[3], next_action=t[4],
                                           terminal=t[5], time_steps=None, n_step_return=nsr)

        agent._optimize_td_loss(writer, i)
        writer.add_scalar('Reward', np.array(episode_reward), i)
        writer.add_scalar('Average Entanglement Entropy', np.mean(np.array(info['entropy'])), i)
        writer.add_scalar('Max Entanglement Entropy', np.max(np.array(info['entropy'])), i)

        returns.append(episode_reward)
        timesteps.append(j)

        total_reward += episode_reward
        if i % 100 == 0:
            print('{0:5s} R:{1:.4f} r:{2:.4f}'.format(str(i + 1), total_reward / (i + 1), episode_reward))
    end_time_train = time.time()

    returns = env.get_episode_rewards()
    np.save(os.path.join(dir, config.title + "{}".format(str(config.seed))), np.column_stack((returns, timesteps)))

    if config.evaluation_episodes > 0:
        print("Evaluating agent over {} episodes".format(config.evaluation_episodes))
        agent.epsilon_final = 0.
        agent.epsilon = 0.
        agent.noise = None
        agent.actor.eval()
        agent.critic.eval()
        start_time_eval = time.time()
        evaluation_results = evaluate(env, agent, config.qbits, config.path, config.evaluation_episodes)  # returns, timesteps, goals
        end_time_eval = time.time()
        print("Ave. evaluation return =", sum(evaluation_results[:, 0]) / evaluation_results.shape[0])
        print("Ave. timesteps =", sum(evaluation_results[:, 1]) / evaluation_results.shape[0])
        np.save(os.path.join(dir, config.title + "{}e".format(str(config.seed))), evaluation_results)
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
    config = InstanceConfig(seed=2000000, train_id='initial_superposition', episodes=15000)
    run(config)

