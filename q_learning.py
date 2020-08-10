from environment import MountainCar
import sys
import random
import math
import numpy as np


def q_learning(env, mode, weights, bias, num_episodes, max_episode_length, learning_rate, discount_factor, epsilon, returns_out):
    total_step = 0.0
    with open(returns_out, "w") as f:
        f.write("")
    for i in range(num_episodes):
        env_state = env.reset()
        total_rewards = 0
        for step in range(max_episode_length):
            env_state = env.transform(env.state)
            if mode == 'raw':
                state = np.zeros((1, 2), dtype=float)
                state[0][0] = env_state[0]
                state[0][1] = env_state[1]
            else:
                state = np.zeros((1, 2048), dtype=float)
                for i in env_state.keys():
                    state[0][i] = env_state[i]
            # print(state)
            q = np.dot(state, weights) + bias
            explore = random.random() < epsilon
            if explore:
                action = random.randint(0, 2)
            else:
                action = np.argmax(q[0], axis=0)

            new_state, reward, is_terminal = env.step(action)
            next_action = 0
            if mode == 'raw':
                next_state = np.zeros((1, 2), dtype=float)
                next_state[0][0] = new_state[0]
                next_state[0][1] = new_state[1]
            else:
                next_state = np.zeros((1, 2048), dtype=float)
                for i in new_state.keys():
                    next_state[0][i] = new_state[i]
            next_q = np.dot(next_state, weights) + bias
            next_action = np.argmax(next_q[0], axis=0)
            derivative_q = np.zeros((len(weights), len(weights[0])), dtype=float)
            derivative_q[:, action] = state.flatten()
            #print(learning_rate * (q[0][action] - (reward + discount_factor * next_q[0][next_action])) * derivative_q)
            weights = weights - learning_rate * (q[0][action] - (reward + discount_factor * next_q[0][next_action])) * derivative_q
            bias = bias - learning_rate * (q[0][action] - (reward + discount_factor * next_q[0][next_action]))

            total_rewards += reward

            if is_terminal:
                break
        env.render()
        print(str(i) + "\n")
        print(str(total_rewards) + "\n")

        with open(returns_out, "a") as f:
            f.write(str(total_rewards) + "\n")
    return weights, bias


if __name__ == "__main__":
    mode = sys.argv[1]
    weight_out = sys.argv[2]
    returns_out = sys.argv[3]
    episodes = sys.argv[4]
    max_iterations = sys.argv[5]
    epsilon = sys.argv[6]
    gamma = sys.argv[7]
    lr = sys.argv[8]

    bias = 0.0
    if mode == 'raw':
        weights = np.zeros((2, 3), dtype=float)
    else:
        weights = np.zeros((2048, 3), dtype=float)

    env = MountainCar(mode)

    weights, bias = q_learning(env, mode, weights, bias, int(episodes), int(max_iterations), float(lr), float(gamma), float(epsilon), returns_out)
    print(weights)
    print(bias)
    
    with open(weight_out, "w") as f:
        f.write(str(bias) + "\n")
        for i in range(0, len(weights)):
            for j in range(0, len(weights[0])):
                f.write(str(weights[i][j]) + "\n")
