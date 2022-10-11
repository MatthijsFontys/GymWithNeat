import gym
import neat
import os
import numpy as np


def main():
    # Neat
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat-config')
    neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                              neat.DefaultStagnation, config_path)

    neat_population = neat.Population(neat_config)
    neat_population.add_reporter(neat.StdOutReporter(False))
    # Uncomment the line below for checkpoint files
    # neat_population.add_reporter(neat.Checkpointer(5))

    # Gym
    env = gym.make("CartPole-v1")
    winner = neat_population.run(lambda genomes, config: run_neat(genomes, config, env, 20_000, 10))
    # Showing the trained model
    env = gym.make("CartPole-v1", render_mode="human")
    run_neat([(1, winner)], neat_config, env, np.inf, 1)
    env.close()


def run_neat(genomes, config, env, limit, runs):
    for _, g in genomes:
        network = neat.nn.FeedForwardNetwork.create(g, config)
        g.fitness = 0
        for i in range(runs):
            observation, info = env.reset()
            terminated = False
            while not terminated:
                output = network.activate(observation)
                action = 0 if output[0] > output[1] else 1
                observation, reward, terminated, truncated, info = env.step(action)
                g.fitness += reward
                # Terminating early in training to prevent perfect model to run forever
                if runs > 1 and (abs(observation[0]) > 1 or g.fitness > limit * (i + 1)):
                    terminated = True


# You might need to run the program a few times to get a 'perfect' model that never leaves the screen
if __name__ == "__main__":
    main()
