"""
2-input XOR example -- this is most likely the simplest possible example.
"""

import os
from pyboy import PyBoy

import neat
import visualize

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]
def press_buttons(game_instance, output, output_names):
    for out, name in zip(output,  output_names):
        if out:
            game_instance.button_press(name)
        else:
            game_instance.button_release(name)

def eval_genomes(genomes, config):
    
    pyboy = PyBoy("galaga.gb", window="null")
    pyboy.load_state(open('galaga.gb.state', 'rb'))
    for genome_id, genome in genomes:
        print("running genome:", genome_id)
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        button_config = ["left", "right", "a"]
        for x in range (500):
        #while pyboy.tick():
            pyboy.tick()
            flat_list = [item for sublist in pyboy.game_area() for item in sublist]
            output = net.activate(flat_list)
            button_output = [round(x) for x in output]
            #print(button_output, ": ", genome_id)
            press_buttons(pyboy, button_output, button_config)

        pyboy.load_state(open('galaga.gb.state', 'rb'))


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # for xi, xo in zip(xor_inputs, xor_outputs):
    #     output = winner_net.activate(xi)
    #     print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    # node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
    # visualize.draw_net(config, winner, True, node_names=node_names)
    # visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)

    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    #p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward-partial-ga')
    run(config_path)
