"""
2-input XOR example -- this is most likely the simplest possible example.
"""

import os
from pyboy import PyBoy

import neat
import visualize

# 2-input XOR inputs and expected outputs.

# SCORE_WEIGHT = .5
TIME_WEIGHT  = .5
ACCURACY_WEIGHT = 100

def to_int(lst):
    idx = 0
    ret = 0
    for x in lst:
        ret+=x*(10**idx)
        idx+=1
    return ret

def calc_fitness(score,time,shots,hits):
    
    score = to_int(score) * 10
    time = time[0] + (time[1]/60)
    shots = to_int(shots)
    hits =  to_int(hits)
    
    if shots == 0:
        accuracy = 0
    else:  
        accuracy = hits/shots
    
    score_weighted = score
    time_weighted = time * TIME_WEIGHT
    accu_weighted = accuracy*ACCURACY_WEIGHT
    # print(score)
    # print(time)
    # print(accuracy)


    fitness = score_weighted + time_weighted + accu_weighted # max fitness = 1
    
    return fitness

def press_buttons(game_instance, output, output_names):
    for out, name in zip(output,  output_names):
        if out:
            game_instance.button_press(name)
        else:
            game_instance.button_release(name)

def eval_genomes(genomes, config):
    
    pyboy = PyBoy("galaga.gb", window="null")
    pyboy.set_emulation_speed(0)
    pyboy.load_state(open('galaga.gb.state', 'rb'))
    for genome_id, genome in genomes:
        print("running genome:", genome_id)
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        button_config = ["left", "right", "a"]
        while pyboy.memory[0xcc80:0xcc81][0] != 1: # check if genome died
        #while pyboy.tick():
            pyboy.tick()
            flat_list = [item for sublist in pyboy.game_area() for item in sublist]
            #time_elapsed = pyboy.memory[0xcc70:0xcc72]
            lives = pyboy.memory[0xcc80:0xcc81]
            #shot_count = pyboy.memory[0xcc84:0xcc86]
            #kill_count = pyboy.memory[0xcc86:0xcc88]
            #score = pyboy.memory[0xcc7a:0xcc7f]
            #print(time_elapsed, shot_count, kill_count, score)
            #print(lives)
            output = net.activate(flat_list)
            button_output = [round(x) for x in output]
            #print(button_output, ": ", genome_id)
            press_buttons(pyboy, button_output, button_config)
        time_elapsed = pyboy.memory[0xcc70:0xcc72]
        #lives = pyboy.memory[0xcc80:0xcc81]
        shot_count = pyboy.memory[0xcc84:0xcc86]
        kill_count = pyboy.memory[0xcc86:0xcc88]
        score = pyboy.memory[0xcc7a:0xcc7f]
        genome.fitness = calc_fitness(score, time_elapsed, shot_count,kill_count)
        print(genome.fitness)
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
