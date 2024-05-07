"""
2-input XOR example -- this is most likely the simplest possible example.
"""
import multiprocessing
import os
from pyboy import PyBoy

import neat
import visualize

# max stag = 4
# pop size = 20

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
    '''if(score>500):
        score*=1.5
    elif(score>1000):
        score*=2
    elif(score>2000):
        score*=4
    elif(score>3000):
        score*=8
    elif score>5000:
        score*=10
    elif score>8000:
        score*=15'''
    


    fitness = score*accuracy*(time/254)
    
    return fitness

def press_buttons(game_instance, output, output_names):
    for out, name in zip(output,  output_names):
        if out:
            game_instance.button_press(name)
        else:
            game_instance.button_release(name)

def eval_genome(genome, config):
    pyboy = PyBoy("!mario.gb", window="null")
    pyboy.set_emulation_speed(0)
    pyboy.load_state(open('!mario.gb.state', 'rb'))

    TIMEOUT_CONST = 30
    genome.fitness = 4.0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    button_config = ["left", "right", "a","b"]
    disp = 0
    last_dist = 0
    dist2 = 0
    jumps = 0
    timeout = 30
    while pyboy.memory[0xc0ac:0xc0ad][0] == 38: # check if genome died
    #while pyboy.tick():
        pyboy.tick()
        flat_list = [item for sublist in pyboy.game_area() for item in sublist]
        vel  = pyboy.memory[0xC20C:0xc20D][0]
        dire = pyboy.memory[0xC20D:0xc20e][0]
        dist = pyboy.memory[0xd113]

        if (pyboy.memory[0xc0a4:0xc0a5][0] == 0x39):
           break



        
        if(timeout<1):
            break
        
        if(dist == last_dist):
            timeout-=1
        else:
            timeout = 30


        if(vel>6):
            vel = 6
        if(dist == last_dist):
            dist2 = 0
        else:
            dist2 = 1
        
        if(dire&0x10):
            dire = 1
        elif dire & 0x20:
            dire = -1
        disp+= vel*dire*dist2 #+((~(pyboy.memory[0xc20a:0xc20b][0]))+2)*0.1
        last_dist = dist

        level_block = pyboy.memory[0xC0AB]
        mario_x = pyboy.memory[0xC202]
        scx = pyboy.screen.tilemap_position_list[16][0]
        level_progress = level_block*16 + (scx-7) % 16 + mario_x

        output = net.activate(flat_list)
        button_output = [round(x) for x in output]
        #print(button_output, ": ", genome_id)
        press_buttons(pyboy, button_output, button_config)
    #time_elapsed = pyboy.memory[0xcc70:0xcc72]
    #lives = pyboy.memory[0xcc80:0xcc81]
    dist = pyboy.memory[0xc202:0xc203]

    pyboy.stop()
    score = to_int(pyboy.memory[0xc0a0:0xc0a2])
    fitness = level_progress#disp#+to_int(pyboy.memory[0xc0a0:0xc0a3])
    print(fitness)
    return fitness


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
    p.add_reporter(neat.Checkpointer(1000,filename_prefix="para/mario_para_NOJUMP_"))
    import os
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'para')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    # Run for up to 300 generations.
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = p.run(pe.evaluate, 100000)

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
    config_path = os.path.join(local_dir, 'alt_conf')
    run(config_path)
