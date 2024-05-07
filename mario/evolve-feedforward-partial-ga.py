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

def bcd_byte_to_int(bcd_byte):
    # Extract the high nibble (first digit)
    high_nibble = (bcd_byte & 0xF0) >> 4
    # Extract the low nibble (second digit)
    low_nibble = bcd_byte & 0x0F
    
    
    # Validate the digits
    if high_nibble > 9 or low_nibble > 9:
        raise ValueError("Invalid BCD digit detected")
    
    # Convert the BCD digits to an integer
    decimal = high_nibble * 10 + low_nibble
    return decimal

def to_int(lst,byte=0):
    if(byte):
        __lst = []
        for x in range(0,byte):
            __lst.append(bcd_byte_to_int(lst[x]))

        lst = __lst
            
    idx = 0
    ret = 0
    for x in lst:
        ret+=x*(10**idx)
        idx+=1+(byte-1)
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
    if(score>500):
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
        score*=15
    


    fitness = score*accuracy
    
    return fitness

def press_buttons(game_instance, output, output_names):
    for out, name in zip(output,  output_names):
        if out:
            game_instance.button_press(name)
        else:
            game_instance.button_release(name)

def eval_genomes(genomes, config):
    
    pyboy = PyBoy("!mario.gb",)
    #pyboy.set_emulation_speed(0)
    pyboy.load_state(open('!mario.gb.state', 'rb'))
    for genome_id, genome in genomes:
        print("running genome:", genome_id)
        #genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        button_config = ["left", "right", "a","b"]
        disp = 0
        last_dist = 0
        dist2 = 0
        speed_effect = 0
        TIMEOUT_CONST = 350
        timeout = TIMEOUT_CONST
        level_progress = 0
        print(genome.fitness)
        while ((pyboy.memory[0xc0ac:0xc0ad][0] == 38)): # check if genome died
            if(genome.fitness):
                if(genome.fitness<2600):
                    break
            else:break
        #while pyboy.tick():
            pyboy.tick()
            flat_list = [item for sublist in pyboy.game_area() for item in sublist]
            vel  = pyboy.memory[0xC20C]
            dire = pyboy.memory[0xC20D:0xc20e][0]
            dist = pyboy.memory[0xc001:0xc002][0]

            #print(pyboy.memory[0xc0ac:0xc0ad][0])
            

            if (pyboy.memory[0xc0a4:0xc0a5][0] == 0x39):
                break
            if(timeout<1):
                break
            if(dist == last_dist):
                timeout-=1
            else:
                timeout = TIMEOUT_CONST
            #print(timeout)
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
            speed_effect += vel/6
            #print(disp,vel,dire,dist,last_dist,dist2)
           # print(pyboy.memory[0xc0ac:0xc0ad])
           # time_elapsed = pyboy.memory[0xcc70:0xcc71]
           # if(time_elapsed[0]==0xfe):
           #     break
            #lives = pyboy.memory[0xcc80:0xcc81]
            #shot_count = pyboy.memory[0xcc84:0xcc86]
            #kill_count = pyboy.memory[0xcc86:0xcc88]
            #score = pyboy.memory[0xcc7a:0xcc7f]
            #print(time_elapsed, shot_count, kill_count, score)
            #print(lives)
            #print(disp)
            output = net.activate(flat_list)
            button_output = [round(x) for x in output]
            #print(button_output, ": ", genome_id)
            press_buttons(pyboy, button_output, button_config)
        #time_elapsed = pyboy.memory[0xcc70:0xcc72]
        #lives = pyboy.memory[0xcc80:0xcc81]
        shot_count = pyboy.memory[0xcc84:0xcc86]
        kill_count = pyboy.memory[0xcc86:0xcc88]
        score = pyboy.memory[0xcc7a:0xcc7f]
        time = to_int(pyboy.memory[0xda01:0xda03],byte=2)
        genome.fitness = level_progress# - time + speed_effect
        print(genome.fitness)
        pyboy.load_state(open('!mario.gb.state', 'rb'))


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Checkpointer.restore_checkpoint('mario_para_NOJUMP_13464')#neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(100,filename_prefix="mario_part"))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 10000)

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
