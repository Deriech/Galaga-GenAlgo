import multiprocessing
import os
from pyboy import PyBoy

import neat
import visualize

def press_buttons(game_instance, output, output_names):
    for out, name in zip(output,  output_names):
        if out:
            game_instance.button_press(name)
        else:
            game_instance.button_release(name)
            
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward-partial-ga')
 
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
p = neat.Checkpointer.restore_checkpoint('Galaga/checkpoints/neat-checkpoint-147')
best_g = sorted([p for p in list(p.population.values()) if p.fitness != None], key= lambda p: p.fitness, reverse=True)
pyboy = PyBoy("galaga.gb")
pyboy.set_emulation_speed(0)
pyboy.load_state(open('galaga.gb.state', 'rb'))
net = neat.nn.FeedForwardNetwork.create(best_g[0], config)
button_config = ["left", "right", "a"]
while pyboy.tick():
    pyboy.tick()
    flat_list = [item for sublist in pyboy.game_area() for item in sublist]
    output = net.activate(flat_list)
    button_output = [round(x) for x in output]
    print(button_output)
    press_buttons(pyboy, button_output, button_config)
pyboy.stop()
