from pyboy import PyBoy
from tkinter import Tk, Canvas, PhotoImage, mainloop
import numpy as np
from PIL import Image, ImageTk
import threading


isfree = True

pics = {}
game_instances = {}
buttons = {"1": [0,1,0], "2":[1,0,0]}
def writepics(data, id):
    global isfree
    if isfree:
        isfree = False
        global pics
        pics[id] = data
        isfree = True

def readpics(id):
    if isfree :
        return pics[id]  
    else:
        return None
        
def press_buttons(game_instance, id):
            for idx, x in enumerate(buttons[id]):
                if x:
                    game_instance.button_press(button_config[idx])
                else:
                    game_instance.button_release(button_config[idx])
           
def pyboy_stuff(id):
    for x in range(5):
            game_instances[str(x)] = PyBoy("galaga.gb")
            with open('galaga.gb.state', 'rb') as f:
                game_instances[str(x)].load_state(f)
    while 1: 
        for id, instance in game_instances.items():
            instance.tick() 
            if str(id) in buttons:
                press_buttons(instance, id)
            


def tkinter_stuff(id):
    mult = 8
    WIDTH, HEIGHT = 32 * mult, 32 * mult
    window = Tk()
    canvas = Canvas(window, width=WIDTH, height=HEIGHT, bg="#000000")
    canvas.pack()
    img = PhotoImage(width=WIDTH, height=HEIGHT)
    
    
    def update(img):
        try:
            if id in pics:
                game_area = np.array(readpics(id))
                game_area = np.repeat(game_area,mult,axis=0)
                game_area = np.repeat(game_area,mult,axis=1)
                np.shape(game_area)
                img = ImageTk.PhotoImage(image = Image.fromarray(game_area))
                canvas.create_image(WIDTH/2, HEIGHT/2, image=img)
            
         
        except Exception as e:
            print("The error is: ", e.with_traceback())
            pass
        canvas.after(5, lambda : update(img))
    update(img)
    mainloop()
    

    
t1 = threading.Thread(target=pyboy_stuff, args=("1"))
# t2 = threading.Thread(target=pyboy_stuff, args=("2"))
# t3 = threading.Thread(target=tkinter_stuff, args=("1"))
# t4 = threading.Thread(target=tkinter_stuff, args=("2"))

t1.start()
# t2.start()
# #t3.start()
# #t4.start()

t1.join()
# t2.join()
# #t3.join()
# #t4.join()

print("Done!")