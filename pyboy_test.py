from pyboy import PyBoy
from tkinter import Tk, Canvas, PhotoImage, mainloop
import numpy as np
from PIL import Image, ImageTk
import threading


isfree = True

pics = []
def writepics(data):
    global isfree
    if isfree:
        isfree = False
        global pics
        pics = data
        isfree = True

def readpics():
    if isfree :
        return pics 
    else:
        return None

           
def pyboy_stuff():
    pyboy = PyBoy("galaga.gb")
    print(pyboy.tilemap_window[0:32,0:32])
    while pyboy.tick(): 
            

        writepics()
    pyboy.stop()
            


def tkinter_stuff():
    mult = 8
    WIDTH, HEIGHT = 32 * mult, 32 * mult
    window = Tk()
    canvas = Canvas(window, width=WIDTH, height=HEIGHT, bg="#000000")
    canvas.pack()
    img = PhotoImage(width=WIDTH, height=HEIGHT)
    
    
    def update(img):
        try:
            game_area = np.array(readpics())
            game_area = np.repeat(game_area,mult,axis=0)
            game_area = np.repeat(game_area,mult,axis=1)
            np.shape(game_area)
            img = ImageTk.PhotoImage(image = Image.fromarray(game_area))
            canvas.create_image(WIDTH/2, HEIGHT/2, image=img)
 
        except Exception as e:
            #print("The error is: ", e.with_traceback(e.__traceback__))
            pass
        canvas.after(5, lambda : update(img))
    update(img)
    mainloop()
    

    
t1 = threading.Thread(target=pyboy_stuff)
# t2 = threading.Thread(target=pyboy_stuff, args=("2"))
t3 = threading.Thread(target=tkinter_stuff)
# t4 = threading.Thread(target=tkinter_stuff, args=("2"))

t1.start()
# t2.start()
t3.start()
# #t4.start()

t1.join()
# t2.join()
t3.join()
# #t4.join()

print("Done!")