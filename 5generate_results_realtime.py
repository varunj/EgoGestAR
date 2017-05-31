from kivy.app import App

from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Ellipse, Line
from kivy.gesture import Gesture, GestureDatabase

import glob, os
import pandas as pd
import numpy as np
from dtw import dtw
import operator

# canvas size M*N
M = 800
N = 600

# read ground truths
dic_grndFiles = {}
for fileName in glob.glob("./grnd/*"):
    file = pd.read_csv(fileName, delim_whitespace = True, header = None)
    arr = np.array(file.ix[:, :])
    dic_grndFiles[fileName] = arr 

def simplegesture(name, point_list):
    """
    A simple helper function
    """
    g = Gesture()
    g.add_stroke(point_list)
    g.normalize()
    g.name = name
    return g


class GestureBoard(FloatLayout):
    """
    Our application main widget, derived from touchtracer example, use data
    constructed from touches to match symboles loaded from my_gestures.
    """
    def __init__(self, *args, **kwargs):
        super(GestureBoard, self).__init__()
        self.gdb = GestureDatabase()


    def on_touch_down(self, touch):
        # start collecting points in touch.ud create a line to display the points
        userdata = touch.ud
        with self.canvas:
            Color(1, 1, 0)
            d = 30.
            Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
            userdata['line'] = Line(points=(touch.x, touch.y))
        return True

    def on_touch_move(self, touch):
        # store points of the touch movement
        try:
            touch.ud['line'].points += [touch.x, touch.y]
            return True
        except (KeyError) as e:
            pass

    def on_touch_up(self, touch):
        # touch is over, display informations, and check if it matches some known gesture.
        g = simplegesture('', list(zip(touch.ud['line'].points[::2],
                                       touch.ud['line'].points[1::2])))
        # save points to file
        f = open('xgesture_temp.txt', 'w+')
        pointList = touch.ud['line'].points
        for x in range(0, len(pointList)-1, 2):
        	f.write("%e" %float(pointList[x]*640.0/M) + ' ' + "%e" %float((N-pointList[x+1])*480.0/N))
        	f.write('\n')
        f.close()

        # read temp file to detect gesture
        file = pd.read_csv("xgesture_temp.txt", delim_whitespace = True, header = None)
        arr_seed = np.array(file.ix[:, :])   
        dic_ans = {}
        for grnd_name, arr_grnd in dic_grndFiles.items():
            distance, cost, acc, path = dtw(arr_seed, arr_grnd, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
            dic_ans[str(grnd_name)] = distance*100.0/(640+480)
        print(min(dic_ans, key=dic_ans.get))
        print(sorted(dic_ans.items(), key=operator.itemgetter(1)))
        print

        # erase the lines on the screen, this is a bit quick&dirty, since we can have another touch event on the way...
        self.canvas.clear()


class DemoGesture(App):
    def build(self):
        return GestureBoard()


if __name__ == '__main__':
    DemoGesture().run()