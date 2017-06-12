# generates train_time_x and train_x
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Ellipse, Line
from kivy.gesture import Gesture, GestureDatabase
import time

# canvas size M*N
M = 1280
N = 800
time_start = 0

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
        global time_start
        time_start = time.clock()

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
        # save time to file
        global time_start
        fTime = open('train_time_.txt', 'a')
        fTime.write(str(time.clock()-time_start) + '\n')
        fTime.close()

        # save points to file
        f = open('train_.txt', 'a')
        pointList = touch.ud['line'].points
        for x in range(0, len(pointList)-1, 2):
        	f.write("%e" %float(pointList[x]*640.0/M) + ' ' + "%e" %float((N-pointList[x+1])*480.0/N))
        	f.write('\n')
        f.write('-----------\n')
        # erase the lines on the screen, this is a bit quick&dirty, since we can have another touch event on the way...
        self.canvas.clear()

class DemoGesture(App):
    def build(self):
        return GestureBoard()
        
if __name__ == '__main__':
    DemoGesture().run()