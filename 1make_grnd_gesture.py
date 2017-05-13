from kivy.app import App

from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Ellipse, Line
from kivy.gesture import Gesture, GestureDatabase

# canvas size M*N
M = 800
N = 600

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
        f = open('gesture_out.txt', 'a')
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