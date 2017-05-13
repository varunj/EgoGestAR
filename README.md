TCS_DTW_Gestures
================================================

DTW for Gesture Recognition. Includes gesture generator.

## 0make_grnd_basic.py
1. Run to generate ground truth files for left, right, up, down gestures.

## 1make_grnd_gesture.py
1. Run to generate ground truth for custom gestures.
2. Gestures saved in gesture_out.txt separated by '-----------'.

## 2make_grnd_sparse.py
1. Reads from /grnd.
2. Saves alternate points to /grnd_sparse.

## 3display_files.py
1. Reads from the specified folder.
2. Generates a graphical representation of the gestures in 10 graphs (to interpret the gesture).

## 4generate_results.py
1. Reads from /seeds and /grnd_sparse and output the result to stdout.
2. Gives the absolute best match and costs corresponding to each ground truth gesture.

## 5generate_results_realtime.py
1. Accepts the input gesture during runtime.
2. Outputs as above in <4generate_results>.

## Note
1. Requires Kivy (https://kivy.org/docs/installation/installation.html)