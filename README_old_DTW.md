TCS_DTW_Gestures
================================================

DTW for Gesture Recognition. Includes gesture generator.

## 1make_grnd_basic.py
1. Run to generate ground truth files for left, right, up, down gestures.

## 2make_grnd_gesture.py
1. Run to generate ground truth for custom gestures and corresponding time taken to draw each gesture.
2. Gestures saved in gesture_out.txt separated by '----------'.

## 3display_files.py
1. Reads from the specified folder.
2. Generates a graphical representation of the gestures in 10 graphs (to interpret the gesture).

## 4generate_results.py
1. Reads from /seeds and /grnd_sparse and output the result to stdout.
2. Gives the absolute best match and costs corresponding to each ground truth gesture.

## 5generate_results_realtime.py
1. Accepts the input gesture during runtime.
2. Outputs as above in <4generate_results>.

## 6trainRawProcess.py
1. Reads '-----------' separated gestures and stores as different files.

## 7histogram_gesturelen.py
1. Outputs a histogram of the length of raw gestures captured.

## 8resample_to_200.py
1. Up/Down samples raw gestures to 200 length.

## Note
1. Requires Kivy (https://kivy.org/docs/installation/installation.html)
