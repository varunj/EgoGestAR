# **EgoGestAR** *(ego-gesture)*
An egocentric pointing gesture dataset. Developed for [DrawInAir: A Lightweight Gestural Interface Based on Fingertip Regression ](https://ilab-ar.github.io/DrawInAir/ "DrawInAir: A Lightweight Gestural Interface Based on Fingertip Regression ")

# Links
Codebase: *will come soon*

Public EgoGestAR Dataset: https://github.com/varunj/EgoGestAR

Public Testing Dataset: https://github.com/varunj/EgoGestAR/tree/master/testvideo

Project Website: https://ilab-ar.github.io/DrawInAir/


# The Gestures
- The left column shows standard input gesture sequences shown to the users before the data collection.
- The right column depicts the variation in the data samples.

These gestures could be applied to different use cases. 
- The black block depicts swipe gestures (Left, Right, Up, Down) for list interaction.
- Green block showing Rectangle and Circle gestures for Region of Interest (RoI) highlighting.
- Red block (Checkmark: Yes, Caret: No, X: Delete, Star: Bookmark) gestures for evidence capture (in, say, industrial AR applications.)


*The highlighted point in each gesture indicates the starting position of the gesture.*

![](https://github.com/varunj/EgoGestAR/blob/master/ztemp_cvpreccv_webpage/pointgestar_img/fig4_fig5.png)


# Usage

## test/
- Lists 500 gesture inputs used for testing. 50 gestures per class.
- File naming convention: test\_*gesturename*\_*serialnumber*.txt 

## test_images/
- Lists 500 gesture images corresponding to the inputs. Plotted in black and white with a circular dot representing the starting of the gesture. 
- File naming convention: test\_*gesturename*\_*serialnumber*.png

## test_time/
- Lists the time taken (in seconds) to perform each test gesture.

## train/
- Lists 2000 gesture inputs used for training. 200 gestures per class.
- File naming convention: train\_*gesturename*\_*serialnumber*.txt 

## train_images/
- Lists 2000 gesture images corresponding to the inputs. Plotted in black and white with a circular dot representing the starting of the gesture. 
- File naming convention: train\_*gesturename*\_*serialnumber*.png 

## train_time/
- Lists the time taken (in seconds) to perform each training gesture.

