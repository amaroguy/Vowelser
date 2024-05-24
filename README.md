# Vowelser - A Voice Controlled Controller

## What is this?

Vowelser is my final project for the LIGN 168 "Computational Speech Processing" class at UCSD. It is a controller that you can control with only your voice. The control stick is controlled by the F1 and F2 formants of your voice by making vowels, and buttons can be pressed using phonemes like /p/ /t/ /k/. I do not recommend it for any serious gaming, but its a cool proof of concept.

# Setup

## 1. requirements.txt

Install any requirements listed in requirements.txt

## 2. Calibrating the control stick to your voice

This is easier said than done. The file for this is ```test_controlstick_and_visualize.py``` which will visualize
the controller hitbox using a window, and use a red dot to visualize where the control stick is currently pointing.

For each of the corner vowels, /i/, /ae/, /u/, and /a/ do the following.

1. Start producing the corner vowel
2. While producing the vowel, run ```test_controlstick_and_visualize.py```, keep producing the vowel.
3. After ~5s of producing, halt the script using Ctrl + C, this will print the median F1, F2 formants.

Even this script needs to be calibrated, I personally used Praat to analyze my formants to make sure what this script was outputting lined up with an implementation of this I knew was correct. If Praat is saying something different, you might have to play with ```BUFFER_SIZE``` or the ```order``` parameter when Librosa's LPC function is called.

After you have your F1, F2s for each of the vowels, take the following maxes and mins

``` MAX_F1, MAX_F2 = max(F1_VALUES), max(F2_VALUES)```

``` MIN_F1, MIN_F2 = min(F1_VALUES), min(F2_VALUES)```

Put these values into the ```F1_RANGE, F2_RANGE``` variables, like so

``` F1_RANGE = (MIN_F1, MAX_F1) ```

``` F2_RANGE = (MIN_F2, MAX_F2) ```

Rerun ```test_controlstick_and_visualize.py```, the dot should follow your formants now most of the time according to common visualizations of vowel space like the [one here](https://en.wikipedia.org/wiki/Vowel_diagram)


## 3. Making a model to recognize your button inputs

This also took me a while. The Jupyter notebook for this section is ```Classification.ipynb``` which builds a Logistic Regression model based off of the training data you will setup in ```./classifier/training/data```. 

### Data Collection

I personally used Audacity for this, and used the phonemes /p/, /t/, /k/ and /d͡ʒ/ for button presses. Each of the wav files labeled used 44kHz sampling rate and a bit depth of 16. I recorded for each of the phonemes, a 5 minute file of me repeating the phoneme. Since we don't want button presses to happen when these arent being said, I also recorded 5 minutes of vowel-age (going /i/ /a/ /u/ and all the other vowels) and 5 minutes of just room noise. This gives us 6 classes for the model to predict: /p/, /t/, /k/, /d͡ʒ/, vowel, and silence.

Take this 30 minutes of audio, and label at least 200 of each of the classes in the format ```{CLASS_LABEL}_{SAMPLE_NUMBER}```, for example for my 168th /p/ sample, it would be ```P_168```. Export the labeled sections into their own individual wav files, each file named using the label described previously. 

Put these files into your directory as follows
```
├── p/
│ ├── p_1.wav
│ ├── p_2.wav
│ └── p_3.wav
├── t/
│ ├── t_1.wav
│ ├── t_2.wav
│ └── t_3.wav
├── k/
│ ├── k_1.wav
│ ├── k_2.wav
│ └── k_3.wav
├── dj/
│ ├── dj_1.wav
│ ├── dj_2.wav
│ └── dj_3.wav
```
### Notebook

You can run the cells top to bottom, making adjustments to ```build_df``` or ```train_model``` if you don't want to use the train/test split I used or want to adjust hyperparameters. After this, you should have a ```lr_dump.joblib``` file with your trained LogisticRegression model.

## 4. Running the controller

Take the F1,F2 range you set up in step 2 and put it into ```start.py```, along with any modifications you might have made to the control stick. You now have a virtual controller that can be controlled with your voice.

## 5. Setup with emulator

I tried this controller with only Super Mario 64, and tried to use the most popular emulator Project64, which would not recognize the controller. I had success with [Rosalie's Mupen64 GUI](https://github.com/Rosalie241/RMG) however. I highly recommend using ```sync.py``` to set up the controller in the emulator's settings. The process is as follows

1. Start the script
2. For example if it says "Pressing A button in 3 seconds", move to the emulator and press on whatever button
you want it to listen for input for (in this case, you tell it to listen for the A press). Repeat for the other buttons and stick.

Your controller is now synced!

# Whats the rest of this repo?

## ./classifier/training/deeplearning_fail

I originally tried using Deep Learning for this project, which failed miserably. A lot of the code is adapted from [TheSoundOfAI's](https://www.youtube.com/@ValerioVelardoTheSoundofAI) YouTube series ["PyTorch For Audio + Music Processing"](https://www.youtube.com/playlist?list=PL-wATfeyAMNoirN4idjev6aRu8ISZYVWm) and ["Audio Signal Processing for Machine Learning"](https://www.youtube.com/playlist?list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0). Unfortunately his tutorial which used a 10+ hour audio dataset did not transfer over very well for my 5 minute dataset. I kept it here for reference since I finally learned what CNNs were doing from attempting this approach. 

## ./lpc_examples_generator.ipynb

Used to generate the visualizations used in the final project writeup.