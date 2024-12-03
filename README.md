# Running Prediction

Many recreational runners, including myself, may want to step up to the next level and sign up for a race to challenge themselves. Race participants often set goals for themselves, such as a sub-two-hour half marathon or a sub-four-hour full marathon. Such a feat requires strategic pacing, efficient running and breathing techniques, and consistent training.

This project will utilize ML algorithms to analyze real running data to develop a predictive model for race finish times. The models will analyze data from Strava (distance, pace, heart rate, elevation, etc) and utilize regression techniques to estimate finish times based on the user’s training patterns. In addition, classification algorithms can be used to separate actual training data from recovery runs or commute runs to improve prediction accuracy.

# Directory

<b>./images</b>
Folder for images of my runs.

<b>./results/decision_tree_entropy</b>
Folder for the image results of the evaluation metrics on the decision tree model.

<b>./DSC412_001_FA24_PR_sfeng9.pdf</b>
The plan and proposal of this project.

<b>./DSC412_001_FA24_FR_sfeng9.pdf</b>
The final report of this project.

<b>./RunningPrediction.ipynb</b>
The python notebook file that contains the actual data processing and model building.

<b>./evaluation_fn.py</b>
Python file for evaluation functions on the decision tree performance.

<b>./requirements.txt</b>
Text file that contains the list of libraries required to run the project.

<b>./strava_data.csv</b>
The dataset containing running information.

# Data

<b>Size</b>
43 rows, 16 variables

<b>Description</b>
This dataset contains information about 43 different observations (runs) from 5/23/2024 to 10/26/2024. Each observation includes information such as distance, time, speed, heartrate, shoe, and other variables about the run. We can use this data to find any relationship amongst the variables and train both classification and regression models with the optimal variables.

<b>Source</b>
The data is collected from Strava, a recreational application that connects active people around the world through tracking and sharing.
https://www.strava.com/athletes/140162123

During my runs, I have a Garmin Forerunner 265 and heart rate monitor that collects information. This information is processed by both the Garmin Connect and Apple Health apps and sent to Strava for better formatting and tracking.

<b>Variables</b>
- date: The date of the activity (datetime54)
- time: The duration of the activity (object) (format: H:MM:SS)
- distance: The total distance covered during the activity (float64) (unit: miles)
- avg_speed: The average speed maintained during the activity (object) (format: H:MM:SS)
- max_speed: The maximum speed reached during the activity (object) (format: H:MM:SS)
- avg_heartrate: The average heart rate during the activity (int64) (unit: bpm)
- max_heartrate: The maximum heart rate recorded during the activity (int64) (unit: bpm)
- elevation_gain: The total elevation gained during the activity (int64) (unit: feet)
- avg_power: The average power output during the activity (float64) (unit: watts)
- max_power: The maximum power output achieved during the activity (float64) (unit:watts)
- total_work: The total work performed during the activity (float64) (unit: joules)
- avg_cadence: The average cadence during the activity (float64) (unit: steps per minute)
- max_cadence: The maximum cadence achieved during the activity (float64) (unit: steps per minute)
- calories_burned: The total number of calories burned during the activity (int64) (unit: kcals)
- shoe: The specific shoe used for the activity (object)
- run_type: The type of run (object)


# Getting Started (from lecture material)

Make sure your terminal is in this directory. You can confirm that is true by typing `pwd` in terminal.

Create a virtual environment with

`python -m venv ./.venv`

Then activate it in terminal:

Windows: `.\.venv\Scripts\activate`

Mac: `source ./.venv/bin/activate`

Linux: `source ./.venv/bin/activate`

You should see `.venv` appear in the terminal on the left side of the command line.

Then run `pip install -r requirements.txt` in terminal
