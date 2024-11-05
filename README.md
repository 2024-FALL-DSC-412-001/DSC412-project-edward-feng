# Running Prediction

Many recreational runners, including myself, may want to step up to the next level and sign up for a race to challenge themselves. Race participants often set goals for themselves, such as a sub-two-hour half marathon or a sub-four-hour full marathon. Such a feat requires strategic pacing, efficient running and breathing techniques, and consistent training.

This project will utilize ML algorithms to analyze real running data to develop a predictive model for race finish times. The models will analyze data from Strava (distance, pace, heart rate, elevation, etc) and utilize regression techniques to estimate finish times based on the user’s training patterns. In addition, classification algorithms can be used to separate actual training data from recovery runs or commute runs to improve prediction accuracy.

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
