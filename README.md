# Deep Q-Learning

Final project in T-747-RELE (Reinforcement Learning) at Reykjavík University Spring semester 2021.

An attempt at reproducing the methods proposed by Mnih et al in their paper Human-level Control Through Deep Reinforcement Learning (https://www.nature.com/articles/nature14236).

The first idea was to apply the methods to the recommendations problem by using the MovieLens dataset. However, that turned out to be problematic, so the focus turned on reproducing the methods on the Atari game Breakout using the Open AI Gym framework (https://gym.openai.com/)

The recommendation code runs, but does not yield any great results.

The breakout code was run for 8.400 episodes, yielding a highest 100 episode moving average of 12.11 points and a highest reward of 26 points. Learning had not stopped at 8.4000 epsiodes, but the run had to be cancelled due to time constraints.

![ep_8400](https://user-images.githubusercontent.com/8950246/118146052-bf267b00-b3fd-11eb-859e-7f6b50a09fa5.png)

# Example videos
## First episode
https://user-images.githubusercontent.com/8950246/118146163-da918600-b3fd-11eb-9e27-c082486c80d4.mp4

## After 2000 episode
https://user-images.githubusercontent.com/8950246/118147106-c601bd80-b3fe-11eb-9696-4fc39b3e7cdd.mp4

## After 8400 episodes
https://user-images.githubusercontent.com/8950246/118147204-e2055f00-b3fe-11eb-8b7f-daa9bfd9f834.mp4

# Running the code
NOTE: This project used Python 3.8. It does not run on Python 3.9 due to the Gym framework not working for Python 3.9 at the moment. This may be fixed in a later release of Gym.

Create a virtual environment

    python3 -m venv env

Activate the virtual environment

    source env/bin/activate

Install 3rd level requirements

    pip install -r requirements.txt

To run the recommendations problem

    python3 runRecommendations.py

To run the Breakout problem

    python3 runBreakout.py
    
