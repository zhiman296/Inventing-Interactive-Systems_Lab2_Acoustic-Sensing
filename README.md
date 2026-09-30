## Back Row Nap
[Demo video]https://youtu.be/Xf2TaWxrriE 

A small reaction game controlled by sound. 
You are a student who fell asleep in the back row, and the teacher is walking toward you. 

Ring a bell the right number of times to wake up and look busy before you get caught.

The game uses the laptop's microphone as the input and detects the sound of a bell ringing. 
(I used a pen to strike my water bottle to mimic the sound of a real bell.)

### How to Play
The teacher walks toward you in three danger levels. 

The number of red exclamation marks above his head tells you how many times to ring:

 !	  ->  Ring once
 
!!	  ->  Ring twice

!!!	  ->  Ring three times
  
  - Correct count: the teacher goes back to the board and you pretend to study.
  - Wrong count or too slow: he moves one level closer and gets angrier.
  - Level 3 failed: you are caught and lose a life.
  - Ringing when it is safe (his back is turned) makes him turn around, so stay quiet.
  - You have 3 lives. The game speeds up as you survive more patrols.
  - Rings less than about half a second apart count as one group, so "ring-ring" means 2.
  
  Ring the bell on the title screen to start, and ring again on the game over screen to restart.

### Python Setup and Running the Script 
1. Install the required packages using: `pip install pygame numpy sounddevice`
2. Run the script: `Back_Row_Nap.py`
3. "Ring the bell" to start playing.
