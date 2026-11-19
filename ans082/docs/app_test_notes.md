# App test notes

I ran 28 files through the same code as the app (python src/run_test_cases.py). The app did not crash on any of them.

## Bad files
- Blank black, blank white, a tiny picture (30 x 30), a damaged file and an empty file were all rejected with a clear message.
- A phone photo stored sideways gave the same answer as the upright photo (glass, 98%).
- A very large picture (5000 x 4000), a grayscale picture and a transparent picture worked. The model said "not sure".

## Real photos
- Photos with two items, items the model does not know, and a photo of a person all got "not sure".
- One dark or blurry photo got the answer glass (95%). 
- Of 9 plastic photos, only 1 got an answer (plastic, 98%). The other 8 got "not sure", because the model was less than 90% sure.
- One screenshot got a confident wrong answer (paper, 98%). The model has no class for "not a waste item", so it can be sure and still wrong.

## What I learned
The "not sure" rule protects the user from many wrong answers, but on real phone photos it is used a lot.
The model cannot tell if a picture is outside its 10 classes.
Each photo takes about 1 second on my laptop CPU, and the app works without internet.