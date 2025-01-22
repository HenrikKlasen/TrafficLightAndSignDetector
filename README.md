# TrafficLightAndSignDetector
This project was part of the curriculum of the Bachelor in Computer Science program at the University of Luxembourg. It was about a combined traffic light and sign detection using a joined dataset of traffic signs and traffic lights.

## Motivation
The current Traffic light detector of RoboCar only recognizes traffic lights. For Traffic signs it uses an HD map, which provides the correct information _most of the time_. 

## Results and Features
For the times it does not provide the right information, this system is made to identify the most important traffic signs:
- Speed limits (20, 30, 40, 50, 60, 70, 80, 100, 120)
- Yield
- Stop
- Entry prohibited/No entry
- One-way street
- Roadworks (and end of playstreet)
- Play street (and end of playstreet)
- Pedestrian crossing
- Dead end
Next to these, it can recognize green and red traffic lights.

## Specs
These models were run on a RTX3060 mobile GPU with 6GB of VRAM. If you want to retrain the model, please use the DriveU Traffic Light Dataset, BelgianTS dataset (camera00 and camera01), as well as the traffic signs and traffic lights dataset by sonia, available on roboflow.
The medium model (`best_m.pt`) was trained on 448px $\times$ 448px, while the best small model (`best_s.pt`) was trained on 640px $\times$ 640px.
