% # Code and Data for Paper "Learning to Navigate Unseen Environments: Back Translation with Environmental Dropout" 

## Environment Installation
Please refer to https://github.com/airsplay/R2R-EnvDrop to install the Matterport3D simulators, download Room-toroom dataset and install the python enviornments.

## Code

### Speaker
```
bash run/speaker.bash 0
```
0 is the id of GPU. It will train the speaker and save the snapshot under snap/speaker/

### Agent
```
bash run/agent.bash 0
```
0 is the id of GPU. It will train the agent and save the snapshot under snap/agent/. Unseen success rate would be around 46%.

### Agent + Speaker (Back Translation)
After pre-training the speaker and the agnet,
```
bash run/bt_envdrop.bash 0
```
0 is the id of GPU. 
It will load the pre-trained agent and run back translation with environmental dropout.

Currently, the result with PyTorch 1.1 is a little bit lower than my NAACL reported number. It still easily reaches a success rate of 50% (+4% from w/o back translation).


2. Release pre-trained snapshots.
3. Check PyTorch 1.1 configurations.
4. Update pip requirement with version specifications.

