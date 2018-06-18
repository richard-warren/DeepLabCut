# coding: utf-8

############################
# This configuration file sets various parameters for generation of training
# set file & evalutation of results
############################

# myconfig.py:

########################################
# Step 1:
Task = 'run4lowres'
########################################

# Filename and path to behavioral video:
vidpath = '.'
filename = 'reachingvideo1.avi'

cropping = False

# ROI dimensions / bounding box (only used if cropping == True)
# x1,y1 indicates the top left corner and
# x2,y2 is the lower right corner of the croped region.

x1 = 0
x2 = 640
y1 = 277
y2 = 624


# Portion of the video to sample from in step 1. Set to 1 by default.
portion = 1

########################################
# Step 2:
########################################

# bodyparts = ['paw1', 'paw2', 'paw3', 'paw4', 'gen', 'tailBase', 'tailMid', 'paw1LH', 'paw2LF', 'paw3RF', 'paw4RH', 'tailBaseTop', 'tailMidTop']
bodyparts = ['paw1LH_top', 'paw2LF_top', 'paw3RF_top', 'paw4RH_top', 'tailBase_top', 'tailMid_top', 'nose_top', 'obs_top', 'paw1LH_bot', 'paw2LF_bot', 'paw3RF_bot', 'paw4RH_bot', 'tailBase_bot', 'tailMid_bot', 'nose_bot', 'obsHigh_bot', 'obsLow_bot']

# annotator in *.csv file
Scorers = ['']  # who is labeling?

# When importing the images and the labels in the csv/xls files should be in the same order!
# During labeling in Fiji one can thus (for occluded body parts) click in the origin of the image 
#(i.e. top left corner (close to 0,0)), these "false" labels will then be removed. To do so set the following variable:
#set this to 0 if no labels should be removed!
invisibleboundary=1 # If labels are closer to origin than this number they are set to NaN (not a number)

########################################
# Step 3:
########################################

date = ''
scorer = ''

# Userparameters for training set. Other parameters can be set in pose_cfg.yaml
Shuffles = [1]  # Ids for shuffles, i.e. range(5) for 5 shuffles
TrainingFraction = [0.95]  # Fraction of labeled images used for training

# Which resnet to use
# (these are parameters reflected in the pose_cfg.yaml file)
resnet = 50

# trainingsiterations='1030000'

# For Evaluation/ Analyzing videos
# To evaluate model that was trained most set this to: "-1"
# To evaluate all models (training stages) set this to: "all"

snapshotindex = -1
shuffleindex = 0
