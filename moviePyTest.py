from moviepy.editor import VideoFileClip
from skimage.util import img_as_ubyte
import numpy as np
import ipdb
import scipy.misc
from tqdm import tqdm



# # get save frame from specified file
session = '180124_000' # 241346




clip = VideoFileClip('C:\\Users\\rick\\Google Drive\\columbia\\obstacleData\\sessions\\' + session + '\\runBot.mp4')


nframes_approx = int(np.ceil(clip.duration*clip.fps)+10)
# ipdb.set_trace()

for counter in tqdm(range(0,nframes_approx)):
	
	image = clip.reader.read_frame()
	if image == []:
		nframes = counter
		print('got a blank frame yo!')
		break
	else:
		image = img_as_ubyte(image)
print('session: ' + str(nframes) + ' frames')




