import scipy.io as sio
import os
import numpy as np

# Path to the directories of features and labels
feature_dir = '/home/ye/Works/pain'
label_dir = '/home/ye/Works/pain/Sequence_Labels'
feature_name = 'feature_from_verification_model.mat'
label_name = 'OPR'

def downsampling(input_data, after_len):
	"""Change the number of frames for each video. 
	Args:
		input_data: 2D numpy array. The dimention is num_frames x feature_dim.
	Returns:
		after.T: 2D numpy array. The dimention is num_frames_after_change x feature_dim."""
	mid_index = int(len(input_data[0])/2)
	if len(input_data[0]) > after_len: 
		after = input_data.T[mid_index-after_len/2:mid_index+after_len/2] # take the frames from the middle of the video to the two sides.
		after = after.T
	elif len(input_data[0]) <= after_len:
		after = np.lib.pad(input_data, ((0,0),(0,after_len-len(input_data[0]))), 'constant', constant_values=(0,)) # padding zaros at the end.
	return after.T

def get_feature_tensor(feature_dir,feature_name, max_len):
	"""Get the features from the mat file. 
	Args:
		feature_dir: path to feature directory.
		feature_name: name of the mat file.
		max_len: the number of frames for each video.
	Returns:
		x[1:]: 3D numpy array for the features. The dimention is num_videos x num_frames x feature_dim."""
	feature = sio.loadmat(os.path.join(feature_dir, feature_name))
	x = np.zeros((1,max_len,512))
	for person, video_feature in enumerate(feature['video_feature'].T[0]):
		for id, vid in enumerate(video_feature):
			vid_mat = vid[0].T # num_frames x feature_dim
			# make num_frame be same for each video
			vid_mat_mid = downsampling(vid_mat.T, max_len) 
			vid_mat_extend = np.expand_dims(vid_mat_mid, axis=0) # 1 x num_frames x feature_dim
			x = np.concatenate((x, vid_mat_extend),axis=0)
	return x[1:]

def get_labels(label_dir, label_name):
	"""Get the label list for all videos. 
	Args:
		label_dir: path to video labels directory.
		label_name: name of label. ex.'OPR'.
	Returns:
		labels: a list for labels for all videos."""
	path1 = os.path.join(label_dir, label_name)
	# print path1
	fold_names = [f for f in os.listdir(path1)]
	sort_name = np.sort(fold_names)
	# print sort_name
	labels = []
	for id, dir in enumerate(sort_name):
		path2 = os.path.join(path1, dir)
		sort_file = np.sort(os.listdir(path2))
		file_names = [f for f in sort_file if os.path.isfile(os.path.join(path2, f))]
		for label_file in file_names:
			path3 = os.path.join(path2, label_file)
			label_value = int(np.loadtxt(path3))
			labels.append(label_value)
	return labels

def get_frame_labels(feature_dir,feature_name, max_len):
	"""Get the labels for each frame for all videos from the mat file. 
	Args:
		feature_dir: path to feature directory.
		feature_name: name of the mat file.
		max_len: the number of frames for each video.
	Returns:
		x[1:]: 3D numpy array for the frame labels. The dimention is num_videos x num_frames x 1."""
	np.set_printoptions(threshold='nan')
	feature = sio.loadmat(os.path.join(feature_dir, feature_name))
	x = np.zeros((1,max_len,1))
	for person, video_feature in enumerate(feature['video_pain_level'].T[0]):
		for id, vid in enumerate(video_feature):
			vid_mat = vid[0].T # num_frames x 1
			# make num_frame be same for each video
			vid_mat_mid = downsampling(vid_mat.T, max_len)
			vid_mat_extend = np.expand_dims(vid_mat_mid, axis=0) # 1 x num_frames x 1
			x = np.concatenate((x, vid_mat_extend.T),axis=0)
	return x[1:]

def get_frame_01_labels(feature_dir,feature_name, max_len):
	"""Get the 0/1 label for each frame for all videos from the mat file. If the label > 0, assign 1 to it.
	Args:
		feature_dir: path to feature directory.
		feature_name: name of the mat file.
		max_len: the number of frames for each video.
	Returns:
		x[1:]: 3D numpy array for the 0/1 frame labels. The dimention is num_videos x num_frames x 1."""
	feature = sio.loadmat(os.path.join(feature_dir, feature_name))
	x = np.zeros((1,max_len,1))
	for person, video_feature in enumerate(feature['video_pain_level'].T[0]):
		for id, vid in enumerate(video_feature):
			vid_mat = vid[0] # 1 x num_frames 
			for i, lab in enumerate(vid_mat[0]):
				# If the label > 0, assign 1 to it.
				if lab>0:
					vid_mat[0][i] = 1
			# make num_frame be same for each video
			vid_mat_mid = downsampling(vid_mat, max_len)
			vid_mat_extend = np.expand_dims(vid_mat_mid, axis=0) # 1 x num_frames x 1
			x = np.concatenate((x, vid_mat_extend.T),axis=0) 
	return x[1:]

def non_zero_data(x,y,max_len, y_frame, use_y_frame=False):
	"""Get frame features, frame labels and vidoes labels for all videos with non-zero labels.
	Args:
		x: frame features array.
		y: vieo label list.
		max_len: the number of frames for each video.
		y_frame: frame labels. 
		use_y_frame: whether to use the frame labels. If setting False, will have two returns.
	Returns:
		x_new[1:]: 3D numpy array for the frame features. The dimention is num_videos x num_frames x feature_dim.
		labels_new: list for video labels.
		y_frame_new[1:]: 3D numpy array for the frame labels. The dimention is num_videos x num_frames x 1."""
	x_new = np.zeros((1,max_len,512))
	if use_y_frame == False:
		labels_new = []
		for i, n in enumerate(y):
			if n != 0:
				labels_new.append(y[i])
				x_expend = np.expand_dims(x[i], axis=0)
				x_new = np.concatenate((x_new, x_expend),axis=0)
		return x_new[1:], labels_new
	elif use_y_frame == True:
		y_frame_new = np.zeros((1,max_len,1))
		labels_new = []
		for i, n in enumerate(y):
			if n != 0:
				labels_new.append(y[i])
				x_expend = np.expand_dims(x[i], axis=0)
				y_frame_expend = np.expand_dims(y_frame[i], axis=0)
				x_new = np.concatenate((x_new, x_expend),axis=0)
				y_frame_new = np.concatenate((y_frame_new, y_frame_expend),axis=0)
		return x_new[1:], labels_new, y_frame_new[1:]



if __name__ == '__main__':
	max_len = 80
	x1 = get_feature_tensor(feature_dir,feature_name, max_len)
	y2 = get_labels(label_dir, label_name)
	y3 = get_frame_labels(feature_dir,feature_name, max_len)
	y4 = get_frame_01_labels(feature_dir,feature_name, max_len)
	x5, y5, y5f = non_zero_data(x1,y2, max_len, y3, use_y_frame=True)
	x6, y6 = non_zero_data(x1,y2, max_len, y3, use_y_frame=False)
	print x1.shape, len(y2), y3.shape, y4.shape
	print x5.shape, len(y5), y5f.shape
	print x6.shape, len(y6)


