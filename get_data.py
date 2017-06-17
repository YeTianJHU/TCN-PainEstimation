import scipy.io as sio
import os
import numpy as np

feature_dir = '/home/ye/Works/pain'
label_dir = '/home/ye/Works/pain/Sequence_Labels'
feature_name = 'feature_from_verification_model.mat'
label_name = 'OPR'

def downsampling(input_data, after_len):
	mid_index = int(len(input_data[0])/2)
	if len(input_data[0]) > after_len:
		after = input_data.T[mid_index-after_len/2:mid_index+after_len/2]
		after = after.T
	elif len(input_data[0]) <= after_len:
		# after = np.zeros((len(input_data[1]),len(input_data[0])))
		after = np.lib.pad(input_data, ((0,0),(0,after_len-len(input_data[0]))), 'constant', constant_values=(0,))
	return after.T

def get_feature_tensor(feature_dir,feature_name, max_len):
	feature = sio.loadmat(os.path.join(feature_dir, feature_name))
	x = np.zeros((1,max_len,512))
	for person, video_feature in enumerate(feature['video_feature'].T[0]):
		# print 'person: ',person,'video_feature: ', video_feature.shape
		for id, vid in enumerate(video_feature):
			vid_mat = vid[0].T
			# print 'vid_mat', vid_mat.shape
			vid_mat_mid = downsampling(vid_mat.T, max_len)
			vid_mat_extend = np.expand_dims(vid_mat_mid, axis=0)
			# print 'vid_mat_mid', vid_mat_mid.shape
			# print 'vid_mat_extend', vid_mat_extend.shape
			# print 'x', x.shape
			x = np.concatenate((x, vid_mat_extend),axis=0)
		# 	print ''
		# print 'x: ', x.shape
		# print '====================='
		# print ''
	return x[1:]

def get_labels(label_dir, label_name):
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
	np.set_printoptions(threshold='nan')
	feature = sio.loadmat(os.path.join(feature_dir, feature_name))
	x = np.zeros((1,max_len,1))
	for person, video_feature in enumerate(feature['video_pain_level'].T[0]):
		# print 'person: ',person,'video_feature: ', video_feature.shape
		for id, vid in enumerate(video_feature):
			vid_mat = vid[0]
			# print 'vid_mat', vid_mat
			vid_mat_mid = downsampling(vid_mat, max_len)
			vid_mat_extend = np.expand_dims(vid_mat_mid, axis=0)
			# print 'vid_mat_mid', vid_mat_mid.shape
			# print 'vid_mat_extend', vid_mat_extend.shape
			# print 'x', x.shape
			x = np.concatenate((x, vid_mat_extend.T),axis=0)
		# 	print ''
		# print 'x: ', x.shape
		# print '====================='
		# print ''
	return x[1:]

def get_frame_01_labels(feature_dir,feature_name, max_len):
	feature = sio.loadmat(os.path.join(feature_dir, feature_name))
	x = np.zeros((1,max_len,1))
	for person, video_feature in enumerate(feature['video_pain_level'].T[0]):
		# print 'person: ',person,'video_feature: ', video_feature.shape
		for id, vid in enumerate(video_feature):
			vid_mat = vid[0]
			for i, lab in enumerate(vid_mat[0]):
				if lab>0:
					vid_mat[0][i] = 1

			# print 'vid_mat[0]',vid_mat[0]
			# print 'vid_mat', vid_mat.shape
			vid_mat_mid = downsampling(vid_mat, max_len)
			vid_mat_extend = np.expand_dims(vid_mat_mid, axis=0)
			# print 'vid_mat_mid', vid_mat_mid.shape
			# print 'vid_mat_extend', vid_mat_extend.shape
			# print 'x', x.shape
			x = np.concatenate((x, vid_mat_extend.T),axis=0)
		# 	print ''
		# print 'x: ', x.shape
		# print '====================='
		# print ''
	return x[1:]

def non_zero_data(x,y,max_len, y_frame, use_y_frame=False):
	x_new = np.zeros((1,max_len,512))
	if use_y_frame == False:
		labels_new = []
		for i, n in enumerate(y):
			if n != 0:
				labels_new.append(y[i])
				x_expend = np.expand_dims(x[i], axis=0)
				# y_frame_expend = np.expand_dims(y_frame[i], axis=0)
				# print 'x_new', x_new.shape, 'x[i]', x[i].shape, 'x_expend', x_expend.shape
				x_new = np.concatenate((x_new, x_expend),axis=0)
				# y_frame_new = np.concatenate((y_frame_new, y_frame_expend),axis=0)
		return x_new[1:], labels_new
	elif use_y_frame == True:
		y_frame_new = np.zeros((1,max_len,1))
		labels_new = []
		for i, n in enumerate(y):
			if n != 0:
				labels_new.append(y[i])
				x_expend = np.expand_dims(x[i], axis=0)
				y_frame_expend = np.expand_dims(y_frame[i], axis=0)
				# print 'x_new', x_new.shape, 'x[i]', x[i].shape, 'x_expend', x_expend.shape, 'y_frame_expend',y_frame_expend.shape
				x_new = np.concatenate((x_new, x_expend),axis=0)
				y_frame_new = np.concatenate((y_frame_new, y_frame_expend),axis=0)
		return x_new[1:], labels_new, y_frame_new[1:]






	

y = get_frame_labels(feature_dir,feature_name, 100)
# x = get_frame_01_labels(feature_dir,feature_name,400)
# y = get_labels(label_dir, label_name)
# x_new, labels_new = non_zero_data(x,y,400)
# print x_new.shape, labels_new
# x = get_frame_01_labels(feature_dir,feature_name,400)
# print x.shape