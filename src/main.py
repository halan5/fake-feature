### importing important libraries
import sys
import os
from hyperopt import hp, tpe, fmin

from utils import *



def train():
	args=parse_args()
	sys.path.append('../parameters/'+args.dataset_name+'/')
	from parameters import hyper_para
	init_hyper_para(hyper_para)
	handle_args(args)
	
	
	train_loader, valid_loader, test_loader_kwn, test_loader_unk=get_loaders()
	
	F, losses, fake_features=train_model(train_loader, hyper_para.kwn)
	train_features, train_label=run_model(F, train_loader)
	validation_features, validation_label=run_model(F, valid_loader)
	

	now = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
	os.mkdir('../save_folder/'+hyper_para.dataset_name+'/'+now)
	sio.savemat('../save_folder/'+hyper_para.dataset_name+'/'+now+'/losses.mat', {'losses': losses.numpy()})
	sio.savemat('../save_folder/'+hyper_para.dataset_name+'/'+now+'/classes.mat', {'kwn': hyper_para.kwn, 'unk': hyper_para.unk})
	save((train_features, train_label, validation_features, validation_label, fake_features), ('training_features', 'training_labels', 'validation_features', 'validation_labels', 'fake_features'), now)
	test_features_kwn, test_labels_kwn=run_model(F, test_loader_kwn)
	test_features_unk, _=run_model(F, test_loader_unk)
	save((test_features_kwn, test_labels_kwn, test_features_unk), ('test_features_kwn', 'test_labels_kwn', 'test_features_unk'), now)
if __name__ == "__main__":
	train()