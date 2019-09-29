import torchvision
import argparse
import os
import torch
from torch import nn, optim
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall
import helpers
import json
from tqdm import tqdm

from data_loader import load_data

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

NUM_EPOCH = 5
BATCH_SIZE = 50
log_interval = 10
train_accuracy = []
val_accuracy = []
train_loss = []
val_loss = []
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='FashionMNIST on pytorch')
	parser.add_argument('--test', dest='test_only', action='store_true', help='test model', default=False)
	parser.add_argument('--file', dest='test_file', help='test model file')
	parser.add_argument('--epochs', dest='num_epochs', help='number of epochs', type=int, default=NUM_EPOCH)
	parser.add_argument('--batchs', dest='batch_size', type=int, help='batch size', default=BATCH_SIZE)
	args = parser.parse_args()

	if args.test_only and (args.test_file is None):
		parser.error("--test requires --file")

	model = torchvision.models.resnet18(pretrained=True)
	for param in model.parameters():
		param.requires_grad = False
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, 2)
	parameters = model.fc.parameters()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(parameters, lr=1e-3)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	train_loader, val_loader, test_loader = load_data(batch_size=args.batch_size)

	model.to(device)
	trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
	evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(), 'loss': Loss(criterion)},
	                                        device=device)

	desc = "ITERATION - loss: {:.2f}"
	pbar = tqdm(initial=0, leave=False, total=len(train_loader), desc=desc.format(0))


	@trainer.on(Events.ITERATION_COMPLETED)
	def log_training_loss(engine):
		iter = (engine.state.iteration - 1) % len(train_loader) + 1

		if iter % log_interval == 0:
			pbar.desc = desc.format(engine.state.output)
			pbar.update(log_interval)


	@trainer.on(Events.EPOCH_COMPLETED)
	def log_training_results(engine):
		pbar.refresh()
		evaluator.run(train_loader)
		metrics = evaluator.state.metrics
		avg_accuracy = metrics['accuracy']
		avg_loss = metrics['loss']
		train_accuracy.append(avg_accuracy)
		train_loss.append(avg_loss)
		# precision = metrics['pre']
		# recall = metrics['recall']
		# F1 = (precision * recall * 2 / (precision + recall)).mean()
		tqdm.write("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
		           .format(engine.state.epoch, avg_accuracy, avg_loss))


	@trainer.on(Events.EPOCH_COMPLETED)
	def log_validation_results(engine):
		evaluator.run(val_loader)
		metrics = evaluator.state.metrics
		avg_accuracy = metrics['accuracy']
		avg_loss = metrics['loss']
		val_accuracy.append(avg_accuracy)
		val_loss.append(avg_loss)
		# precision = metrics['pre']
		# recall = metrics['recall']
		# F1 = (precision * recall * 2 / (precision + recall)).mean()
		tqdm.write(
			"Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
				.format(engine.state.epoch, avg_accuracy, avg_loss)
		)

		pbar.n = pbar.last_print_n = 0


	trainer.run(train_loader, max_epochs=args.num_epochs)
	pbar.close()

	tester = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(),
	                                                     'loss': Loss(criterion),
	                                                     'pre': Precision(average=True),
	                                                     'recall': Recall(average=False)
	                                                     }, device=device)
	tester.run(test_loader)
	metrics = tester.state.metrics
	test_accuracy = metrics['accuracy']
	test_loss = metrics['loss']
	print("Precision", metrics['pre'])
	print("Recall", metrics['recall'])
	print("Test Results - Avg accuracy: {:.2f} Avg loss: {:.2f}".format(test_accuracy, test_loss))
	stats = {
		'train_accuracy': train_accuracy,
		'train_loss': train_loss,
		'val_accuracy': val_accuracy,
		'val_loss': val_loss,
		'test_accuracy': test_accuracy,
		'test_loss': test_loss
	}
	with open('training_pretrained.json', 'w') as json_f:
		json.dump(stats, json_f)
