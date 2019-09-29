from matplotlib import pyplot as plt
import json

with open('training_pretrained.json', 'r') as tr_js:
	training_stats = json.load(tr_js)
	plt.plot(training_stats['train_loss'], label='Train')
	plt.plot(training_stats['val_loss'], label='Validation')
	plt.title('Loss vs epoch')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend(loc='upper right')
	plt.show()
