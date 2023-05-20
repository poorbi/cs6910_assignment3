import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import wandb

# Define the actual and predicted words
data = data = pd.read_csv('predictions_attention.csv',sep=",",names=['actual_X', 'actual_Y','predicted_Y'])
actual_words = data['actual_Y'].tolist()
predicted_words = data['predicted_Y'].tolist()

# padding words
for i in range(len(actual_words)):
    maxlen = max(len(actual_words[i]),len(predicted_words[i]))
    actual_words[i] += ' ' * (maxlen - len(actual_words[i]))
    predicted_words[i] += ' ' * (maxlen - len(predicted_words[i]))

# Create the confusion matrix character-wise
actual_chars = [char for word in actual_words for char in word]
predicted_chars = [char for word in predicted_words for char in word]

# Pad the lists if they have different lengths
max_length = max(len(actual_chars), len(predicted_chars))
actual_chars += [''] * (max_length - len(actual_chars))
predicted_chars += [''] * (max_length - len(predicted_chars))

# Get unique characters as labels
unique_chars = sorted(set(actual_chars + predicted_chars))

# Create the confusion matrix
num_labels = len(unique_chars)

# Initialize confusion matrix
confusion_mat = [[0] * num_labels for _ in range(num_labels)]

# Update confusion matrix
for actual, predicted in zip(actual_chars, predicted_chars):
    actual_idx = unique_chars.index(actual)
    predicted_idx = unique_chars.index(predicted)
    # print(actual_idx,predicted_idx)
    confusion_mat[actual_idx][predicted_idx] += 1

# Set a font that supports Devanagari characters
font_path = "MANGAL.TTF"
font_prop = fm.FontProperties(fname=font_path)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(12, 10))
ax = sns.heatmap(confusion_mat, annot=False, cmap='viridis', cbar=False, square=True)

tick_locations = list(range(len(unique_chars)))
ax.set_xticks(tick_locations)
ax.set_yticks(tick_locations)

ax.set_xticklabels(unique_chars, rotation=90, fontproperties=font_prop, fontsize=8)
ax.set_yticklabels(unique_chars, rotation=0, fontproperties=font_prop, fontsize=8)
plt.title('Character-wise Confusion Matrix', fontsize=14)
plt.xlabel('Predicted Character', fontsize=12)
plt.ylabel('Actual Character', fontsize=12)

wandb.init(project="A3", entity="cs22m064",name="confusion_matrix_attention")
# Get the wandb log directory
log_dir = wandb.run.dir

# Save the figure to the log directory
figure_path = f"{log_dir}/confusion_matrix.png"
plt.savefig(figure_path)

# Log the figure in wandb
wandb.log({"confusion_matrix": wandb.Image(figure_path)})
wandb.finish()

# Display the figure
plt.show()
