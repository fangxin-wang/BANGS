import re

file_name_part = 'log/cautious-GCN-Cora-itr40-top50-seed6666'
location = file_name_part + '-m0-Conf.txt'

# loc2 = file_name_part + '-m0-Conf.txt'
# loc3 = file_name_part + '-m1-IGP.txt'
# loc4 = file_name_part + '-m0-IGP.txt'

with open(location, 'r') as file:
    log = file.read()

# Define regex patterns
noisy_portion_pattern = r"Noisy_portion: ([0-9.]+)"
test_results_pattern = r"Test set results: loss= [0-9.]+, accuracy= ([0-9.]+)"
best_test_accuracy_pattern = r"best test accuracy: ([0-9.]+)"
final_test_accuracy_pattern = r"final test accuracy: ([0-9.]+)"
early_stopped_acc_pattern = r"Best Acc Early Stopped by Valid Acc: ([0-9.]+)"

# Find all matches
noisy_portion = re.findall(noisy_portion_pattern, log)
# test_accuracies = re.findall(test_results_pattern, log)
best_test_accuracy = re.findall(best_test_accuracy_pattern, log)
final_test_accuracy = re.findall(final_test_accuracy_pattern, log)
early_stopped_acc = re.findall(early_stopped_acc_pattern, log)

# Store results in a dictionary
results = {}

for i in range(len(noisy_portion)):
    results[noisy_portion[i]] = [float(best_test_accuracy[i]), float(final_test_accuracy[i]),
                                 float(early_stopped_acc[i])]
    print(noisy_portion[i], best_test_accuracy[i], early_stopped_acc[i])
# Print results
print(results)
