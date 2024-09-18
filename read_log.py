import os
import re

import pandas as pd


def parse_log_content(content):
    """Parse the log_before_12 content to extract specified metrics."""
    metrics = {}
    # Regex pattern to find the necessary lines and capture the required values
    pattern = re.compile(
        r".*INFO.* - original acc: (\d+\.\d+), best test accuracy: (\d+\.\d+), "
        r"final test accuracy: (\d+\.\d+),.*\n"
        r".*INFO.* - Best Acc Early Stopped by Valid Acc: (\d+\.\d+)"
    )

    # Find all matches in the content
    matches = pattern.findall(content)

    # Check if there are any matches
    if matches:
        # Get the last match from the list
        last_match = matches[-1]

        # Assign the captured values to the corresponding keys in the dictionary
        metrics['origin_accuracy'] = float(last_match[0])
        metrics['best_test_accuracy'] = float(last_match[1])
        metrics['final_test_accuracy'] = float(last_match[2])
        metrics['best_acc_early_stopped'] = float(last_match[3])

    return metrics


def read_logs_from_directory(directory):
    data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                content = file.read()
            metrics = parse_log_content(content)
            if metrics:
                data[filename] = metrics
    return dict(sorted(data.items()))  # Sort the dictionary by filename


log_data = read_logs_from_directory('log')


# print(log_data)
def extract_info_from_filename(filename):
    """Extract components from the filename using regex."""
    pattern = re.compile(
        r"cautious-(\w+)-(\w+)-itr(\d+)-top(\d+)-seed(\d+)-m(\d+)-(\w+)-ft(\w+)\.txt"
    )
    match = pattern.search(filename)
    if match:
        return {
            'Model': match.group(1),
            'Dataset': match.group(2),
            'Iteration': int(match.group(3)),
            'Node Num Each Iteration': int(match.group(4)),
            'Seed': int(match.group(5)),
            'Multiview': int(match.group(6)),
            'Selection Criterion': match.group(7),
            'Fine Tuning': match.group(8),
        }
    #print('No match in file name')
    return {}



# Create a list for DataFrame rows
rows = []

for filename, metrics in log_data.items():
    info = extract_info_from_filename(filename)
    if info:
        row = {**info, **metrics}  # Combine extracted info and metrics into one dictionary
        rows.append(row)

# Convert list of dictionaries to DataFrame
df = pd.DataFrame(rows)

# Reorder and rename columns according to the requirement
df = df[['Model', 'Dataset', 'Iteration', 'Node Num Each Iteration', 'Seed', 'Multiview', 'Selection Criterion', 'Fine Tuning',
         'origin_accuracy',
         'best_test_accuracy', 'final_test_accuracy', 'best_acc_early_stopped']]
df.columns = ['Model', 'Dataset', 'Iteration', 'Node Num', 'Seed', 'Multiview', 'Selection Criterion', 'FT',
              'Origin',
              'Best Test', 'Final', 'Early Stopped']

# print(df)
# df_select = df [ ~( (df ['Multiview'] == 0) & (df ['Selection Criterion'] == 'Conf')) & (df['Model'] == 'GCN') ]
# df_select = df[ (df['Node Num'] == 100) & (df['Dataset']=='Cora') & (df['Model']== 'GCN') &(df.Seed.isin([ 1204,1234,1111,8888,6666]) ) ]
df_select = df[(df.Dataset.isin (['Flickr','obgnarxiv']) ) &(df.Seed.isin([910, 911, 912, 913, 914, 915])) & (df['Iteration'] == 40)]
#df_select = df[ (df.Seed.isin([910, 911, 912])) & (df ['Selection Criterion']!= 'random') ]
# df_select = df[  (df.Seed.isin([ 1204,1234,1111,8888,6666]) ) & (df['Iteration']==40) & (df['Model'] != 'GCN')]

# print(df_select)


df_select[['Origin', 'Best Test', 'Early Stopped']] = df_select[['Origin', 'Best Test', 'Early Stopped']] * 100

df_select['improve'] = df_select['Best Test'] - df_select['Origin']
print(df_select)

res = df_select.groupby(['Dataset', 'Model', 'Node Num', 'Multiview', 'Selection Criterion','FT'])[
    ['Origin', 'Best Test', 'Early Stopped']].agg(['mean', 'std', 'count'])
print(res)
