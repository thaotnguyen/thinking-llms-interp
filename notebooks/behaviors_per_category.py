# %%

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re

# Load data
with open('../data/tasks.json', 'r') as f:
    tasks = json.load(f)

with open('../data/annotated_responses_deepseek-r1-distill-llama-8b.json', 'r') as f:
    responses = json.load(f)['responses']

# %%

# Create task lookup dictionary
task_dict = {task['task_uuid']: task['task_category'] for task in tasks}

# Initialize data structure to store counts
category_behavior_counts = defaultdict(lambda: defaultdict(int))
category_total_sections = defaultdict(int)

# Pattern to match labeled sections
pattern = r'\["([\w-]+)"\](.*?)\["end-section"\]'

# Process each response
for response in responses:
    category = task_dict[response['task_uuid']]
    text = response['annotated_response']
    
    # Find all labeled sections
    matches = re.finditer(pattern, text, re.DOTALL)
    
    # Count behaviors for this response
    for match in matches:
        behavior = match.group(1)
        category_behavior_counts[category][behavior] += 1
        category_total_sections[category] += 1

# Convert counts to fractions
categories = sorted(set(task_dict.values()))
behaviors = ['initializing', 'deduction', 'adding-knowledge', 
            'example-testing', 'uncertainty-estimation', 'backtracking']

fractions = {category: {behavior: category_behavior_counts[category][behavior] / category_total_sections[category] 
                       if category_total_sections[category] > 0 else 0
                       for behavior in behaviors}
            for category in categories}

# Plotting
fig, ax = plt.subplots(figsize=(15, 8))
x = np.arange(len(categories))
width = 0.12  # Width of bars
multiplier = 0

# Plot bars for each behavior
for behavior in behaviors:
    behavior_fractions = [fractions[category][behavior] for category in categories]
    offset = width * multiplier
    rects = ax.bar(x + offset, behavior_fractions, width, label=behavior)
    multiplier += 1

# Customize plot
ax.set_ylabel('Fraction of Sections')
ax.set_title('Distribution of Behavioral Patterns Across Task Categories')
ax.set_xticks(x + width * 2.5)
ax.set_xticklabels([cat.replace('-', '\n') for cat in categories], rotation=0)
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
ax.grid(True, axis='y', alpha=0.3)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save plot
plt.savefig('../figures/behavior_distribution.pdf', bbox_inches='tight', dpi=300)
plt.show()

