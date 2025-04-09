import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.ticker import MaxNLocator

def read_csv_results(filepath):
    """Reads a CSV file and returns a dictionary of categories and corresponding indicators"""
    results = {}
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['class'] not in ['accuracy', 'macro avg', 'weighted avg']:
                results[row['class']] = {
                    'precision': float(row['precision']),
                    'recall': float(row['recall']),
                    'f1-score': float(row['f1-score']),
                    'support': float(row['support'])
                }
    return results

# Read the result files of two models
pointnet_results = read_csv_results('classification_report_10.csv')
pointnet_plus_results = read_csv_results('classification2_report_10_update2.csv')

# Get a list of common categories (sorted alphabetically)
common_classes = sorted(set(pointnet_results.keys()) & set(pointnet_plus_results.keys()))
if len(common_classes) != 40:
    print(f"Warning: only find {len(common_classes)} classes，Expected 40")

# Prepare drawing data
pointnet_scores = [pointnet_results[cls]['f1-score'] for cls in common_classes]
pointnet_plus_scores = [pointnet_plus_results[cls]['f1-score'] for cls in common_classes]
differences = [pp - pn for pp, pn in zip(pointnet_plus_scores, pointnet_scores)]

# --- Highlight processing logic ---
# Find the N classes with the largest performance improvement
top_n = 5
top_improvements = sorted(zip(common_classes, differences), key=lambda x: x[1], reverse=True)[:top_n]
highlight_classes = [cls for cls, diff in top_improvements]

plt.figure(figsize=(22, 10))
x = np.arange(len(common_classes))
width = 0.35

# Creating a Bar Chart
rects1 = plt.bar(x - width/2, pointnet_scores, width, label='PointNet', color='#1f77b4', edgecolor='gray')
rects2 = plt.bar(x + width/2, pointnet_plus_scores, width, label='PointNet++', color='#ff7f0e', edgecolor='gray')

# --- Highlighting significantly improved columns ---
for i, cls in enumerate(common_classes):
    if cls in highlight_classes:
        # Add red border
        rects2[i].set_edgecolor('red')
        rects2[i].set_linewidth(3)
        # Add an asterisk to the top of the column
        plt.text(x[i] + width/2, pointnet_plus_scores[i] + 0.02,
                '★', ha='center', va='bottom', color='red', fontsize=14)


plt.xlabel('Classes', fontsize=14)
plt.ylabel('F1-Score', fontsize=14)
plt.title('Model Performance Comparison (F1-Score) on ModelNet40\n★: Top 5 Improved Classes', fontsize=16, pad=20)
plt.xticks(x, common_classes, rotation=90, fontsize=10)
plt.yticks(fontsize=12)
plt.gca().yaxis.set_major_locator(MaxNLocator(20))
plt.legend(loc='upper right', fontsize=12)


plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add value labels above the columns
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8, rotation=45)

add_labels(rects1)
add_labels(rects2)


plt.tight_layout()

# Print performance changes
print("\nTop 5 classes with the biggest performance improvements:")
for cls, diff in top_improvements:
    print(f"{cls}: +{diff:.3f} (PointNet: {pointnet_results[cls]['f1-score']:.3f} → PointNet++: {pointnet_plus_results[cls]['f1-score']:.3f})")

if any(d < 0 for d in differences):
    top_declines = sorted([(c, d) for c, d in zip(common_classes, differences) if d < 0], key=lambda x: x[1])[:5]
    print("\nClasses of performance degradation:")
    for cls, diff in top_declines:
        print(f"{cls}: {diff:.3f}")


plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()