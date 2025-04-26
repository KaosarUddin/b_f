#figure2...
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
# Tau values for the x-axis
tau_values = [100, 200, 300, 400, 500, 600, 700, 800, 900]
# Data for each distance type and task
tasks_ai_distance = {
'REST': [0.8820, 0.8984, 0.8820, 0.9042, 0.9171, 0.9299, 0.9241, 0.9346, 0.9404],
'EMOTION': [0.1238, 0.1133, 0.2021, 0.2395, 0.2605, 0.2617, 0.2769, 0.2792, 0.2780],
'GAMBLING': [0.4019, 0.2523, 0.4591, 0.5035, 0.5012, 0.5082, 0.5140, 0.4895, 0.4860],
'LANGUAGE': [0.6787, 0.5561, 0.6343, 0.6729, 0.6904, 0.7044, 0.6998, 0.7009, 0.7056],
'MOTOR': [0.4603, 0.2652, 0.2558, 0.2780, 0.2944, 0.2956, 0.3026, 0.3026, 0.3096],
'RELATIONAL': [0.2909, 0.2839, 0.4147, 0.4544, 0.4743, 0.4790, 0.4953, 0.4930, 0.4825],
'SOCIAL': [0.5806, 0.3680, 0.5596, 0.5888, 0.6075, 0.6238, 0.6355, 0.6285, 0.6215],
'WM': [0.6904, 0.6682, 0.5958, 0.5993, 0.6180, 0.6343, 0.6355, 0.6402, 0.6659]
}
tasks_log_euclidean = {
'REST': [0.8411, 0.9147, 0.9357, 0.8949, 0.8902, 0.9077, 0.9077, 0.9159, 0.9194],
'EMOTION': [0.2325, 0.1157, 0.2105, 0.2266, 0.2488, 0.2406, 0.2523, 0.2488, 0.2395],
'GAMBLING': [0.4720, 0.5981, 0.4451, 0.5023, 0.4871, 0.5082, 0.5175, 0.4603, 0.4322],
'LANGUAGE': [0.7126, 0.8143, 0.7255, 0.6612, 0.6787, 0.6892, 0.6986, 0.6764, 0.6812],
'MOTOR': [0.5222, 0.6437, 0.3832, 0.2745, 0.2803, 0.2850, 0.2827, 0.2757, 0.2512],
'RELATIONAL': [0.3703, 0.4591, 0.5140, 0.5417, 0.4661, 0.4509, 0.4836, 0.4673, 0.4521],
'SOCIAL': [0.6098, 0.6951, 0.5456, 0.5643, 0.5841, 0.5829, 0.6192, 0.5912, 0.5923],
'WM': [0.6787, 0.7675, 0.8107, 0.6624, 0.5935, 0.6262, 0.6379, 0.6402, 0.6507]
}
tasks_alpha_z = {
'REST': [0.7850, 0.9264, 0.9591, 0.9813, 0.9883, 0.9918, 0.9942, 0.9965, 0.9977],
'EMOTION': [0.3423, 0.5666, 0.6640, 0.7196, 0.7383, 0.7605, 0.7664, 0.7745, 0.7979],
'GAMBLING': [0.5631, 0.8072, 0.8867, 0.9159, 0.9322, 0.9490, 0.9509, 0.9533, 0.9603],
'LANGUAGE': [0.7126, 0.8890, 0.9498, 0.9579, 0.9661, 0.9650, 0.9731, 0.9755, 0.9778],
'MOTOR': [0.4264, 0.6939, 0.8341, 0.8879, 0.9077, 0.9241, 0.9381, 0.9404, 0.9439],
'RELATIONAL': [0.5011, 0.7547, 0.8668, 0.8995, 0.9124, 0.9241, 0.9322, 0.9381, 0.9416],
'SOCIAL': [0.6437, 0.8283, 0.9147, 0.9439, 0.9521, 0.9638, 0.9708, 0.9731, 0.9685],
'WM': [0.6262, 0.8213, 0.9030, 0.9369, 0.9533, 0.9661, 0.9685, 0.9731, 0.9778]
}
tasks_alpha_procrust = {
'REST': [0.8423, 0.9229, 0.9591, 0.8925, 0.8376, 0.8563, 0.8598, 0.8598, 0.8738],
'EMOTION': [0.2336, 0.1893, 0.2582, 0.2780, 0.3026, 0.3213, 0.3563, 0.3657, 0.3797],
'GAMBLING': [0.4755, 0.6016, 0.4708, 0.5140, 0.5397, 0.5666, 0.5818, 0.5923, 0.6145],
'LANGUAGE': [0.7114, 0.8201, 0.6904, 0.6812, 0.6998, 0.7150, 0.7255, 0.7430, 0.7558],
'MOTOR': [0.4860, 0.6425, 0.3750, 0.3808, 0.4112, 0.4357, 0.4544, 0.4673, 0.4953],
'RELATIONAL': [0.3914, 0.4521, 0.4311, 0.4708, 0.4988, 0.5035, 0.5327, 0.5432, 0.5654],
'SOCIAL': [0.6040, 0.6951, 0.5584, 0.6121, 0.6355, 0.6577, 0.6624, 0.6822, 0.6927],
'WM': [0.6636, 0.7792, 0.8177, 0.6484, 0.6659, 0.6951, 0.6986, 0.7114, 0.7255]
}
tasks_bw_distance = {
'REST': [0.6881, 0.7862, 0.8364, 0.8458, 0.8400, 0.8563, 0.8423, 0.8493, 0.8493],
'EMOTION': [0.1484, 0.2009, 0.2407, 0.2465, 0.2453, 0.2605, 0.2664, 0.25, 0.2547],
'GAMBLING': [0.3411, 0.3972, 0.4755, 0.3890, 0.3890, 0.3949, 0.4019, 0.3820, 0.4007],
'LANGUAGE': [0.5199, 0.5935, 0.6495, 0.6308, 0.6308, 0.6355, 0.6390, 0.6262, 0.6308],
'MOTOR': [0.2360, 0.2815, 0.3341, 0.3610, 0.3750, 0.3843, 0.3984, 0.3995, 0.4206],
'RELATIONAL': [0.2886, 0.3598, 0.4159, 0.4112, 0.4030, 0.4019, 0.4217, 0.4042, 0.4334],
'SOCIAL': [0.3949, 0.4907, 0.5654, 0.5432, 0.5304, 0.5397, 0.5397, 0.5409, 0.5479],
'WM': [0.4579, 0.5444, 0.6040, 0.6075, 0.6063, 0.6098, 0.6016, 0.5853, 0.5806]
}

# Update matplotlib parameters for publication-ready appearance
rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10
})

# Dummy data setup (replace with actual data)
tau_values = [100, 200, 300, 400, 500, 600, 700, 800, 900]
tasks = ['REST', 'EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']

# Define marker styles and background color
marker_styles = ['o', 's', '^', 'v', 'D', '*', 'P', 'X']
bg_color = 'white'

# Create the subplots
fig, axs = plt.subplots(3, 2, figsize=(15, 12))
fig.patch.set_facecolor('lightyellow')

# List of task data and titles
plot_data = [
(tasks_ai_distance, 'AI'),
(tasks_log_euclidean, 'Log-Euclidean'),
(tasks_alpha_z, 'Alpha-Z'),
(tasks_alpha_procrust, 'Alpha-Procrust'),
(tasks_bw_distance, 'BW')
]

# Plotting
for i, (task_data, title) in enumerate(plot_data):
    row, col = divmod(i, 2)
    ax = axs[row, col]
    ax.set_facecolor(bg_color)
    
    for j, (task, values) in enumerate(task_data.items()):
        marker = marker_styles[j % len(marker_styles)]
        ax.plot(tau_values, values, marker=marker, label=task, linewidth=1)
    
    ax.set_title(title)
    ax.set_ylim(0, 1)

    # Show x-axis label only for bottom row
    if row == 2:
        ax.set_xlabel('Parcellations')
    else:
        ax.set_xlabel('')

    # Show y-axis label only for leftmost column
    if col == 0:
        ax.set_ylabel('ID Rate')
    else:
        ax.set_ylabel('')

# Remove the last (6th) empty subplot
fig.delaxes(axs[2, 1])

# Create a combined legend on the bottom right
handles, labels = axs[1, 1].get_legend_handles_labels()
legend = fig.legend(
    handles,
    labels,
    loc='center right',
    title="Tasks",
    ncol=3,
    bbox_to_anchor=(0.9, 0.20),
    frameon=True,
    borderpad=1.5,
    fontsize=10,
    title_fontsize=12
)

# Style the legend box
legend.get_frame().set_facecolor('lightcyan')
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(1.5)

# Adjust layout
fig.suptitle('ID Rate Comparisons Across Different Metrics', fontsize=14, fontweight="bold", y=0.96)
plt.tight_layout(rect=[0.06, 0.06, 0.95, 0.95])

# Save the figure
fig.savefig("D:/Research AU/figure21.pdf", bbox_inches='tight')
fig.savefig("D:/Research AU/figure21.svg", bbox_inches='tight')
plt.show()

# Other style for figure 2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Style configuration
rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10
})

# Parameters
tau_values = [100, 200, 300, 400, 500, 600, 700, 800, 900]
tasks = ['REST', 'EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
marker_styles = ['o', 's', '^', 'v', 'D', '*', 'P', 'X']
colors = ['blue', 'orange', 'green', 'purple', 'brown', 'red', 'pink', 'gray']

# Replace with your actual task data dictionaries
plot_data = [
    (tasks_ai_distance, 'AI'),
    (tasks_log_euclidean, 'Log-Euclidean'),
    (tasks_alpha_z, 'Alpha-Z'),
    (tasks_alpha_procrust, 'Alpha-Procrust'),
    (tasks_bw_distance, 'BW')
]

# Plot setup
fig, axs = plt.subplots(2, 3, figsize=(14, 8))
fig.patch.set_facecolor('lightyellow')
axs = axs.flatten()

# Main plotting loop
for i, (task_data, title) in enumerate(plot_data):
    ax = axs[i]
    ax.set_facecolor('white')

    for j, task in enumerate(tasks):
        ax.plot(tau_values, task_data[task],
                label=task,
                marker=marker_styles[j % len(marker_styles)],
                color=colors[j % len(colors)],
                linewidth=1.2,
                markersize=4)

    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_xlim(100, 900)
    ax.set_xticks(tau_values)

    row, col = divmod(i, 3)

    # Label only last row
    if row == 1:
        ax.set_xlabel('Parcellations')
    else:
        ax.set_xlabel('')

    # Label only first column
    if col == 0:
        ax.set_ylabel('ID Rate')
    else:
        ax.set_ylabel('')

# Use last subplot (6th slot) for legend
axs[-1].axis('off')
handles, labels = axs[0].get_legend_handles_labels()
legend = axs[-1].legend(
    handles, labels,
    loc='center',
    title='Tasks',
    ncol=2,
    frameon=True,
    fontsize=10,
    title_fontsize=10,
    markerscale=1
)
legend.get_frame().set_facecolor('lightcyan')
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(1)

# Layout and export
plt.tight_layout(rect=[0.03, 0.03, 0.98, 0.95])
fig.suptitle('ID Rate Comparisons Across Different Metrics', fontsize=14, fontweight="bold", y=0.96)
fig.savefig("D:/Research AU/figure2..pdf", bbox_inches='tight')
fig.savefig("D:/Research AU/figure2..svg", bbox_inches='tight')
plt.show()



##figure3

###
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline
import matplotlib as mpl

parcellations = [100, 200, 300, 400, 500, 600, 700, 800, 900]
tasks_data_page_1 = [
     {
         'AI Distance': [0.8820, 0.8984, 0.8820, 0.9042, 0.9171, 0.9299, 0.9241, 0.9346, 0.9404],
         'BW Distance': [0.6881, 0.7862, 0.8364, 0.8458, 0.8400, 0.8563, 0.8423, 0.8493, 0.8493],
         'Log-Euclidean': [0.8411, 0.9147, 0.9357, 0.8949, 0.8902, 0.9077, 0.9077, 0.9159, 0.9194],
         'Alpha_Z': [0.7850, 0.9264, 0.9591, 0.9813, 0.9883, 0.9918, 0.9942, 0.9965, 0.9977],
         'Alpha Procrust': [0.8423, 0.9229, 0.9591, 0.8925, 0.8376, 0.8563, 0.8598, 0.8598, 0.8738]
     },
     {
         'AI Distance': [0.6787, 0.5561, 0.6343, 0.6729, 0.6904, 0.7044, 0.6998, 0.7009, 0.7056],
        'BW Distance': [0.5199, 0.5935, 0.6495, 0.6308, 0.6308, 0.6355, 0.6390, 0.6262, 0.6308],
         'Log-Euclidean': [0.7126, 0.8143, 0.7255, 0.6612, 0.6787, 0.6892, 0.6986, 0.6764, 0.6810],
         'Alpha_Z': [0.7126, 0.8890, 0.9498, 0.9579, 0.9661, 0.9650, 0.9731, 0.9755, 0.9778],
         'Alpha Procrust': [0.7114, 0.8201, 0.6904, 0.6812, 0.6998, 0.7150, 0.7255, 0.7430, 0.7558]
     },
     {
         'AI Distance': [0.5806, 0.3680, 0.5596, 0.5888, 0.6075, 0.6238, 0.6355, 0.6285, 0.6215],
         'BW Distance': [0.3949, 0.4907, 0.5654, 0.5432, 0.5304, 0.5397, 0.5397, 0.5409, 0.5479],
         'Log-Euclidean': [0.6098, 0.6951, 0.5456, 0.5643, 0.5841, 0.5829, 0.6192, 0.5912, 0.5923],
         'Alpha_Z': [0.6437, 0.8283, 0.9147, 0.9439, 0.9521, 0.9638, 0.9708, 0.9731, 0.9685],
        'Alpha Procrust': [0.6040, 0.6951, 0.5584, 0.6121, 0.6355, 0.6577, 0.6624, 0.6822, 0.6927]
     },
     {
         'AI Distance': [0.6904, 0.6682, 0.5958, 0.5993, 0.6180, 0.6343, 0.6355, 0.6402, 0.6659],
         'BW Distance': [0.4579, 0.5444, 0.6040, 0.6075, 0.6063, 0.6098, 0.6016, 0.5853, 0.5806],
         'Log-Euclidean': [0.6787, 0.7675, 0.8107, 0.6624, 0.5935, 0.6262, 0.6379, 0.6402, 0.6507],
         'Alpha_Z': [0.6262, 0.8213, 0.9030, 0.9369, 0.9533, 0.9661, 0.9685, 0.9731, 0.9778],
         'Alpha Procrust': [0.6636, 0.7792, 0.8177, 0.6484, 0.6659, 0.6951, 0.6986, 0.7114, 0.7255]
     }
 ]

titles_page_1 = ['REST', 'LANGUAGE', 'SOCIAL', 'WM']
parcellations = [100, 200, 300, 400, 500, 600, 700, 800, 900]
tasks_data_page_2 = [
    {
        'AI Distance': [0.1238, 0.1133, 0.2021, 0.2395, 0.2605, 0.2617, 0.2769, 0.2792, 0.2780],
        'BW Distance': [0.1484, 0.2009, 0.2407, 0.2465, 0.2453, 0.2605, 0.2664, 0.25, 0.2547],
        'Log-Euclidean': [0.2325, 0.1157, 0.2105, 0.2266, 0.2488, 0.2406, 0.2523, 0.2488, 0.2395],
        'Alpha_Z': [0.3423, 0.5666, 0.6640, 0.7196, 0.7383, 0.7605, 0.7664, 0.7745, 0.7979],
        'Alpha Procrust': [0.2336, 0.1893, 0.2582, 0.2780, 0.3026, 0.3213, 0.3563, 0.3657, 0.3797]
    },
    {
        'AI Distance': [0.4019, 0.2523, 0.4591, 0.5035, 0.5012, 0.5082, 0.5140, 0.4895, 0.4860],
        'BW Distance': [0.3411, 0.3972, 0.4755, 0.3890, 0.3890, 0.3949, 0.4019, 0.3820, 0.4007],
        'Log-Euclidean': [0.4720, 0.5981, 0.4451, 0.5023, 0.4871, 0.5082, 0.5175, 0.4603, 0.4322],
        'Alpha_Z': [0.5631, 0.8072, 0.8867, 0.9159, 0.9322, 0.9490, 0.9509, 0.9533, 0.9603],
        'Alpha Procrust': [0.4755, 0.6016, 0.4708, 0.5140, 0.5397, 0.5666, 0.5818, 0.5923, 0.6145]
    },
    {
        'AI Distance': [0.4603, 0.2652, 0.2558, 0.2780, 0.2944, 0.2956, 0.3026, 0.3026, 0.3095],
        'BW Distance': [0.2360, 0.2815, 0.3341, 0.3610, 0.3750, 0.3843, 0.3984, 0.3995, 0.4206],
        'Log-Euclidean': [0.5222, 0.6437, 0.3832, 0.2745, 0.2803, 0.2850, 0.2827, 0.2757, 0.2512],
        'Alpha_Z': [0.4264, 0.6939, 0.8341, 0.8879, 0.9077, 0.9241, 0.9381, 0.9404, 0.9439],
        'Alpha Procrust': [0.4860, 0.6425, 0.3750, 0.3808, 0.4112, 0.4357, 0.4544, 0.4673, 0.4953]
    },
    {
        'AI Distance': [0.2909, 0.2839, 0.4147, 0.4544, 0.4743, 0.4790, 0.4953, 0.4930, 0.4825],
        'BW Distance': [0.2886, 0.3598, 0.4159, 0.4112, 0.4030, 0.4019, 0.4217, 0.4042, 0.4334],
        'Log-Euclidean': [0.3703, 0.4591, 0.5140, 0.5417, 0.4661, 0.4509, 0.4836, 0.4673, 0.4521],
        'Alpha_Z': [0.5011, 0.7547, 0.8668, 0.8995, 0.9124, 0.9241, 0.9322, 0.9381, 0.9416],
        'Alpha Procrust': [0.3914, 0.4521, 0.4311, 0.4708, 0.4988, 0.5035, 0.5327, 0.5432, 0.5654]
    }
]

titles_page_2 = ['EMOTION', 'GAMBLING', 'MOTOR', 'RELATIONAL']

def plot_tasks_for_iscience(tasks_data, titles, parcellations, legend_position='bottom'):
    # Clean, journal-style formatting
    mpl.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10
    })

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6), dpi=300)
    fig.patch.set_facecolor('lightyellow')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'v', 'D']
    linestyles = ['-', '--', '-.', ':', '-']
    distance_labels = list(tasks_data[0].keys())

    sample_lines = {}

    for idx, (ax, data_dict, title) in enumerate(zip(axs.flat, tasks_data, titles)):
        for i, distance_type in enumerate(distance_labels):
            x = np.array(parcellations)
            y = np.array(data_dict[distance_type])

            if len(x) >= 4:
                x_smooth = np.linspace(x.min(), x.max(), 100)
                y_smooth = make_interp_spline(x, y, k=3)(x_smooth)
            else:
                x_smooth, y_smooth = x, y

            ax.plot(x_smooth, y_smooth,
                    color=colors[i],
                    linestyle=linestyles[i],
                    linewidth=1.2)

            ax.scatter(x, y,
                       color=colors[i],
                       edgecolor='black',
                       linewidth=0.4,
                       marker=markers[i],
                       s=25,
                       zorder=3)

            if distance_type not in sample_lines:
                sample_lines[distance_type] = Line2D(
                    [0], [0],
                    color=colors[i],
                    marker=markers[i],
                    linestyle=linestyles[i],
                    markersize=4,         # smaller marker
                    linewidth=1.2,
                    label=distance_type
                )

        ax.set_title(f"{title}", fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_xlim(min(parcellations), max(parcellations))
        if row == 2:
                ax.set_xticks(parcellations)
                ax.set_xticklabels(parcellations, rotation=45)
        else:
             ax.set_xticklabels([])
        ax.tick_params(axis='both', labelsize=11)
        for spine in ax.spines.values():
            spine.set_alpha(0.3)

    # Shared axis labels
    fig.text(0.5, 0.02, 'Number of Parcellations', ha='center', fontsize=13)
    fig.text(0.02, 0.5, 'ID Rate', va='center', rotation='vertical', fontsize=13)

    # Global legend placement
    if legend_position == 'bottom':
        fig.legend(
            handles=sample_lines.values(),
            loc='upper center',
            ncol=len(distance_labels),
            bbox_to_anchor=(0.5, -0.03),
            title='Distance Metrics',
            title_fontsize=9,
            fontsize=8,
            frameon=True,
            handlelength=2.0,
            handletextpad=0.8,
            columnspacing=1.0,
            labelspacing=0.4
        )
       # plt.tight_layout(rect=[0.05, 0.18, 0.95, 0.92])  # leave space at bottom
    elif legend_position == 'right':
        fig.legend(
            handles=sample_lines.values(),
            loc='center left',
            bbox_to_anchor=(.95, 0.5),
            title='Distance Metrics',
            title_fontsize=10,
            fontsize=9,
            frameon=True
        )
        #plt.tight_layout(rect=[0.05, 0.05, 0.88, 0.95])  # leave space at right

    # Title and export
    plt.tight_layout(rect=[0.06, 0.06, 0.95, 0.95])
    fig.suptitle("Best Performance's Tasks", fontsize=14, fontweight='bold', y=0.96)
    fig.savefig("D:/Research AU/figure3.pdf", bbox_inches='tight')
    fig.savefig("D:/Research AU/figure3.svg", bbox_inches='tight')
    plt.show()


# Plot the first set of tasks
plot_tasks_for_iscience(tasks_data_page_1, titles_page_1, parcellations)
# Plot the second set of tasks
plot_tasks_for_iscience(tasks_data_page_2, titles_page_2, parcellations)

# all task in one page
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib as mpl
from matplotlib.lines import Line2D
def plot_3x3_with_legend(tasks_data, titles, parcellations, panel_title="Tasks Panel"):
    # --- Styling for publication ---
    mpl.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9
    })

    fig, axs = plt.subplots(3, 3, figsize=(10, 8), dpi=300)
    fig.patch.set_facecolor('lightyellow')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'v', 'D']
    linestyles = ['-', '--', '-.', ':', '-']
    distance_labels = list(tasks_data[0].keys())
    sample_lines = {}

    for idx in range(9):
        row, col = divmod(idx, 3)
        ax = axs[row, col]

        if idx < 8:
            data_dict = tasks_data[idx]
            title = titles[idx]

            for i, dist in enumerate(distance_labels):
                x = np.array(parcellations)
                y = np.array(data_dict[dist])

                # Smooth if possible
                if len(x) >= 4:
                    x_smooth = np.linspace(x.min(), x.max(), 100)
                    y_smooth = make_interp_spline(x, y, k=3)(x_smooth)
                else:
                    x_smooth, y_smooth = x, y

                line, = ax.plot(x_smooth, y_smooth,
                                color=colors[i],
                                linestyle=linestyles[i],
                                linewidth=1.5)
                ax.scatter(x, y, color=colors[i], edgecolor='black', linewidth=0.3,
                           marker=markers[i], s=20, zorder=3)

                if dist not in sample_lines:
                    sample_lines[dist] = line

            ax.set_title(f"{title}", fontsize=11)
            ax.set_ylim(0, 1)
            ax.set_xlim(min(parcellations), max(parcellations))
            if row == 2:
                ax.set_xticks(parcellations)
                ax.set_xticklabels(parcellations, rotation=45)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel("ID Rate")
            else:
                ax.set_yticklabels([])

        else:
            # Legend panel (bottom-right)
            ax.axis('off')

             # Create legend handles with markers + lines
            legend_handles = [
                  Line2D([0], [0],
                 color=colors[i],
                 marker=markers[i],
                 linestyle=linestyles[i],
                 markersize=6,
                 linewidth=1.5,
                 label=dist)
                 for i, dist in enumerate(distance_labels)
                ]

             # Plot the actual legend
            ax.legend(
                  handles=legend_handles,
                   loc='center',
                  title='Distance Metrics',
                   title_fontsize=10,
                  fontsize=9,
                  frameon=True,
                    borderpad=1.5
                )

    # Shared axis labels
    fig.text(0.5, 0.03, 'Number of Parcellations', ha='center', fontsize=11)
    #fig.text(0.04, 0.5, 'ID Rate', va='center', rotation='vertical', fontsize=11)

    # Panel title
    fig.suptitle(panel_title, fontsize=13, fontweight='bold', y=0.97)

    # Layout tweaks
    plt.tight_layout(rect=[0.06, 0.06, 0.95, 0.95])
    fig.savefig("D:/Research AU/figure_3x3_with_legend.pdf", bbox_inches='tight')
    fig.savefig("D:/Research AU/figure_3x3_with_legend.svg", bbox_inches='tight')
    plt.show()

plot_3x3_with_legend(
    tasks_data_page_1 + tasks_data_page_2[:4],  # 8 task dictionaries
    titles_page_1 + titles_page_2[:4],          # 8 corresponding titles
    parcellations,
    panel_title="Task-wise Performance Across Parcellations"
)

#figure4..
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# All parcellation-level data
data = [
    {'AI': [0.8820, 0.1238, 0.4019, 0.6787, 0.4603, 0.2909, 0.5806, 0.6904],
     'BW': [0.6881, 0.1484, 0.3411, 0.5199, 0.2360, 0.2886, 0.3949, 0.4579],
     'Log-Euclidean': [0.8411, 0.2325, 0.4720, 0.7126, 0.5222, 0.3703, 0.6098, 0.6787],
     'Alpha Z': [0.7850, 0.3423, 0.5631, 0.7126, 0.4264, 0.5012, 0.6437, 0.6262],
     'Alpha Procrust': [0.8423, 0.2336, 0.4755, 0.7114, 0.4860, 0.3914, 0.6040, 0.6636]},
    {'AI': [0.8984, 0.1133, 0.2523, 0.5561, 0.2652, 0.2839, 0.3680, 0.6682],
     'BW': [0.7862, 0.2009, 0.3972, 0.5935, 0.2815, 0.3598, 0.4907, 0.5444],
     'Log-Euclidean': [0.9147, 0.1157, 0.5981, 0.8143, 0.6437, 0.4591, 0.6951, 0.7675],
     'Alpha Z': [0.9264, 0.5666, 0.8072, 0.8890, 0.6939, 0.7547, 0.8283, 0.8213],
     'Alpha Procrust': [0.9229, 0.1893, 0.6016, 0.8201, 0.6425, 0.4521, 0.6951, 0.7792]},
    {'AI': [0.8820, 0.2021, 0.4591, 0.6343, 0.2558, 0.4147, 0.5596, 0.5958],
     'BW': [0.8364, 0.2407, 0.4755, 0.6495, 0.3341, 0.4159, 0.5654, 0.6040],
     'Log-Euclidean': [0.9357, 0.2105, 0.4451, 0.7255, 0.3832, 0.5140, 0.5456, 0.8107],
     'Alpha Z': [0.9591, 0.6694, 0.8867, 0.9498, 0.8341, 0.8668, 0.9147, 0.9030],
     'Alpha Procrust': [0.9591, 0.2582, 0.4708, 0.6904, 0.3750, 0.4311, 0.5584, 0.8177]},
    {'AI': [0.9042, 0.2395, 0.5035, 0.6729, 0.2780, 0.4544, 0.5888, 0.5993],
     'BW': [0.8458, 0.2465, 0.3890, 0.6308, 0.3610, 0.4112, 0.5432, 0.6075],
     'Log-Euclidean': [0.8949, 0.2266, 0.5023, 0.6612, 0.2745, 0.5417, 0.5643, 0.6624],
     'Alpha Z': [0.9813, 0.7196, 0.9159, 0.9579, 0.8879, 0.8995, 0.9439, 0.9369],
     'Alpha Procrust': [0.8925, 0.2780, 0.5140, 0.6811, 0.3808, 0.4708, 0.6121, 0.6484]},
    {'AI': [0.9171, 0.2605, 0.5012, 0.6904, 0.2944, 0.4743, 0.6075, 0.6180],
     'BW': [0.8400, 0.2453, 0.3890, 0.6308, 0.3750, 0.4030, 0.5304, 0.6063],
     'Log-Euclidean': [0.8902, 0.2488, 0.4871, 0.6787, 0.2803, 0.4661, 0.5841, 0.5935],
     'Alpha Z': [0.9883, 0.7383, 0.9322, 0.9661, 0.9077, 0.9124, 0.9521, 0.9533],
     'Alpha Procrust': [0.8376, 0.3026, 0.5397, 0.6998, 0.4112, 0.4988, 0.6355, 0.6659]},
    {'AI': [0.9299, 0.2617, 0.5082, 0.7044, 0.2956, 0.4790, 0.6238, 0.6343],
     'BW': [0.8563, 0.2605, 0.3949, 0.6355, 0.3843, 0.4019, 0.5397, 0.6098],
     'Log-Euclidean': [0.9077, 0.2406, 0.5082, 0.6892, 0.2850, 0.4509, 0.5829, 0.6262],
     'Alpha Z': [0.9918, 0.7605, 0.9486, 0.9650, 0.9241, 0.9241, 0.9638, 0.9661],
     'Alpha Procrust': [0.8563, 0.3213, 0.5666, 0.7150, 0.4357, 0.5035, 0.6577, 0.6951]},
    {'AI': [0.9241, 0.2769, 0.5140, 0.6998, 0.3026, 0.4953, 0.6355, 0.6355],
     'BW': [0.8423, 0.2664, 0.4019, 0.6390, 0.3984, 0.4217, 0.5397, 0.6016],
     'Log-Euclidean': [0.9077, 0.2523, 0.5175, 0.6986, 0.2827, 0.4836, 0.6192, 0.6379],
     'Alpha Z': [0.9942, 0.7664, 0.9509, 0.9731, 0.9381, 0.9322, 0.9708, 0.9685],
     'Alpha Procrust': [0.8598, 0.3563, 0.5818, 0.7255, 0.4544, 0.5327, 0.6624, 0.6986]},
    {'AI': [0.9346, 0.2792, 0.4895, 0.7009, 0.3026, 0.4930, 0.6285, 0.6402],
     'BW': [0.8493, 0.2500, 0.3820, 0.6262, 0.3995, 0.4042, 0.5409, 0.5853],
     'Log-Euclidean': [0.9159, 0.2488, 0.4603, 0.6764, 0.2757, 0.4673, 0.5912, 0.6402],
     'Alpha Z': [0.9965, 0.7745, 0.9533, 0.9755, 0.9404, 0.9381, 0.9731, 0.9731],
     'Alpha Procrust': [0.8598, 0.3657, 0.5923, 0.7430, 0.4673, 0.5432, 0.6822, 0.7114]},
    {'AI': [0.9404, 0.2780, 0.4860, 0.7056, 0.3095, 0.4825, 0.6215, 0.6659],
     'BW': [0.8493, 0.2547, 0.4007, 0.6308, 0.4206, 0.4334, 0.5479, 0.5806],
     'Log-Euclidean': [0.9194, 0.2395, 0.4322, 0.6810, 0.2512, 0.4521, 0.5923, 0.6507],
     'Alpha Z': [0.9977, 0.7979, 0.9603, 0.9778, 0.9439, 0.9416, 0.9685, 0.9778],
     'Alpha Procrust': [0.8738, 0.3797, 0.6145, 0.7558, 0.4953, 0.5654, 0.6928, 0.7255]}
]

# ---- STYLE SETTINGS FOR PUBLICATION ----
mpl.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10
})

# ---- PARAMETERS ----
tasks = ['RT', 'EMN', 'GML', 'LNG', 'MT', 'REL', 'SCL', 'WM']
methods = ['Alpha Z', 'Alpha Procrust', 'AI', 'Log-Euclidean', 'BW']
titles = [
    '100 Parcellations', '200 Parcellations', '300 Parcellations',
    '400 Parcellations', '500 Parcellations', '600 Parcellations',
    '700 Parcellations', '800 Parcellations', '900 Parcellations'
]
colors = ['#e41a1c', '#1f77b4', '#ff00ff', '#FFB6C1', '#90EE90']

# ---- FIGURE SETUP ----
fig, axs = plt.subplots(3, 3, figsize=(11, 8.5), dpi=300)  # A4-ish aspect ratio
axs = axs.flatten()
fig.patch.set_facecolor('lightyellow')

bar_width = 0.13
x = np.arange(len(tasks))  # 8 tasks

# ---- PLOT ----
for idx, ax in enumerate(axs):
    for i, method in enumerate(methods):
        values = data[idx][method]  # You must define 'data' outside this block
        ax.bar(x + i * bar_width, values, width=bar_width, color=colors[i],
               label=method if idx == 0 else "")

    ax.set_title(titles[idx])

    if idx >= 6:  # Bottom row: show task labels
        ax.set_xticks(x + bar_width * 2)
        ax.set_xticklabels(tasks, rotation=45, ha='center')
    else:
        ax.set_xticks(x + bar_width * 2)
        ax.set_xticklabels([])

    if idx % 3 == 0:  # Left column: show Y axis
        ax.set_ylabel("ID Rate")
        ax.set_yticks(np.linspace(0, 1, 5))
        ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(0, 1, 5)])
    else:
        ax.set_yticks([])

        ax.set_ylim(0, 1)
        ax.tick_params(axis='both')

# ---- GLOBAL LEGEND ----
legend = fig.legend(
    methods,
    loc='lower center',
    ncol=len(methods),
    fontsize=10,
    title='Distance Metrics',
    title_fontsize=11,
    frameon=True
)
legend.get_frame().set_facecolor('lightgrey')
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(0.8)

# ---- MAIN TITLE ----
fig.suptitle("Impact of Parcellation Granularity on Identification Rates",
             fontsize=14, fontweight='bold', y=0.96)
fig.text(0.5, 0.06, 'Tasks', ha='center', fontsize=13)

# ---- FINAL LAYOUT AND EXPORT ----
plt.tight_layout(rect=[0.06, 0.06, .95, 0.95])  # Leave space for legend and title

fig.savefig("D:/Research AU/figure52.pdf", bbox_inches='tight')
fig.savefig("D:/Research AU/figure52.svg", bbox_inches='tight')
plt.show()
#final one for figure 4
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# ---- STYLE SETTINGS ----
mpl.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10
})

# ---- PARAMETERS ----
tasks = ['RT', 'EMN', 'GML', 'LNG', 'MT', 'REL', 'SCL', 'WM']
methods = ['Alpha Z', 'Alpha Procrust', 'AI', 'Log-Euclidean', 'BW']
titles = [
    '100 Parcellations', '200 Parcellations', '300 Parcellations',
    '400 Parcellations', '500 Parcellations', '600 Parcellations',
    '700 Parcellations', '800 Parcellations', '900 Parcellations'
]
colors = ['#e41a1c', '#1f77b4', '#ff00ff', '#FFB6C1', '#90EE90']

# ---- FIGURE SETUP ----
fig, axs = plt.subplots(3, 3, figsize=(11, 8.5), dpi=300)
axs = axs.flatten()
fig.patch.set_facecolor('lightyellow')

bar_width = 0.13
x = np.arange(len(tasks))  # 8 tasks

# ---- PLOT ----
for idx, ax in enumerate(axs):
    for i, method in enumerate(methods):
        values = data[idx][method]
        ax.bar(x + i * bar_width, values, width=bar_width, color=colors[i],
               label=method if idx == 0 else "")

    ax.set_title(titles[idx])

    ax.set_xticks(x + bar_width * 2)
    if idx >= 6:
        ax.set_xticklabels(tasks, rotation=30, ha='center')
    else:
        ax.set_xticklabels([])

    if idx % 3 == 0:
        ax.set_ylabel("ID Rate")
        ax.set_yticks(np.linspace(0, 1, 5))
        ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(0, 1, 5)])
    else:
        ax.set_yticks([])

    ax.set_ylim(0, 1)
    ax.tick_params(axis='both')

# ---- MAIN TITLE ----
fig.suptitle("Impact of Parcellation Granularity on Identification Rates",
             fontsize=14, fontweight='bold', y=0.96)

# ---- XLABEL (ABOVE LEGEND!) ----
fig.text(0.5, 0.12, 'Tasks', ha='center', fontsize=13)  # moved up to stay above the legend

# ---- GLOBAL LEGEND BELOW EVERYTHING ----
legend = fig.legend(
    methods,
    loc='lower center',
    ncol=len(methods),
    fontsize=10,
    title='Distance Metrics',
    title_fontsize=11,
    frameon=True,
    bbox_to_anchor=(0.5, 0.01)  # lower placement
)
legend.get_frame().set_facecolor('lightgrey')
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(0.8)

# ---- FINAL LAYOUT ----
plt.tight_layout(rect=[0.06, 0.12, .95, 0.95])  # leave more space at bottom

# ---- EXPORT ----
fig.savefig("D:/Research AU/figure5.pdf", bbox_inches='tight')
fig.savefig("D:/Research AU/figure5.svg", bbox_inches='tight')
plt.show()



#figure5..
# Regulization compare (figure 5)
import matplotlib.pyplot as plt
import numpy as np

# Data for ID rate performance of AI comparison for 100-900 parcellation with tau values until 1
tasks = ['REST', 'EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
parcellations = [100, 200, 300, 400, 500, 600, 700, 800, 900]

# Data for each task (first table)
task_data1 = {
    'REST': [0.8820, 0.8984, 0.8820, 0.3411, 0.1227, 0.0771, 0.0607, 0.0456,.0426],
    'EMOTION': [0.1238, 0.1133, 0.0888, 0.0058, 0.0058, 0.0047, 0.0047, 0.0070, 0.0047],
    'GAMBLING': [0.4019, 0.2523, 0.2079, 0.0105, 0.0070, 0.0058, 0.0047, 0.0035, 0.0070],
    'LANGUAGE': [0.6787, 0.5561, 0.5222, 0.0210, 0.0292, 0.0058, 0.0058, 0.0047,0.0047],
    'MOTOR': [0.4603, 0.2652, 0.1706, 0.0280, 0.0210, 0.0058, 0.0082, 0.0058, 0.0058],
    'RELATIONAL': [0.2909, 0.2839, 0.2395, 0.0164, 0.0082, 0.0070, 0.0058, 0.0047, 0.0047],
    'SOCIAL': [0.5806, 0.3680, 0.3189, 0.0292, 0.0152, 0.0035, 0.0093, 0.0105,0.0105],
    'WM': [0.6904, 0.6682, 0.5958, 0.3621, 0.0257, 0.0152, 0.0199, 0.0035, 0.0058]
}

# Data for ID rate performance of AI comparison for 300-900 parcellation with tau values until 50
# Data for each task (second table)
task_data2 = {
    'REST': [0.8820, 0.9042, 0.9171, 0.9299, 0.9241, 0.9346, 0.9404],
    'EMOTION': [0.2021, 0.2395, 0.2605, 0.2617, 0.2769, 0.2792, 0.2780],
    'GAMBLING': [0.4591, 0.5035, 0.5012, 0.5082, 0.5140, 0.4895, 0.4860],
    'LANGUAGE': [0.6343, 0.6729, 0.6904, 0.7044, 0.6998, 0.7009, 0.7056],
    'MOTOR': [0.2558, 0.2780, 0.2944, 0.2956, 0.3026, 0.3026,0.3096],
    'RELATIONAL': [0.4147, 0.4544, 0.4743, 0.4790, 0.4953, 0.4930, 0.4825],
    'SOCIAL': [0.5596, 0.5888, 0.6075, 0.6238, 0.6355, 0.6285, 0.6215],
    'WM': [0.5958, 0.5993, 0.6180, 0.6343, 0.6355, 0.6402, 0.6659]
}

# Corresponding tau values for second table tasks
tau_values = {
    'REST': [4, 10, 10, 12, 28, 20, 22],
    'EMOTION': [30, 40, 46, 48, 48, 48, 48],
    'GAMBLING': [30, 42, 42, 42, 42, 42, 42],
    'LANGUAGE': [22, 30, 50, 42, 34, 40, 42],
    'MOTOR': [26, 44, 42, 40, 42, 42, 42],
    'RELATIONAL': [26, 28, 36, 36, 40, 40, 40],
    'SOCIAL': [40, 44, 40, 42, 42, 42, 40],
    'WM': [0, 26, 36, 36, 32, 36, 46]
}

# final 1
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Style and fonts
mpl.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10
})

colors = plt.cm.tab10.colors

# Plot side-by-side layout
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.5))  # Nature-like size

# -------- Low Regularization --------
for i, (task, data) in enumerate(task_data1.items()):
    ax1.plot(parcellations[:len(data)], data, label=task, marker='o', linewidth=1.5, color=colors[i % 10])
ax1.set_title('Low Regularization (τ ≤ 1)', fontweight='bold')
ax1.set_xlabel('Parcellation')
ax1.set_ylabel('ID Rate')
ax1.set_ylim(0, 1)

# -------- High Regularization --------
for i, (task, data) in enumerate(task_data2.items()):
    ax2.plot(parcellations[2:len(data)+2], data, label=task, linestyle='--', marker='x', linewidth=1.5, color=colors[i % 10])
    for j, tau in enumerate(tau_values[task]):
        ax2.annotate(f'τ={tau}', (parcellations[j+2], data[j]),
                     textcoords="offset points", xytext=(0, 7), ha='center', fontsize=7, color=colors[i % 10])
ax2.set_title('High Regularization (τ > 1)', fontweight='bold')
ax2.set_xlabel('Parcellation')
ax2.set_ylabel('ID Rate')
ax2.set_ylim(0, 1)

# Legend outside
fig.legend(*ax1.get_legend_handles_labels(), loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
plt.tight_layout()

# SAVE AS VECTOR FORMAT
fig.savefig("D:/Research AU/figure2_ai_sensitivity.pdf", bbox_inches='tight')
fig.savefig("D:/Research AU/figure2_ai_sensitivity.svg", bbox_inches='tight')


# final 2 for regularization
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Style for clean output
mpl.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9
})

colors = plt.cm.tab10.colors

# Create stacked figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.2), dpi=300)

# --- Panel (a): Low Regularization ---
for i, (task, data) in enumerate(task_data1.items()):
    ax1.plot(parcellations[:len(data)], data,
             label=task, marker='o', linewidth=1.5, color=colors[i % 10])

ax1.set_title(' Low Regularization (τ ≤ 1)', fontweight='bold', loc='center')
ax1.set_ylabel('ID Rate')
ax1.set_xlim(50, 950)
ax1.set_ylim(0, 1)
ax1.set_xticks(np.arange(100, 901, 100))
ax1.legend(loc='upper right', ncol=2, fontsize=9, frameon=False)

# --- Panel (b): High Regularization ---
tasks_below = ['EMOTION', 'RELATIONAL', 'SOCIAL']
tasks_above = ['REST', 'MOTOR', 'LANGUAGE', 'WM', 'GAMBLING']

for i, (task, data) in enumerate(task_data2.items()):
    ax2.plot(parcellations[2:len(data)+2], data,
             label=task, linestyle='--', marker='x', linewidth=1, color=colors[i % 10])

    for j, tau in enumerate(tau_values[task]):
        x = parcellations[j + 2]
        y = data[j]

        # Decide label position by task name
        if task.upper() in tasks_below:
            y_offset = -5
        else:
            y_offset = 4

        ax2.annotate(f'τ={tau}',
                     (x, y),
                     textcoords="offset points",
                     xytext=(0, y_offset),
                     ha='center',
                     fontsize=7,
                     fontweight='bold',
                     color=colors[i % 10],
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.60))

ax2.set_title(' High Regularization (τ > 1)', fontweight='bold', loc='center')
ax2.set_xlabel('Parcellation')
ax2.set_ylabel('ID Rate')
ax2.set_xlim(250, 950)
ax2.set_ylim(0, 1.15)
ax2.set_xticks(np.arange(300, 901, 100))

# Layout tuning
plt.tight_layout(h_pad=2.5)

# Save high-res outputs
fig.savefig("D:/Research AU/figure6.svg", bbox_inches='tight')
fig.savefig("D:/Research AU/figure6.pdf", bbox_inches='tight')
plt.show()

## final 3 for regularization
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Style for clean output
mpl.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9
})

colors = plt.cm.tab10.colors

# Create side-by-side figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

# --- Panel (a): Low Regularization ---
for i, (task, data) in enumerate(task_data1.items()):
    ax1.plot(parcellations[:len(data)], data,
             label=task, marker='o', linewidth=1.5, color=colors[i % 10])

ax1.set_title('Low Regularization (τ ≤ 1)', fontweight='bold')
ax1.set_xlabel('Parcellation')
ax1.set_xlim(50, 950)
ax1.set_ylim(0, 1)
ax1.set_xticks(np.arange(100, 901, 100))
ax1.legend(loc='upper right', ncol=2, fontsize=9, frameon=False)

# --- Panel (b): High Regularization ---
tasks_below = ['EMOTION', 'RELATIONAL', 'SOCIAL']
inline_tasks = ['MOTOR', 'GAMBLING']

for i, (task, data) in enumerate(task_data2.items()):
    ax2.plot(parcellations[2:len(data)+2], data,
             label=task, linestyle='--', marker='x', linewidth=1, color=colors[i % 10])

    for j, tau in enumerate(tau_values[task]):
        x = parcellations[j + 2]
        y = data[j]

        # Decide label position
        task_upper = task.upper()
        if task_upper in tasks_below:
            y_offset = -5
        elif task_upper in inline_tasks:
            y_offset = 2
        else:
            y_offset = 4

        ax2.annotate(f'τ={tau}',
                     (x, y),
                     textcoords="offset points",
                     xytext=(0, y_offset),
                     ha='center',
                     fontsize=7,
                     fontweight='bold',
                     color=colors[i % 10],
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.60))

ax2.set_title('High Regularization (τ > 1)', fontweight='bold')
ax2.set_xlabel('Parcellation')
ax2.set_xlim(250, 950)
ax2.set_ylim(0, 1.15)
ax2.set_xticks(np.arange(300, 901, 100))

# Shared Y-axis label in the middle
fig.text(0.04, 0.5, 'ID Rate', va='center', rotation='vertical', fontsize=11)

# Layout tuning
plt.tight_layout(rect=[0.07, 0.05, 1, 0.95], w_pad=2.5)

# Save outputs
fig.savefig("D:/Research AU/figure6_horizontal.svg", bbox_inches='tight')
fig.savefig("D:/Research AU/figure6_horizontal.pdf", bbox_inches='tight')
plt.show()

### figure 8....
#bar plot 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# ---- STYLE SETTINGS FOR PUBLICATION ----
mpl.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10
})

# ---- DATA SETUP ----
tasks = ['REST', 'EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
x = np.arange(len(tasks))
bar_width = 0.25

methods = ['Alpha Z', 'Pearson', 'Euclidean']
colors = ['red', 'green', 'blue']

data_100 = [
    [0.7850, 0.3423, 0.5631, 0.7126, 0.4264, 0.5011, 0.6437, 0.6262],
    [0.4533, 0.0958, 0.2629, 0.3773, 0.1565, 0.2114, 0.3236, 0.3353],
    [0.3458, 0.0607, 0.1636, 0.2418, 0.0876, 0.1507, 0.2208, 0.2208]
]
data_200 = [
    [0.9264, 0.5666, 0.8072, 0.8890, 0.6939, 0.7547, 0.8283, 0.8213],
    [0.6075, 0.1530, 0.3797, 0.5456, 0.2336, 0.3294, 0.4825, 0.4836],
    [0.4498, 0.0888, 0.2255, 0.3318, 0.1343, 0.2138, 0.3154, 0.3189]
]
data_300 = [
    [0.9591, 0.6640, 0.8867, 0.9498, 0.8341, 0.8668, 0.9147, 0.9030],
    [0.6671, 0.1963, 0.4790, 0.6308, 0.3189, 0.4019, 0.5864, 0.5853],
    [0.5280, 0.1051, 0.2921, 0.3960, 0.1706, 0.2453, 0.3773, 0.3808]
]
all_data = [data_100, data_200, data_300]
titles = ['100 Parcellations', '200 Parcellations', '300 Parcellations']

# ---- PLOT SETUP ----
fig, axs = plt.subplots(1, 3, figsize=(16, 8), dpi=300, sharey=True)
fig.patch.set_facecolor('lightyellow')

for idx, ax in enumerate(axs):
    for i, method_data in enumerate(all_data[idx]):
        offset = (i - 1) * bar_width
        ax.bar(x + offset, method_data, width=bar_width, color=colors[i], label=methods[i])

    ax.set_title(titles[idx])  # uses mpl.rcParams['axes.titlesize']
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=25, ha='center')
    ax.set_ylim(0, 1)
    ax.set_facecolor('#f9f9f9')
    ax.set_xlabel('Tasks')  # uses axes.labelsize
    if idx == 0:
        ax.set_ylabel('ID Rate')
        ax.legend(loc='upper left', frameon=True, title="Metrics")
    else:
        ax.legend().remove()

# ---- FINAL TITLE ----
fig.suptitle("Comparison of ID Rates Over Traditional Metrics", fontsize=16, fontweight='bold', y=0.96)

# ---- LAYOUT AND EXPORT ----
plt.tight_layout(rect=[0.0, 0.0, .95, .95])
fig.savefig("D:/Research AU/figure7.pdf", bbox_inches='tight')
fig.savefig("D:/Research AU/figure7.svg", bbox_inches='tight')
plt.show()

# optional for figure 8
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Data setup
tasks = ['REST', 'EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
x = np.arange(len(tasks))
bar_width = 0.2  # reduced bar width

# All parcellation data (Pearson, Euclidean, Alpha Z)
data_100 = [
    [0.4533, 0.0958, 0.2629, 0.3773, 0.1565, 0.2114, 0.3236, 0.3353],  # Pearson
    [0.3458, 0.0607, 0.1636, 0.2418, 0.0876, 0.1507, 0.2208, 0.2208],  # Euclidean
    [0.7850, 0.3423, 0.5631, 0.7126, 0.4264, 0.5011, 0.6437, 0.6262]   # Alpha Z
]
data_200 = [
    [0.6075, 0.1530, 0.3797, 0.5456, 0.2336, 0.3294, 0.4825, 0.4836],
    [0.4498, 0.0888, 0.2255, 0.3318, 0.1343, 0.2138, 0.3154, 0.3189],
    [0.9264, 0.5666, 0.8072, 0.8890, 0.6939, 0.7547, 0.8283, 0.8213]
]
data_300 = [
    [0.6671, 0.1963, 0.4790, 0.6308, 0.3189, 0.4019, 0.5864, 0.5853],
    [0.5280, 0.1051, 0.2921, 0.3960, 0.1706, 0.2453, 0.3773, 0.3808],
    [0.9591, 0.6640, 0.8867, 0.9498, 0.8341, 0.8668, 0.9147, 0.9030]
]
all_data = [data_100, data_200, data_300]
titles = ['100 Parcellations', '200 Parcellations', '300 Parcellations']
methods = ['Pearson', 'Euclidean', 'Alpha Z']
cmaps = ['Blues', 'Greens', 'Reds']

# Plot setup
fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
fig.patch.set_facecolor('lightyellow')

for plot_idx, ax in enumerate(axs):
    data = all_data[plot_idx]
    for i, method_data in enumerate(data):
        for j, height in enumerate(method_data):
            x_pos = j + (i - 1) * bar_width  # Offset each method
            gradient = np.linspace(0.2, 1, 256).reshape(-1, 1)
            extent = [x_pos - bar_width / 2, x_pos + bar_width / 2, 0, height]

            # Draw gradient bar
            ax.imshow(gradient, aspect='auto', cmap=cmaps[i], extent=extent, alpha=1, zorder=2)

            # Draw border
            ax.plot(
                [x_pos - bar_width / 2, x_pos + bar_width / 2, x_pos + bar_width / 2, x_pos - bar_width / 2, x_pos - bar_width / 2],
                [0, 0, height, height, 0], color='black', linewidth=0.8
            )

    ax.set_title(titles[plot_idx], fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=25, ha='center')
    ax.set_xlabel("Task")
    ax.set_facecolor('#f9f9f9')
    if plot_idx == 0:
        ax.set_ylabel("ID Rate")
    ax.set_ylim(0, 1)

# Add a global legend manually
legend_patches = [Patch(color=plt.cm.get_cmap(cmap)(0.8), label=method) for cmap, method in zip(cmaps, methods)]
axs[0].legend(handles=legend_patches, title="Metrics", loc='upper left', frameon=True)

fig.suptitle("Comparison of ID Rates over traditional metrices", fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save to PNG
fig.savefig("D:/Research AU/figure7.pdf", bbox_inches='tight')
fig.savefig("D:/Research AU/figure7.svg", bbox_inches='tight')
plt.show()



#figure 8 3d
### Final code for 3D figure.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.patches import FancyBboxPatch

# Function to generate file paths for subjects' connectivity matrices
def generate_file_paths(base_path, scan_type, num_subjects=5):
    file_paths = []
    subject_ids = sorted(os.listdir(base_path))[:num_subjects]
    for subject_id in subject_ids:
        if os.path.isdir(os.path.join(base_path, subject_id)):
            file_path = os.path.join(base_path, subject_id, f'{subject_id}_rfMRI_REST1_{scan_type}_400')
            if os.path.exists(file_path):
                file_paths.append(file_path)
    return file_paths

# Function to load connectivity matrix from a file
def load_connectivity_matrix(file_path):
    try:
        matrix = np.loadtxt(file_path, delimiter=' ')
        if matrix.shape[0] != matrix.shape[1] or not np.allclose(matrix, matrix.T):
            print(f"Matrix at {file_path} is not square or symmetric. Skipping.")
            return None
        return matrix
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to compute the geodesic (AI) distance
def compute_geodesic_distance(A, B):
    from scipy.linalg import sqrtm, logm
    C = np.dot(np.linalg.inv(sqrtm(A)), B)
    C = np.dot(C, np.linalg.inv(sqrtm(A)))
    logC = logm(C)
    distance = np.linalg.norm(logC, 'fro')
    return distance

# Function to compute the Alpha-Z BW divergence distance
def compute_alpha_z_BW_distance(A, B, alpha=0.99, z=1):
    from scipy.linalg import fractional_matrix_power
    part1 = fractional_matrix_power(B, (1-alpha)/(2*z))
    part2 = fractional_matrix_power(A, alpha/z)
    part3 = fractional_matrix_power(B, (1-alpha)/(2*z))
    Q_az = fractional_matrix_power(np.dot(np.dot(part1, part2), part3), z)
    divergence = np.trace((1-alpha) * A + alpha * B) - np.trace(Q_az)
    return np.real(divergence)

# Function to compute the distance matrix for a set of connectivity matrices
def compute_distance_matrix(matrices, distance_func):
    num_matrices = len(matrices)
    distance_matrix = np.zeros((num_matrices, num_matrices))
    for i in range(num_matrices):
        for j in range(i+1, num_matrices):
            distance = distance_func(matrices[i], matrices[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Symmetry
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix

# Function to compute match percentages based on normalized distances
def compute_match_percentages(distance_matrix):
    max_distance = np.max(distance_matrix)
    if max_distance == 0:
        match_percentages = np.ones_like(distance_matrix) * 100
    else:
        match_percentages = 100 * (1 - distance_matrix / max_distance)
    return np.round(match_percentages, 2)

# Function to convert subject IDs to Subject 1, Subject 2 format
def convert_to_generic_subject_ids(subject_ids):
    subject_base_names = sorted(set([sid.split('_')[0] for sid in subject_ids]))
    base_to_subj = {orig: f'Subject {i+1}' for i, orig in enumerate(subject_base_names)}
    new_subject_ids = [f"{base_to_subj[sid.split('_')[0]]}_{sid.split('_')[1]}" for sid in subject_ids]
    return new_subject_ids

# Function to visualize the 3D embeddings with match percentages and highlighted box
def visualize_combined_3d_with_same_subject_percentages(matrices, subject_ids, distance_matrix, ax, title):
    mds = MDS(n_components=3, dissimilarity="precomputed", random_state=42)
    try:
        embeddings = mds.fit_transform(distance_matrix)
    except ValueError as e:
        print(f"Error in MDS fit_transform: {e}")
        return

    match_percentages = compute_match_percentages(distance_matrix)
    box_lines = []

    for i, embedding in enumerate(embeddings):
        color = 'red' if 'LR' in subject_ids[i] else 'blue'
        ax.scatter(*embedding, color=color, s=50)
        ax.text(*embedding, f'{subject_ids[i]}', color='black', fontsize=8, weight='bold')

    for i, subject_id_lr in enumerate(subject_ids):
        if '_LR' in subject_id_lr:
            base_id = subject_id_lr.replace('_LR', '')
            for j, subject_id_rl in enumerate(subject_ids):
                if subject_id_rl == base_id + '_RL':
                    match_percentage = match_percentages[i, j]
                    subject_num = base_id.split()[1] if ' ' in base_id else base_id
                    box_lines.append(f"{base_id}: {match_percentage}%")

    # Draw single annotation box with all subjects
    ax.text2D(0.02, 0.98, '\n'.join(box_lines), transform=ax.transAxes,
              bbox=dict(boxstyle="round,pad=0.8", edgecolor="black", facecolor="#d2f8d2"),
              fontsize=12, verticalalignment='top', color='black')

    ax.set_facecolor('lightyellow')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Component 1', fontsize=10)
    ax.set_ylabel('Component 2', fontsize=10)
    ax.set_zlabel('Component 3', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='z', which='major', labelsize=8)
#f5f5ff
# Example usage
base_path = 'D:/Research AU/connectomes_400/'
lr_paths = generate_file_paths(base_path, 'LR', 5)
rl_paths = generate_file_paths(base_path, 'RL', 5)
subject_ids = [os.path.basename(path).split('_')[0] + '_LR' for path in lr_paths] + \
               [os.path.basename(path).split('_')[0] + '_RL' for path in rl_paths]
subject_ids = convert_to_generic_subject_ids(subject_ids)
matrices = [matrix for path in lr_paths + rl_paths if (matrix := load_connectivity_matrix(path)) is not None]

if matrices:
    distance_matrix_ai = compute_distance_matrix(matrices, compute_geodesic_distance)
    distance_matrix_alpha_z = compute_distance_matrix(matrices, compute_alpha_z_BW_distance)

    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    visualize_combined_3d_with_same_subject_percentages(matrices, subject_ids, distance_matrix_ai, ax1, "AI Distance")

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    visualize_combined_3d_with_same_subject_percentages(matrices, subject_ids, distance_matrix_alpha_z, ax2, "Alpha Z Divergence")

    plt.tight_layout()
    plt.subplots_adjust(right=0.94, left=0.05, bottom=0.08, top=0.92, wspace=0.05)

    fig.savefig("D:/Research AU/figure9.pdf")
    fig.savefig("D:/Research AU/figure9.svg")
    plt.show()
else:
    print("No valid matrices were loaded. Please check file paths and matrix validity.")


#
#Still not upload...
##Null model....
import matplotlib.pyplot as plt
import numpy as np
from math import pi
import matplotlib as mpl

# ---- STYLE SETTINGS FOR PUBLICATION ----
mpl.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10
})

# ---- DATA PARSING FUNCTION ----
def parse_dataset(file_path, tasks_count):
    data = {}
    with open(file_path, 'r') as file:
        current_scale = None
        for line in file:
            line = line.strip()
            if line.startswith("#"):
                current_scale = line[1:].strip()
                data[current_scale] = {"Original": [], "Null": []}
            elif "Original ID Rate" in line:
                original_rate = float(line.split(":")[1].strip())
                data[current_scale]["Original"].append(original_rate)
            elif "Null Model ID Rate" in line:
                null_rate = float(line.split(":")[1].strip())
                data[current_scale]["Null"].append(null_rate)

    for scale in data:
        original_len = len(data[scale]["Original"])
        null_len = len(data[scale]["Null"])
        if original_len < tasks_count:
            data[scale]["Original"].extend([np.nan] * (tasks_count - original_len))
        if null_len < tasks_count:
            data[scale]["Null"].extend([np.nan] * (tasks_count - null_len))
    
    return data

# ---- PARAMETERS ----
file_path = 'D:/Research AU/Multi scale analysis of ID rate/ID_rate_null.txt'
tasks = ['Rest', 'Emotion', 'Gambling', 'Language', 'Motor', 'Relational', 'Social', 'Wm']
parsed_data = parse_dataset(file_path, len(tasks))
scales = list(parsed_data.keys())

# ---- RADAR PLOT SETUP ----
num_rows, num_cols = 2, 5
fig, axes = plt.subplots(num_rows, num_cols, subplot_kw=dict(polar=True), figsize=(15, 7.2))
axes = axes.flatten()

angles = [n / float(len(tasks)) * 2 * pi for n in range(len(tasks))]
angles += angles[:1]
fixed_ticks = np.arange(0, 1.1, 0.2)

# ---- PLOTTING ----
for i, scale in enumerate(scales):
    original = parsed_data[scale]["Original"] + parsed_data[scale]["Original"][:1]
    null = parsed_data[scale]["Null"] + parsed_data[scale]["Null"][:1]

    ax = axes[i]
    ax.plot(angles, original, label='Original', linewidth=1.5, color='blue')
    ax.fill(angles, original, color='purple', alpha=0.15)
    ax.plot(angles, null, label='Null', linewidth=2.5, color='red')
    ax.fill(angles, null, color='red', alpha=0.3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tasks)
    ax.set_yticks(fixed_ticks)         # Grid circles
    ax.set_ylim(0, 1)
    ax.set_rlabel_position(90)         # Label radial scale on top only
    ax.set_title(scale)


# ---- CLEAN LAST AXIS (10th subplot for legend) ----
axes[-1].axis('off')  # Turn off the last (10th) plot
axes[-1].legend(
    handles=[
        plt.Line2D([0], [0], color='blue', lw=6, label='Original'),
        plt.Line2D([0], [0], color='red', lw=6, label='Null')
    ],
    loc='center',
    frameon=False,
    ncol=1
)
# Add legend beside the last plot (900 Parcellations), which is the 9th plot (index 8)
# ---- FINAL ADJUSTMENTS ----
plt.subplots_adjust(hspace=0.2, wspace=0.6, bottom=0.15)
fig.savefig("D:/Research AU/figure10.pdf", bbox_inches='tight')
fig.savefig("D:/Research AU/figure10.svg", bbox_inches='tight')
plt.show()
