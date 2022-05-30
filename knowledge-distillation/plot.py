import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D

NUM_EPOCHS_STOP = 5
NUM_STUDENTS = 2

# Teacher
def plot_experiment(experiment):
    fig, ax = plt.subplots(2, 1, sharex=True)
    
    if experiment == 'MNIST':
        teacher_metrics = torch.load(f'./results/{experiment}/teacher_metrics.pth', map_location='cpu')
        ax[0].plot(teacher_metrics['test_accuracy'], color = 'tab:blue')
        ax[1].plot(teacher_metrics['test_loss'][0:teacher_metrics['stop_epoch']], color = 'tab:blue')

        teacher_metrics = torch.load(f'./results/{experiment}/teacher_extra_metrics.pth', map_location='cpu')
        ax[0].plot(teacher_metrics['test_accuracy'], color = 'tab:gray')
        ax[1].plot(teacher_metrics['test_loss'][0:teacher_metrics['stop_epoch']], color = 'tab:gray')
    elif experiment == 'CIFAR10':
        teacher_metrics = torch.load(f'./pretrained/resnet44-014dd654.th', map_location='cpu')
        ax[0].axhline(teacher_metrics['best_prec1']/100, linestyle = 'dashed', color = 'tab:blue')
        ax[0].text(x = 35,
                   y = teacher_metrics['best_prec1']/100,
                   s = f"{teacher_metrics['best_prec1']/100:.4f}",
                   color = 'tab:blue')

    # Taught students
    for i in range(NUM_STUDENTS):
        student_metrics = torch.load(f'./results/{experiment}/student_{i}_metrics.pth', map_location='cpu')
        ax[0].plot(student_metrics['test_accuracy'], color = 'tab:red')
        ax[0].text(x = student_metrics['stop_epoch'] - 5,
                   y = student_metrics['test_accuracy'][student_metrics['stop_epoch'] - NUM_EPOCHS_STOP],
                   s = f"{student_metrics['test_accuracy'][student_metrics['stop_epoch'] - NUM_EPOCHS_STOP].item():.4f}",
                   color = 'tab:red')
        ax[1].plot(student_metrics['test_loss'], color = 'tab:red')
        ax[1].text(x = student_metrics['stop_epoch'] - NUM_EPOCHS_STOP,
                   y = student_metrics['test_loss'][student_metrics['stop_epoch'] - NUM_EPOCHS_STOP] + 10,
                   s = f"{student_metrics['test_loss'][student_metrics['stop_epoch'] - NUM_EPOCHS_STOP].item():.4f}",
                   color = 'tab:red')

    # Self-learners
    for i in range(NUM_STUDENTS):
        selflearner_metrics = torch.load(f'./results/{experiment}/slearner_{i}_metrics.pth', map_location='cpu')
        ax[0].plot(selflearner_metrics['test_accuracy'], color = 'tab:green')
        ax[0].text(x = selflearner_metrics['stop_epoch'] - NUM_EPOCHS_STOP,
                   y = selflearner_metrics['test_accuracy'][selflearner_metrics['stop_epoch'] - NUM_EPOCHS_STOP],
                   s = f"{selflearner_metrics['test_accuracy'][selflearner_metrics['stop_epoch'] - NUM_EPOCHS_STOP].item():.4f}",
                   color = 'tab:green')
        ax[1].plot(selflearner_metrics['test_loss'][0:selflearner_metrics['stop_epoch'] - 1], color = 'tab:green')
        ax[1].text(x = selflearner_metrics['stop_epoch'] - NUM_EPOCHS_STOP,
                   y = selflearner_metrics['test_loss'][selflearner_metrics['stop_epoch'] - NUM_EPOCHS_STOP] + 10,
                   s = f"{selflearner_metrics['test_loss'][selflearner_metrics['stop_epoch'] - NUM_EPOCHS_STOP].item():.4f}",
                   color = 'tab:green')

    legend_elements = [
        Line2D([0], [0], color='tab:blue', label='Teacher'),
        Line2D([0], [0], color='tab:red', label='KD students'),
        Line2D([0], [0], color='tab:green', label='Self-learners'),
        # Line2D([0], [0], color='tab:gray', label="Self-learner of Teacher's size")
        ]
        
    ax[0].legend(handles = legend_elements, loc='lower right')

    ax[0].set_title('Test accuracy')
    ax[1].set_title('Test loss')
    
    fig.suptitle(f'Knowledge distllation result on {experiment}', fontsize = 16)

    return fig, ax

# fig1, ax1 = plot_experiment('MNIST')
fig1, ax1 = plot_experiment('CIFAR10')
plt.show()
