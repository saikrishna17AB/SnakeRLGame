import matplotlib.pyplot as plt

plt.ion()  # interactive mode

def plot(scores, mean_scores, demo=False):
    plt.clf()
    plt.title('Training...' if not demo else 'Training Graph (Static)')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.ylim(ymin=0)
    plt.legend()
    
    if len(scores) > 0:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    if len(mean_scores) > 0:
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    plt.show(block=False)
    plt.pause(0.01)  # small pause to refresh the plot
