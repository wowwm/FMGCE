import re
import matplotlib.pyplot as plt


def draw_loss_score(log_name):
    log_path = "log/" + log_name
    with open(log_path, 'r') as f:
        log_lines = f.readlines()

    # 定义存储train loss和valid_score的列表
    train_losses = []
    valid_scores = []

    # 提取train loss和valid_score的数值
    for line in log_lines:
        train_loss_match = re.search(r'train loss: (.*)]', line)
        valid_score_match = re.search(r'valid_score: (.*)]', line)
        if train_loss_match:
            train_losses.append(float(train_loss_match.group(1)))
        if valid_score_match:
            valid_scores.append(float(valid_score_match.group(1)))

    # 绘制train loss和valid_score的变化曲线
    epochs = range(len(train_losses))

    # 创建一个垂直排列的subplot布局，分为两行一列
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # 绘制train loss的变化曲线
    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Loss')
    ax1.set_title('Train Loss')
    ax1.legend()
    ax1.set_ylim(0, max(train_losses))
    # 标注train loss的最大值和最小值
    max_loss_index = train_losses.index(max(train_losses))
    min_loss_index = train_losses.index(min(train_losses))
    max_loss = max(train_losses)
    min_loss = min(train_losses)

    ax1.annotate(f'Max: {max_loss}', xy=(max_loss_index, max_loss), xytext=(max_loss_index, max_loss + 1))
    ax1.annotate(f'Min: {min_loss}', xy=(min_loss_index, min_loss), xytext=(min_loss_index, min_loss - 2))

    ax2.plot(epochs, valid_scores, label='Valid Score')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Valid Score')
    ax2.set_title('Valid Score')
    ax2.legend()

    plt.savefig('plot_loss_score_' + log_name + '.png')
    plt.show()


def draw_recall_ndcg(log_name):
    log_path = "log/" + log_name
    with open(log_path, 'r') as f:
        log_lines = f.readlines()

    # 定义存储指标值的列表
    recall_at_10 = []
    recall_at_20 = []
    ndcg_at_10 = []
    ndcg_at_20 = []

    # 提取指标值
    flag = 0
    for line in log_lines:
        recall_10_match = re.search(r'recall@10: (.*?) ', line)
        recall_20_match = re.search(r'recall@20: (.*?) ', line)
        ndcg_10_match = re.search(r'ndcg@10: (.*?) ', line)
        ndcg_20_match = re.search(r'ndcg@20: (.*?) ', line)

        if recall_10_match:
            flag += 1
            if flag % 2 == 0:
                recall_at_10.append(float(recall_10_match.group(1)))
                recall_at_20.append(float(recall_20_match.group(1)))
                ndcg_at_10.append(float(ndcg_10_match.group(1)))
                ndcg_at_20.append(float(ndcg_20_match.group(1)))

    # 绘制曲线
    epochs = range(len(recall_at_10[:-4]))

    plt.plot(epochs, recall_at_10[:-4], label='Recall@10')
    plt.plot(epochs, recall_at_20[:-4], label='Recall@20')
    plt.plot(epochs, ndcg_at_10[:-4], label='NDCG@10')
    plt.plot(epochs, ndcg_at_20[:-4], label='NDCG@20')

    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Metrics')
    plt.legend()
    plt.savefig('plot_recall_ndcg_' + log_name + '.png')
    plt.show()


if __name__ == '__main__':
    log_name = "fdn-1-MGCN-baby-Dec-24-2023-17-32-48.log"
    draw_loss_score(log_name)
    draw_recall_ndcg(log_name)
