import matplotlib.pyplot as plt
import seaborn as sns

num_postive = 7
num_negative = 55
true_positive = 7   # 替换成实际的真正例数量
false_positive = 6  # 替换成实际的假正例数量
false_negative = num_postive - true_positive
true_negative = num_negative - false_positive


def conf_matrix_visualization(tp, fn, fp, tn):
    # 计算混淆矩阵
    conf_matrix = [[tp, fn], [fp, tn]]

    # 设置全局字体为Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"
    sns.set(font_scale=1.8)  # 调整字体大小
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted crack', 'Predicted normal'],
                     yticklabels=['Actual crack', 'Actual normal'], annot_kws={"size": 30}, cbar=False)
    # 设置标签
    ax.set_xlabel('Predicted', fontsize=20, labelpad=10)  # labelpad参数调整与x轴标签的距离
    ax.set_ylabel('Actual', fontsize=20)
    # 交换 x 和 y 轴的标签
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # 移除 x 轴的 tick marks
    ax.tick_params(axis='x', length=0)
    # 使用 plt.text 在底部添加标题，并调整其位置
    plt.text(0.5, -0.08, 'Confusion Matrix', ha='center', va='center', transform=ax.transAxes, fontsize=20)  # -0.05 调整标题与图之间的距离
    plt.show()


    # 计算其他分类指标
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    true_positive_rate = recall
    false_positive_rate = fp / (fp + tn)

    # 打印其他指标
    print("\nAccuracy:{:.2f}".format(accuracy))
    print("Precision:{:.2f}".format(precision))
    print("Recall:{:.2f}".format(recall))
    print("F1 Score:{:.2f}".format(f1))
    print("True positive rate:{:.2%}".format(true_positive_rate))
    print("False positive rate:{:.2%}".format(false_positive_rate))


if __name__ == "__main__":
    conf_matrix_visualization(true_positive, false_negative, false_positive, true_negative)
