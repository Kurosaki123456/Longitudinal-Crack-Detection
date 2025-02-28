import cv2
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from Visualization_of_detection_result import conf_matrix_visualization
import matplotlib.pyplot as plt


class Longitudinal_Crack_Detector:
    def __init__(self):
        # longitudinal crack detection parameters
        self.num_rows = 3
        self.horizontal_thre = 5
        self.vertical_thre = 50
        self.region_height_thre = 5
        self.time_lag_thre = 30     # 相邻行之间的时滞阈值
        self.fp_index = []  # 存储误报样本索引
        self.num_weight_selection = 10  # 训练机器学习模型的权重选择值(数量)，例如=10的时候代表，权重从1变化到10
        self.models = [[] for _ in range(self.num_weight_selection)]
        self.crack_refining_indices = {}
        self.crack_info = [{} for _ in range(self.num_rows)]
        self.crack_num = [0 for _ in range(self.num_rows)]
        self.samples_feature = []   # 图像样本特征集合
        self.combined_features = np.array([])   # 存储每个图像三排的Hog特征及sift特征
        self.threshold_area = 10  # 区域面积小于10忽略
        self.bcorr_expa = []  # 检查是否有包围框距红蓝区域中心太远的情况，即包围效果不好，则需要使用开操作(先腐蚀后膨胀)去除无关噪音
        self.bis_crack_detected = True

    def load_image(self, image_path):
        # 打开文件并读取字节
        with open(image_path, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), np.uint8)

        # 使用 cv2.imdecode 解码图像
        self.image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        self.img_height, self.img_width, self.img_channels = self.image.shape

        return self.image

        # # 使用cv2.imread读取图像
        # self.image = cv2.imread(image_path)
        # self.img_height, self.img_width, self.img_channels = self.image.shape

    def read_images_from_folder(self, folder_path, label):
        image_list = []
        num = 0
        for filename in os.listdir(folder_path):
            if label == 1:
                Sample_index = filename.split('_')[0]
                Sample_column = filename.split('_')[-1]
                image_list.append({'Sample_index': int(Sample_index)})
                image_list[num].update({'Sample_column': int(Sample_column)})
            else:
                image_list.append({'Sample_index': int(filename)})
                image_list[num].update({'Sample_column': -1})
            img_file = os.path.join(folder_path, filename)
            for image_name in os.listdir(img_file):
                img_path = os.path.join(img_file, image_name)
                image_list[num].update({image_name[0:3]: [img_path, image_name.split(" ")[-1][:-4]]})
            image_list[num].update({'label': label})
            num += 1
        return image_list

    def show_img(self, img_name, img):
        pass
        # cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)  # 使用 WINDOW_NORMAL 参数可以调整窗口大小
        # cv2.imshow(img_name, img)
        # # 调整窗口大小
        # cv2.resizeWindow(img_name, img.shape[1] * 3, img.shape[0] * 3)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # show the searched contours on the image
    def show_contour(self, img, contours, idx=-1):
        # make a copy of the original image
        img_copy = img.copy()
        # 绘制轮廓，可以指定颜色（BGR），线宽，-1表示填充轮廓
        # 我们将轮廓用绿色 (0, 255, 0) 的线条绘制在图像上，线宽为2像素。你可以根据需要更改颜色和线宽。
        cv2.drawContours(img_copy, contours, idx, (0, 255, 255), 2)
        # 显示可视化的图像
        self.show_img("Contours Image", img_copy)

    def save_img(self, current_sample, iter_sample, bis_Train, random_state_seed):
        label = iter_sample['label']
        index = iter_sample['Sample_index']

        # 要检查的文件夹名称
        folder_name = "Meisteel_image_detection_result\\"

        # 获取当前工作目录
        current_directory = os.getcwd()

        # 构建完整路径
        folder_path = os.path.join(current_directory, folder_name, "random state = " + str(random_state_seed) + "\\",
                                   "train\\" if bis_Train else "test\\")

        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            # 如果不存在，创建文件夹
            os.makedirs(folder_path)
        else:
            pass

        image_class = "crack" if label == 1 else "normal"
        ##############################################################save image of first row

        saved_path = os.path.join(folder_path, image_class,
                                  str(index) + "_" + "{:.0f}".format(iter_sample['Sample_column'])) if label == 1 \
            else os.path.join(folder_path, image_class, str(index))
        if not os.path.exists(saved_path):
            # 如果不存在，创建文件夹
            os.makedirs(saved_path)
        saved_path += "\\saved_final_image_first_row.jpg"

        scale_factor = 3
        # 使用cv2.resize来调整图像尺寸
        resized_image = cv2.resize(current_sample[0], (self.img_width * scale_factor, self.img_height * scale_factor))
        cv2.imwrite(saved_path, resized_image)
        ###############################################################save image of second row

        saved_path = os.path.join(folder_path, image_class,
                                  str(index) + "_" + "{:.0f}".format(iter_sample['Sample_column'])) if label == 1 \
            else os.path.join(folder_path, image_class, str(index))
        if not os.path.exists(saved_path):
            # 如果不存在，创建文件夹
            os.makedirs(saved_path)
        saved_path += "\\saved_final_image_second_row.jpg"

        scale_factor = 3
        # 使用cv2.resize来调整图像尺寸
        resized_image = cv2.resize(current_sample[1], (self.img_width * scale_factor, self.img_height * scale_factor))
        cv2.imwrite(saved_path, resized_image)
        ###############################################################save image of third row

        saved_path = os.path.join(folder_path, image_class,
                                  str(index) + "_" + "{:.0f}".format(iter_sample['Sample_column'])) if label == 1 \
            else os.path.join(folder_path, image_class, str(index))
        if not os.path.exists(saved_path):
            # 如果不存在，创建文件夹
            os.makedirs(saved_path)
        saved_path += "\\saved_final_image_third_row.jpg"

        scale_factor = 3
        # 使用cv2.resize来调整图像尺寸
        resized_image = cv2.resize(current_sample[2], (self.img_width * scale_factor, self.img_height * scale_factor))
        cv2.imwrite(saved_path, resized_image)

    def find_contour_info(self, contour):
        # 计算轮廓的矩形包围框
        x_box, y_box, w_box, h_box = cv2.boundingRect(contour)

        # 计算轮廓的中心坐标
        center_x = x_box + w_box // 2
        center_y = y_box + h_box // 2

        x_left_boundary = x_box
        x_right_boundary = x_box + w_box

        # if center_x - x_left_boundary > 20 or x_right_boundary - center_x > 20:
        #     if c in self.bcorr_expa:
        #         pass
        #     else:
        #         self.bcorr_expa.append(c)
        # else:
        #     pass

        return {"center_x": center_x, "center_y": center_y, "x_box": x_box, "y_box": y_box, "w_box": w_box,
                "h_box": h_box, "contour": contour}

    def define_mask(self, image, lower_bound, upper_bound):
        # Create a mask for the corresponding region
        defined_mask = cv2.inRange(image, lower_bound, upper_bound)
        return defined_mask

    def find_mask_contour(self, image, mask):
        _, mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.show_contour(image, mask_contours)

        # Filter contours based on area
        filtered_contours = [cnt for cnt in mask_contours if cv2.contourArea(cnt) >= self.threshold_area]
        self.show_contour(image, filtered_contours)

        # # 使用connectedComponentsWithStats找到连通组件和统计信息
        # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # # 遍历每个连通组件（除了第一个，因为它是背景）
        # for label in range(1, num_labels):
        #     # 获取组件的面积
        #     area = stats[label, cv2.CC_STAT_AREA]
        #
        #     # 如果面积大于阈值，则保留该组件，否则将其置为0（背景）
        #     if area > self.threshold_area:
        #         mask[labels == label] = 255
        #     else:
        #         mask[labels == label] = 0

        # cv2.imshow('Filtered Mask', mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        mask_dict = {}
        # iterate over the contour of each mask region
        for i in range(len(filtered_contours)):
            # if len(filtered_contours[i]) >= 5:
            #     # 拟合椭圆(行不通，因为很多纵裂图像拟合的椭圆也是和竖直方向夹角为90°左右)
            #     ellipse = cv2.fitEllipse(filtered_contours[i])
            #
            #     # 获取椭圆的角度
            #     angle = ellipse[2]
            #
            #     # 判断角度是否接近竖直方向
            #     if abs(angle) < 20:
            #         # 这是纵裂
            #         pass
            #     else:
            #         pass
            #         # 在图像上绘制拟合的椭圆,区域不属于纵裂
            #         cv2.ellipse(image, ellipse, (255, 255, 255),  2)
            #
            # else:
            #     pass

            mask_dict[i] = self.find_contour_info(filtered_contours[i])
        return mask_dict, mask

    def confidence_score(self, height_diff, center_x_diff, bb_w, bb_h):

        # height_difference: 红色区域与蓝色区域中心在y方向的高度差
        # center_accuracy: 红色和蓝色区域中心点的定位准确性

        # 定义权重，可以根据具体情况进行调整
        weight_height = 0.4
        weight_center = 0.6

        # 计算得分
        score = weight_height * (height_diff / bb_h) + weight_center * (center_x_diff / bb_w)

        return score

    def non_max_suppression(self, boxes, overlap_threshold=0.15):
        # 如果没有框，直接返回空列表
        if len(boxes) == 0:
            return []

        # 提取框的坐标
        x1 = np.array([item[0]['top left x'] for item in boxes])
        y1 = np.array([item[0]['top left y'] for item in boxes])
        x2 = np.array([item[0]['bottom right x'] for item in boxes])
        y2 = np.array([item[0]['bottom right y'] for item in boxes])

        # 计算每个框的面积
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        # 按照置信度排序
        idxs = np.argsort(np.array([item[1] for item in boxes]))[::-1]

        # 初始化一个空列表来保存筛选后的框
        pick = []

        # 循环直到框的索引列表为空
        while len(idxs) > 0:
            # 获取当前置信度最高的框的索引
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # 计算当前框与其余框的交叠部分的坐标
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # 计算交叠部分的宽度和高度，确保这些值都是正数
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # 计算IoU
            IoU = (w * h) / (area[idxs[:last]] + area[idxs[last]] - w * h)

            idxs = np.delete(idxs, np.concatenate(([last], np.where(IoU > overlap_threshold)[0])))

        # 返回筛选后的框的索引
        return pick

    # 使用 non_max_suppression 函数来应用非极大值抑制
    def apply_nms(self, detected_boxes, overlap_threshold=0.15):
        # 将框的坐标和置信度合并成一个数组
        # boxes = np.column_stack((detected_boxes, scores))
        # 应用非极大值抑制
        selected_indices = self.non_max_suppression(detected_boxes, overlap_threshold)

        return selected_indices

    def crack_recognition_refining(self, bis_train, refining_list):
        # 两个指定的文件夹
        folder_path_1 = ".\\MeiSteel Longitudinal crack detection\\Crack"
        folder_path_2 = ".\\MeiSteel Longitudinal crack detection\\Normal"

        # 读取两类样本的图像数据
        Crack = self.read_images_from_folder(folder_path_1, label=1)
        Normal = self.read_images_from_folder(folder_path_2, label=0)

        # 合并数据
        all_data = pd.DataFrame(Crack + Normal)

        # 分层的训练-测试拆分，保持随机种子一致
        random_state_seed = 42
        train_data, test_data = train_test_split(all_data, test_size=0.2, stratify=all_data['label'],
                                                 random_state=random_state_seed)

        bis_Train = bis_train
        iter_dataset = train_data if bis_Train else test_data
        crack_num = 0
        for _, row in iter_dataset.iterrows():
            if row["Sample_index"] in refining_list:
                current_sample = []
                # load image
                image_row1 = self.load_image(row["第一排"][0])
                image_row2 = self.load_image(row["第二排"][0])
                image_row3 = self.load_image(row["第三排"][0])
                current_sample += [image_row1, image_row2, image_row3]
                self.save_img(current_sample, row, bis_Train, random_state_seed)
            else:
                pass


    def crack_recognition(self, bis_train):
        # 两个指定的文件夹
        folder_path_1 = ".\\MeiSteel Longitudinal crack detection\\Crack"
        folder_path_2 = ".\\MeiSteel Longitudinal crack detection\\Normal"

        # 读取两类样本的图像数据
        Crack = self.read_images_from_folder(folder_path_1, label=1)
        Normal = self.read_images_from_folder(folder_path_2, label=0)

        # 合并数据
        all_data = pd.DataFrame(Crack + Normal)

        # 分层的训练-测试拆分，保持随机种子一致
        random_state_seed = 42
        train_data, test_data = train_test_split(all_data, test_size=0.2, stratify=all_data['label'],
                                                 random_state=random_state_seed)

        # 输出训练集和测试集的信息
        print("Training set:")
        print(train_data['label'].value_counts())
        print("\nTesting set:")
        print(test_data['label'].value_counts())

        bis_Train = bis_train
        iter_dataset = train_data if bis_Train else test_data
        crack_num = 0
        for _, row in iter_dataset.iterrows():
            # if row["label"] == label:
            current_sample = []
            # load image
            image_row1 = self.load_image(row["第一排"][0])
            image_row2 = self.load_image(row["第二排"][0])
            image_row3 = self.load_image(row["第三排"][0])
            current_sample += [image_row1, image_row2, image_row3]
            print("index of current sample = ", row['Sample_index'])
            if row["Sample_column"] != -1:
                print("column of current sample = ", int(row["Sample_column"]))
            else:
                pass
            for i in range(len(current_sample)):
                # 调整后的红色范围
                lower_red = np.array([0, 0, 20], dtype=np.uint8)
                upper_red = np.array([120, 120, 255], dtype=np.uint8)
                # 创建红色掩码
                red_mask = self.define_mask(current_sample[i], lower_red, upper_red)
                plt.imshow(red_mask)
                # 调整后的蓝色范围
                lower_blue = np.array([70, 0, 0], dtype=np.uint8)
                upper_blue = np.array([255, 120, 120], dtype=np.uint8)
                # 创建蓝色掩码
                blue_mask = self.define_mask(current_sample[i], lower_blue, upper_blue)
                plt.imshow(blue_mask)
                # find contours
                red_mask_dict, red_mask_filtered = self.find_mask_contour(current_sample[i], red_mask)
                blue_mask_dict, blue_mask_filtered = self.find_mask_contour(current_sample[i], blue_mask)

                # crack detection
                self.run_detection(red_mask_dict, blue_mask_dict, i, row)
            self.define_crack_info()
            self.capture_crack_info(current_sample)
            for j in range(len(current_sample)):
                self.show_img("detection_result", current_sample[j])
            self.save_img(current_sample, row, bis_Train, random_state_seed)
            if self.bis_crack_detected:
                self.extract_features_from_bbox()
                self.samples_feature.append({'Sample_index': int(row["Sample_index"]), 'label': row["label"],
                                             'features': self.combined_features})
                self.combined_features = np.array([])
                crack_num += len(self.crack_info[0])
                print("the current sample is detected as crack.")
                if row['label'] == 0:
                    self.fp_index.append(row["Sample_index"])
                else:
                    pass
            else:
                print("the current sample is detected as normal.")
            print("-------------------------------------------------------")
            # clear the current list
            self.crack_info = [{} for _ in range(self.num_rows)]
            self.crack_num = [0 for _ in range(self.num_rows)]

        print("the total number of cracks is ", crack_num)

    def crack_dict_sort(self, keys_del_list, i):
        for keys_to_del in keys_del_list:
            del self.crack_info[i][keys_to_del]
        # 对字典的键进行排序
        sorted_keys = sorted(self.crack_info[i].keys())
        # 创建新的字典，按自然数排序后的键重新排列
        self.crack_info[i] = {sorted_keys.index(new_key) + 1: self.crack_info[i][new_key] for new_key in sorted_keys}

    def machine_learning_detection(self, bis_training):
        X = np.empty((0, len(self.samples_feature[0]["features"])))
        y = np.array([])
        Sample_index_array = np.array([])
        for i in range(len(self.samples_feature)):
            X = np.vstack((X, self.samples_feature[i]["features"]))
            y = np.hstack((y, self.samples_feature[i]["label"]))
            Sample_index_array = np.hstack((Sample_index_array, self.samples_feature[i]["Sample_index"]))

        # 存储分类器性能结果
        results = []
        for w in range(1, self.num_weight_selection + 1):
        # for w in range(5, 6):     # 确定权重w的最佳值为5

            # 定义类别权重
            class_weights = {0: 1, 1: w}  # 假设类别0的权重是1，类别1的权重是10
            # class_weights = "balanced"
            # # 定义难分类样本的索引
            # difficult_samples_index = self.fp_index  # 误报样本索引
            if bis_training:
                # 创建分类器列表z
                self.models[w - 1] = [
                    ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight=class_weights)),
                    ("SVM", SVC(probability=True, class_weight=class_weights)),
                    ("Random Forest", RandomForestClassifier(class_weight=class_weights)),
                    # ("Decision Tree", DecisionTreeClassifier(class_weight=class_weights)),
                    ("AdaBoost", AdaBoostClassifier(base_estimator=DecisionTreeClassifier(class_weight=class_weights))),
                    # ("Extra Trees", ExtraTreesClassifier(class_weight=class_weights)),
                ]
            else:
                pass

            self.crack_refining_indices[w] = {}
            for model_name, model in self.models[w - 1]:
                if bis_training:
                    print(f"Training {model_name}...")
                    # 训练模型
                    model.fit(X, y)
                else:
                    print(f"Testing {model_name}...")

                # 预测
                y_pred = model.predict(X)
                self.crack_refining_indices[w].update({model_name: Sample_index_array[np.where(y_pred == 0)[0]]})

                # 评估性能
                accuracy = accuracy_score(y, y_pred)
                report = classification_report(y, y_pred, zero_division=1)
                conf_matrix = confusion_matrix(y, y_pred)

                true_positive = conf_matrix[1, 1]
                false_negative = conf_matrix[1, 0]
                false_positive = conf_matrix[0, 1]
                true_negative = conf_matrix[0, 0]
                # 存储结果
                results.append({
                    "Model": model_name,
                    "Accuracy": accuracy,
                    "Classification Report": report,
                    "Confusion Matrix": conf_matrix
                })

                print(f"Accuracy for {model_name}: {accuracy}")
                print(f"Classification Report for {model_name}:\n{report}")
                print(f"Confusion Matrix for {model_name}:\n{conf_matrix}")

                # output the visualization of confusion matrix as well as the relevant metrics
                conf_matrix_visualization(true_positive, false_negative, false_positive, true_negative)

        # 输出所有模型的性能结果
        results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Classification Report", "Confusion Matrix"])
        print(results_df)
        # 清空特征存储列表
        self.samples_feature.clear()

    def run_detection(self, red_mask_dict, blue_mask_dict, i, row):
        time_captured = row["第一排"][1] if i == 0 else row["第二排"][1] if i == 1 else row["第三排"][1]
        # finding recursively through the red_mask_dict and blue_mask_dict to see if there are any red_contour and blue contour matches
        for red_key, red_value in red_mask_dict.items():
            for blue_key, blue_value in blue_mask_dict.items():
                center_red_x = red_value["center_x"]
                center_red_y = red_value["center_y"]
                red_contour = red_value["contour"]
                red_height = red_value["h_box"]
                center_blue_x = blue_value["center_x"]
                center_blue_y = blue_value["center_y"]
                blue_contour = blue_value["contour"]
                blue_height = blue_value["h_box"]

                if abs(center_red_x - center_blue_x) < self.horizontal_thre and 0 < center_blue_y - center_red_y < self.vertical_thre\
                        and red_height >= self.region_height_thre and blue_height >= self.region_height_thre:

                    # calculate the min and max values of x coordinate between red region and blue region
                    self.crack_info[i][self.crack_num[i]] = {"top left x": min(red_value["x_box"], blue_value["x_box"]),
                                                             "top left y": red_value["y_box"],
                                                             "bottom right x": max(blue_value["x_box"] + blue_value["w_box"],
                                                            red_value["x_box"] + red_value["w_box"]),
                                                             "bottom right y": blue_value["y_box"] + blue_value["h_box"],
                                                             "center_red_x": center_red_x, "center_red_y": center_red_y,
                                                             "center_blue_x": center_blue_x, "center_blue_y": center_blue_y,
                                                             "bb_x": (center_red_x + center_blue_x) / 2,
                                                             "red_contour": red_contour, "blue_contour": blue_contour,
                                                             "time_captured": time_captured}
                    self.crack_num[i] += 1

                else:
                    continue

        # remove redundant searches (bubble sort)
        keys_to_delete = []
        for j in range(self.crack_num[i]):
            for k in range(j + 1, self.crack_num[i]):
                if self.crack_info[i][j]["center_blue_x"] == self.crack_info[i][k]["center_blue_x"] and \
                        self.crack_info[i][j]["center_blue_y"] == self.crack_info[i][k]["center_blue_y"]:
                    if self.crack_info[i][j]["center_red_y"] <= self.crack_info[i][k]["center_red_y"]:
                        keys_to_delete.append(j)
                    else:
                        keys_to_delete.append(k)
                elif self.crack_info[i][j]["center_red_x"] == self.crack_info[i][k]["center_red_x"] and \
                        self.crack_info[i][j]["center_red_y"] == self.crack_info[i][k]["center_red_y"]:
                    if self.crack_info[i][j]["center_blue_y"] <= self.crack_info[i][k]["center_blue_y"]:
                        keys_to_delete.append(k)
                    else:
                        keys_to_delete.append(j)
                else:
                    pass

        # remove duplicate elements in the list
        keys_to_delete = list(set(keys_to_delete))
        self.crack_dict_sort(keys_to_delete, i)

    # judge crack info between different rows
    def define_crack_info(self):
        max_iou = 0.
        keys_to_retain_overall = [[], [], []]
        keys_to_delete_overall = [[], [], []]
        for key_row_1 in self.crack_info[0].keys():
            time_row_1 = self.crack_info[0][key_row_1]["time_captured"]
            time_row_1_sec = int(time_row_1[0:2]) * 3600 + int(time_row_1[3:5]) * 60 + int(time_row_1[6:8])
            for key_row_2 in self.crack_info[1].keys():
                time_row_2 = self.crack_info[1][key_row_2]["time_captured"]
                time_row_2_sec = int(time_row_2[0:2]) * 3600 + int(time_row_2[3:5]) * 60 + int(time_row_2[6:8])
                for key_row_3 in self.crack_info[2].keys():
                    time_row_3 = self.crack_info[2][key_row_3]["time_captured"]
                    time_row_3_sec = int(time_row_3[0:2]) * 3600 + int(time_row_3[3:5]) * 60 + int(time_row_3[6:8])
                    if abs(self.crack_info[0][key_row_1]["bb_x"] - self.crack_info[1][key_row_2]["bb_x"]) < self.horizontal_thre and \
                    abs(self.crack_info[1][key_row_2]["bb_x"] - self.crack_info[2][key_row_3]["bb_x"]) < self.horizontal_thre:
                        time_max_rows = max(time_row_1_sec, time_row_2_sec, time_row_3_sec)
                        bottom_pos_row1 = self.crack_info[0][key_row_1]["bottom right y"] + time_max_rows - time_row_1_sec
                        bottom_pos_row2 = self.crack_info[1][key_row_2]["bottom right y"] + time_max_rows - time_row_2_sec
                        bottom_pos_row3 = self.crack_info[2][key_row_3]["bottom right y"] + time_max_rows - time_row_3_sec
                        if 0 <= bottom_pos_row1 - bottom_pos_row2 <= self.time_lag_thre and 0 <= bottom_pos_row2 - bottom_pos_row3 <= self.time_lag_thre:
                            bbox_row1 = (self.crack_info[0][key_row_1]["top left x"], self.crack_info[0][key_row_1]["top left y"],
                                         self.crack_info[0][key_row_1]["bottom right x"] - self.crack_info[0][key_row_1]["top left x"],
                                         self.crack_info[0][key_row_1]["bottom right y"] - self.crack_info[0][key_row_1]["top left y"])

                            bbox_row2 = (self.crack_info[1][key_row_2]["top left x"], self.crack_info[1][key_row_2]["top left y"],
                                         self.crack_info[1][key_row_2]["bottom right x"] - self.crack_info[1][key_row_2]["top left x"],
                                         self.crack_info[1][key_row_2]["bottom right y"] - self.crack_info[1][key_row_2]["top left y"])

                            bbox_row3 = (self.crack_info[2][key_row_3]["top left x"], self.crack_info[2][key_row_3]["top left y"],
                                         self.crack_info[2][key_row_3]["bottom right x"] - self.crack_info[2][key_row_3]["top left x"],
                                         self.crack_info[2][key_row_3]["bottom right y"] - self.crack_info[2][key_row_3]["top left y"])

                            print("The average IOU of the bounding boxes located at the top left (x = {:.0f}, y = {:.0f}) "
                                  "in the first row, the top left (x = {:.0f}, y = {:.0f}) in the second row, and "
                                  "the top left (x = {:.0f}, y = {:.0f}) in the third row is {:.2f} for absolute value "
                                  "and {:.2f} for relative value.".format(
                            self.crack_info[0][key_row_1]["top left x"], self.crack_info[0][key_row_1]["top left y"],
                                  self.crack_info[1][key_row_2]["top left x"], self.crack_info[1][key_row_2]["top left y"],
                                  self.crack_info[2][key_row_3]["top left x"], self.crack_info[2][key_row_3]["top left y"],
                                  self.calculate_iou_absolute(bbox_row1, bbox_row2, bbox_row3),
                                  self.calculate_iou_relative(bbox_row1, bbox_row2, bbox_row3)))

                            current_iou = self.calculate_iou_relative(bbox_row1, bbox_row2, bbox_row3)
                            if current_iou > max_iou:
                                max_iou = current_iou
                                if not keys_to_retain_overall[0]:
                                    keys_to_retain_overall[0].append(key_row_1)
                                    keys_to_retain_overall[1].append(key_row_2)
                                    keys_to_retain_overall[2].append(key_row_3)
                                else:
                                    keys_to_retain_overall[0].pop()
                                    keys_to_retain_overall[0].append(key_row_1)

                                    keys_to_retain_overall[1].pop()
                                    keys_to_retain_overall[1].append(key_row_2)

                                    keys_to_retain_overall[2].pop()
                                    keys_to_retain_overall[2].append(key_row_3)
                            elif current_iou == max_iou:
                                keys_to_retain_overall[0].append(key_row_1)
                                keys_to_retain_overall[1].append(key_row_2)
                                keys_to_retain_overall[2].append(key_row_3)
                            else:
                                pass

                        else:
                            pass
                    else:
                        pass

        for key_row_1 in self.crack_info[0].keys():
            if key_row_1 not in keys_to_retain_overall[0]:
                keys_to_delete_overall[0].append(key_row_1)
            else:
                continue

        for key_row_2 in self.crack_info[1].keys():
            if key_row_2 not in keys_to_retain_overall[1]:
                keys_to_delete_overall[1].append(key_row_2)
            else:
                continue

        for key_row_3 in self.crack_info[2].keys():
            if key_row_3 not in keys_to_retain_overall[2]:
                keys_to_delete_overall[2].append(key_row_3)
            else:
                continue

        # delete non-crack keys
        for k in range(len(keys_to_delete_overall)):
            self.crack_dict_sort(keys_to_delete_overall[k], k)

    def calculate_iou_absolute(self, bbox1, bbox2, bbox3):
        def iou(b1, b2):
            # 获取宽高
            _, _, w1, h1 = b1
            _, _, w2, h2 = b2

            # 把包围框置于原点位置
            x1, y1, x2, y2 = 0, 0, 0, 0

            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1 + w1, x2 + w2)
            yi2 = min(y1 + h1, y2 + h2)
            inter_width = max(0, xi2 - xi1)
            inter_height = max(0, yi2 - yi1)
            inter_area = inter_width * inter_height

            bbox1_area = w1 * h1
            bbox2_area = w2 * h2
            union_area = bbox1_area + bbox2_area - inter_area

            iou_value = inter_area / union_area
            return iou_value

        # 计算三个包围框之间的IOU值
        iou12 = iou(bbox1, bbox2)
        iou13 = iou(bbox1, bbox3)
        iou23 = iou(bbox2, bbox3)

        # 返回平均值
        return (iou12 + iou13 + iou23) / 3

    def calculate_iou_relative(self, bbox1, bbox2, bbox3):
        def iou(b1, b2):
            # 获取宽高
            x1, y1, w1, h1 = b1
            x2, y2, w2, h2 = b2

            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1 + w1, x2 + w2)
            yi2 = min(y1 + h1, y2 + h2)
            inter_width = max(0, xi2 - xi1)
            inter_height = max(0, yi2 - yi1)
            inter_area = inter_width * inter_height

            bbox1_area = w1 * h1
            bbox2_area = w2 * h2
            union_area = bbox1_area + bbox2_area - inter_area

            iou_value = inter_area / union_area
            return iou_value

        # 计算三个包围框之间的IOU值
        iou12 = iou(bbox1, bbox2)
        iou13 = iou(bbox1, bbox3)
        iou23 = iou(bbox2, bbox3)

        # 返回平均值
        return (iou12 + iou13 + iou23) / 3

    def extract_features_from_bbox(self):

        for i in range(len(self.crack_info)):
            bbox_h = self.crack_info[i][1]["bottom right y"] - self.crack_info[i][1]["top left y"]
            bbox_w = self.crack_info[i][1]["bottom right x"] - self.crack_info[i][1]["top left x"]
            bbox_red_area = cv2.contourArea(self.crack_info[i][1]["red_contour"])
            bbox_blue_area = cv2.contourArea(self.crack_info[i][1]["blue_contour"])
            self.combined_features = np.hstack((self.combined_features, bbox_h / bbox_w, bbox_red_area / bbox_blue_area))

    def capture_crack_info(self, current_sample):
        self.bis_crack_detected = True
        for i in range(len(self.crack_info)):
            self.crack_num[i] = len(self.crack_info[i])
            if not self.crack_num[i]:
                self.bis_crack_detected = False
            else:
                pass
            for key, value in self.crack_info[i].items():

                tlx = self.crack_info[i][key]["top left x"]
                tly = self.crack_info[i][key]["top left y"]
                brx = self.crack_info[i][key]["bottom right x"]
                bry = self.crack_info[i][key]["bottom right y"]

                crx = self.crack_info[i][key]["center_red_x"]
                cry = self.crack_info[i][key]["center_red_y"]
                cbx = self.crack_info[i][key]["center_blue_x"]
                cby = self.crack_info[i][key]["center_blue_y"]

                # draw dash lines that passes the central point
                # horizontal
                cv2.line(current_sample[i], (0, cry), (self.img_width, cry), (0, 165, 255), 1, cv2.LINE_AA)
                cv2.line(current_sample[i], (0, cby), (self.img_width, cby), (0, 165, 255), 1, cv2.LINE_AA)

                # vertical
                cv2.line(current_sample[i], (crx, 0), (crx, self.img_height), (230, 200, 255), 1, cv2.LINE_AA)
                cv2.line(current_sample[i], (cbx, 0), (cbx, self.img_height), (230, 200, 255), 1, cv2.LINE_AA)

                # draw central points of red regions and blue regions
                cv2.circle(current_sample[i], (crx, cry), 1, (0, 255, 255), 2)
                cv2.circle(current_sample[i], (cbx, cby), 1, (0, 255, 255), 2)

                # Draw a bounding box around the longitudinal crack region
                cv2.rectangle(current_sample[i], (tlx, tly), (brx, bry), (0, 0, 0), 2)


if __name__ == "__main__":
    detector = Longitudinal_Crack_Detector()
    detector.crack_recognition(True)
    detector.machine_learning_detection(True)
    detector.crack_refining_indices.clear()     # 清空训练集中的误报样本索引
    detector.crack_recognition(False)
    detector.machine_learning_detection(False)
    detector.crack_recognition_refining(False, detector.crack_refining_indices[1]["SVM"])
