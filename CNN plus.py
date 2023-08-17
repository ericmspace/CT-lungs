from keras.callbacks import LearningRateScheduler
import math
from keras.applications import InceptionResNetV2

# 导入必要的库和模块
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, LeakyReLU
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import AdamW
from sklearn.metrics import confusion_matrix

# 定义训练、验证和测试数据的路径
train_dir = r"D:\archive\chest_xray\train"
val_dir = r"D:\archive\chest_xray\val"
test_dir = r"D:\archive\chest_xray\test"

# 设置数据增强配置以增加模型的泛化能力
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1. / 255,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.2],
    width_shift_range=0.15,
    height_shift_range=0.1
)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# 从指定目录加载训练、验证和测试数据
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    shuffle=True,
    class_mode='categorical'
)
val_data = test_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    shuffle=False,
    class_mode='categorical'
)
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    shuffle=False,
    class_mode='categorical'
)

# 使用预训练的 InceptionResNetV2 作为特征提取器
base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(150, 150, 3))

# 构建模型结构
model = Sequential()
model.add(base_model)

# 冻结预训练模型的所有层，这样在训练过程中就不会更新它们的权重
for layer in base_model.layers:
    layer.trainable = False

# 在模型后面添加自定义的层
model.add(GlobalAveragePooling2D())
model.add(LeakyReLU(alpha=0.01))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax', kernel_regularizer=l2(0.001)))  # 二分类输出层


# 定义余弦退火学习率调度策略
def cosine_annealing(epoch, max_lr=0.002, min_lr=0.00005, cycles=3):
    cos_in = (math.pi * (epoch % cycles)) / cycles
    return min_lr + (max_lr - min_lr) / 2 * (1 + math.cos(cos_in))


cosine_annealer = LearningRateScheduler(cosine_annealing, verbose=1)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)  # 提前终止策略，避免过拟合
callbacks_list = [cosine_annealer, early_stopper]

# 编译模型
opt = AdamW(learning_rate=0.0002, weight_decay=0.01)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

# 开始训练模型
h = model.fit(train_data, validation_data=val_data, epochs=50, callbacks=callbacks_list)

# 绘制训练和验证损失曲线
plt.plot(h.history["loss"], label="Training Loss")
plt.plot(h.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 预测测试数据
pred = model.predict(test_data)
pred = np.argmax(pred, axis=1)

# 打印模型的结构
model.summary()

# 输出预测结果
print(pred)

# 生成并输出混淆矩阵
cm = confusion_matrix(test_data.classes, pred)
print(cm)

# 计算并输出准确率
accuracy = (cm[0, 0] + cm[1, 1]) / sum(sum(cm))
print(accuracy)

# 从混淆矩阵中获取TP, FP, FN, TN的值
TP = cm[1, 1]
FP = cm[0, 1]
FN = cm[1, 0]
TN = cm[0, 0]

# 计算精确度
precision = TP / (TP + FP)

# 计算召回率
recall = TP / (TP + FN)

# 计算F1分数
f1_score = 2 * (precision * recall) / (precision + recall)

# 输出指标
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")

