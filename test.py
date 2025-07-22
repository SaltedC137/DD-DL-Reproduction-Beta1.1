from utils.augmentation import *
from utils.Dataset import *
from utils.LabelProcessor import *
from utils.Metrics import *
import numpy as np
import matplotlib.pyplot as plt
import config
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

BATCH_SIZE = config.BATCH_SIZE
miou_list = [0]

my_test = MyDataset([config.TEST_ROOT, config.TEST_LABEL], get_validation_augmentation(config.val_size))
test_data = DataLoader(my_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

net = config.usemodel
net.eval()
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
else:
    net=net.to(device)
#参数选择
if config.Parameter_selection=="best":
    net.load_state_dict(torch.load(config.best_pth))  # 加载训练好的模型参数
elif config.Parameter_selection=="last":
    net.load_state_dict(torch.load(config.last_pth))  # 加载训练好的模型参数
else:
    print("Error!(请输入best或者last在config.Parameter_selection中)")
    raise("请返回重新输入！")

net = net.to(device)

error = 0
train_mpa = 0
train_miou = 0
train_class_acc = 0
train_pa = 0
train_recall=0
train_f1=0
train_precision=0
train_kappa=0
for i, sample in enumerate(test_data):
    data = Variable(sample['img']).to(device)
    label = Variable(sample['label']).to(device)
    out = net(data)
    out = F.log_softmax(out, dim=1)

    pre_label = out.max(dim=1)[1].data.cpu().numpy()
    pre_label = [i for i in pre_label]

    true_label = label.data.cpu().numpy()
    true_label = [i for i in true_label]

    eval_metrix = eval_semantic_segmentation(pre_label, true_label)
    train_mpa = eval_metrix['mean_class_accuracy'] + train_mpa
    train_miou = eval_metrix['miou'] + train_miou
    train_pa = eval_metrix['pixel_accuracy'] + train_pa
    train_recall=eval_metrix["recall"]+train_recall
    train_f1=eval_metrix["f1"]+train_f1
    train_precision=eval_metrix["precision"]+train_precision
    train_kappa=eval_metrix["kappa"]+train_kappa


    if len(eval_metrix['class_accuracy']) < config.class_num:                  #注意几分类
        eval_metrix['class_accuracy'] = 0
        train_class_acc = train_class_acc + eval_metrix['class_accuracy']
        error += 1
    else:
        train_class_acc = train_class_acc + eval_metrix['class_accuracy']

    print(eval_metrix['class_accuracy'], '================', i)


epoch_str = ('test_miou: {:.5f}, test_accuracy(pa): {:.5f},  test_recall: {:.5f},test_f1:{:.5f},test_precision:{:.5f},test_kappa:{:.5f}'.format(
    train_miou / (len(test_data) - error),
    train_pa / (len(test_data) - error),
    #train_class_acc/(len(test_data)-error),#类别精度
    train_recall / (len(test_data) - error),
    train_f1/ (len(test_data) - error),
    train_precision/ (len(test_data) - error),
    train_kappa/ (len(test_data) - error),
))
with open(config.test_result, 'w') as file:
    file.write(epoch_str)
if train_miou/(len(test_data)-error) > max(miou_list):
    miou_list.append(train_miou/(len(test_data)-error))
    print(epoch_str+'==========last')
