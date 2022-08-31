import torch
from sklearn.metrics import confusion_matrix,accuracy_score,balanced_accuracy_score
from tqdm import tqdm

def evaluate(model,dataloader_test,device,fout):
    with torch.no_grad():
        y_true, y_pred = test_evaluate(model,dataloader_test,device)
        accuracy = accuracy_score(y_true,y_pred)
        b_acc = balanced_accuracy_score(y_true,y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        print('accuracy: ' + str(accuracy))
        print('balanced_accuracy: ' + str(b_acc))
        print(conf_matrix)
        fout.write('accuracy: ' + str(accuracy) + '\n' +
            'balanced_accuracy: ' + str(b_acc) + '\n' +
            'conf_matrix: ' + '\n' + str(conf_matrix) + '\n'
            )
    return None

def test_evaluate(model,dataloader_test,device):
    model.eval()
    label_list,output_list = [], []
    for i, (feature, label) in enumerate(tqdm((dataloader_test),desc='Evaluation')):
        feature = feature.to(device)
        label = label.to(device)
        output = model(feature)
        for j in range(label.shape[0]):
            label_list.append(label[j])
            output_list.append(output[j])
    label_all = torch.stack(label_list,dim=0)
    # label_all = label_all.reshape(label_all.shape[0]*label_all.shape[1],label_all.shape[2])
    pred_all= torch.stack(output_list,dim=0)
    # pred_all = pred_all.reshape(pred_all.shape[0]*pred_all.shape[1],pred_all.shape[2])
    pred_all = torch.argmax(pred_all,dim=1)

    return label_all.data.cpu().numpy(), pred_all.data.cpu().numpy()
