The whole process mainly consists of two parts, the first part extracts BERT features, and the second part trains GRU. 1. BERT feature extraction see bert_process_CEC.py and bert_process_THUCNews.py e.g. python bert_process_THUCNews.py
--limit_file_num 10000 --limit_file_length 20 --feature_save_path ./bert_features/ python bert_process_THUCNews.py --help You can view the settings of related parameters

2. The GRU training code is see main.py e.g. python main.py --batch_size 20 --epochs 35 --lr 2e-4
--inputpath /home/data/toy/bert_feature_CEC/ --outputresult ./result/exp-CEC-best/ python main.py --batch_size 50 --epochs 15 --lr 1e-5
--inputpath /home/data/toy/bert_feature20/ --outputresult ./result/exp-THUCNews-best/

3. All model architectures are in model.py, including the bert model and the gru model. For details, see the two classes my_bert_model(nn.Module) class my_gru(nn.Module) The simplest single-layer one-way model used by GRU . After testing, the effect of double-layer/two-way is not as good as that of single-layer one-way. For the mathematical details of the GRU model, please refer to the official pytorch documentation: https://pytorch.org/docs/stable/generated/torch.nn.GRU.html?highlight=gru#torch.nn.GRU [Note] The output layer of the gru model The number of nodes needs to be equal to the number of sample categories, otherwise an error will be reported, see model.py line 41: nn.Linear(128,5)) ###if for THUCNews dataset, change to:nn.Linear(128,14) 4. The most See the records in best_run.sh for the best results. 5. Please pay attention to modify the code related to dataset path/feature file path/input path when running on your machine. 6. The native python environment is python3.6, pytorch1.8.0+cu111, NVIDIA-SMI 470.141.03 Driver Version: 470.141.03 CUDA Version: 11.4

    
