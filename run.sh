python main.py --epochs 30 --lr 1e-4 --GRU_layers 2 --bidirectional True --outputresult ./result/exp-2gru-bidirectional/
python main.py --epochs 30 --lr 5e-4 --GRU_layers 2 --bidirectional True --outputresult ./result/exp-2gru-bidirectional-02/
python main.py --epochs 30 --lr 1e-2 --GRU_layers 2 --bidirectional True --outputresult ./result/exp-2gru-bidirectional-04/
python main.py --epochs 30 --lr 1e-3 --GRU_layers 2 --bidirectional True --outputresult ./result/exp-2gru-bidirectional-05/
python main.py --epochs 30 --lr 1e-6 --GRU_layers 2 --bidirectional True --outputresult ./result/exp-2gru-bidirectional-06/
python main.py --epochs 300 --lr 1e-6 --GRU_layers 2 --bidirectional True --outputresult ./result/exp-2gru-bidirectional-06/
python main.py --epochs 71 --lr 1e-4 --GRU_layers 2 --bidirectional True --outputresult ./result/exp-2gru-bidirectional-08/
python main.py --epochs 30 --lr 5e-5 --GRU_layers 2 --bidirectional True --outputresult ./result/exp-2gru-bidirectional-10/
### change model: add linear layers after GRU
python main.py --epochs 30 --lr 1e-5 --GRU_layers 2 --bidirectional True --outputresult ./result/exp-2gru-bidirectional-11/ --dropout 0.3
python main.py --epochs 30 --lr 1e-5 --GRU_layers 2 --bidirectional True --outputresult ./result/exp-2gru-bidirectional-12/ --dropout 0.5
python main.py --epochs 30 --lr 1e-5 --bidirectional True --outputresult ./result/exp-2gru-bidirectional-13/ #gru一层和两层效果很接近
python main.py --epochs 30 --lr 1e-5  --outputresult ./result/exp-2gru-bidirectional-14/ #单向和双向效果一样


#change model：
        #line39-40
            # nn.Linear(256,128)，
            # nn.ReLU(),
        #line28
            #self.nhid_pos=128 -> self.nhid_pos=512
python main.py --epochs 30 --lr 1e-5  --outputresult ./result/exp-1/

#use more lines in text: 10 ->20
python main.py --epochs 30 --lr 1e-5  --inputpath /home/data/toy/bert_feature20/ --outputresult ./result/exp-morelines-01/
python main.py --epochs 30 --lr 1e-5 --GRU_layers 2 --bidirectional True --outputresult ./result/exp-morelines-2gru-bidirectional-12/ --dropout 0.5


#fix_bug: match data and labels
python main.py --epochs 30 --lr 1e-5  --inputpath /home/data/toy/bert_feature20/ --outputresult ./result/exp-morelinesfix-01/
python main.py --epochs 20 --lr 1e-5  --inputpath /home/data/toy/bert_feature20/ --outputresult ./result/exp-morelinesfix-01-best/
python main.py --epochs 30 --lr 1e-5 --GRU_layers 2 --bidirectional True --outputresult ./result/exp-morelinesfix-2gru-bidirectional-01/ --dropout 0.5
python main.py --epochs 30 --lr 1e-5 --GRU_layers 2 --bidirectional True --outputresult ./result/exp-morelinesfix-2gru-bidirectional-02/ --dropout 0.3


python main.py --epochs 20 --lr 1e-5  --inputpath /home/data/toy/bert_feature20/ --outputresult ./result/exp-morelinesfix-01-best/
# CEC dataset
python main.py --batch_size 20 --epochs 30 --lr 1e-6  --inputpath /home/data/toy/bert_feature_CEC/ --outputresult ./result/exp-CEC-01/
    # not learnning lr maybe to large or too little
python main.py --batch_size 20 --epochs 30 --lr 1e-7  --inputpath /home/data/toy/bert_feature_CEC/ --outputresult ./result/exp-CEC-02/

python main.py --batch_size 20 --epochs 30 --lr 1e-4  --inputpath /home/data/toy/bert_feature_CEC/ --outputresult ./result/exp-CEC-03/

python main.py --batch_size 20 --epochs 35 --lr 2e-4  --inputpath /home/data/toy/bert_feature_CEC/ --outputresult ./result/exp-CEC-best/