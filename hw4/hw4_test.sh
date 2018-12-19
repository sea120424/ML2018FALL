wget https://www.dropbox.com/s/2st6vh5d9usgrnx/w2v_3.bin?dl=0 -O w2v_3.bin
wget https://www.dropbox.com/s/se13nzjf284v2v4/best_4.h5?dl=0 -O best_4.h5
python3 predict_w2v.py $1 $2 $3 
