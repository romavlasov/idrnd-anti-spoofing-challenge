# ID R&D Anti-spoofing Challenge Solution ([link](https://datasouls.com/c/idrnd-antispoof/leaderboard))

Train model: <code>python train.py --config 'configuration file'</code>

Final solution is based on ensemble of two models: se_resnext50 and densnet121. 

The following configs are used to train this models: [se_resnext50_bce.yaml](https://github.com/romavlasov/idrnd-anti-spoofing-challenge/blob/master/config/se_resnext50_bce.yaml) and [densnet121_bce.yaml](https://github.com/romavlasov/idrnd-anti-spoofing-challenge/blob/master/config/densenet121_bce.yaml)

Weights: [se_resnext50](https://www.dropbox.com/s/dmb8ptwk5ssgwyj/se_resnext50_bce_model_epoch_best.pth?dl=0) , [densnet121](https://www.dropbox.com/s/8oxkz2c8fkaucmu/densenet121_bce_model_epoch_best.pth?dl=0)
