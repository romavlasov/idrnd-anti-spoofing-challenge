# ID R&D Anti-spoofing Challenge 1st place Solution ([link](https://datasouls.com/c/idrnd-antispoof/leaderboard))

Train model: <code>python train.py --config 'configuration file'</code>

Predict: <code>python test.py --path-images-csv 'annotation' --path-test-dir 'path to images' --path-submission-csv 'submission'</code>

Final solution is based on ensemble of two models: se_resnext50 and densnet121. 

The following configs are used to train this models: [se_resnext50_focal.yaml](https://github.com/romavlasov/idrnd-anti-spoofing-challenge/blob/master/config/se_resnext50_focal.yaml) and [densnet121_focal.yaml](https://github.com/romavlasov/idrnd-anti-spoofing-challenge/blob/master/config/densenet121_focal.yaml)

Pretrained models: [se_resnext50_focal.pth](https://www.dropbox.com/s/o0mpw0ep7ntamzv/se_resnext50_focal_model_epoch_best.pth?dl=0) and [densnet121_focal.pth](https://www.dropbox.com/s/i5utd1nooulyh7z/densenet121_focal_model_epoch_best.pth?dl=0)
