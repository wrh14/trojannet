# trojannet

## Dependencies:
```
pytorch, imgaug, itertools, numpy, scipy, wget
```

## Example: training on public task and secret task. 

### Run TrojanResnet50 on CIFAR10 and SVHN. 
```
python train.py --epochs 300 --datasets_name cifar10 svhn --model trojan_resnet50 --seed 0 --data_root ./data --save_dir checkpoint
```
### Run TrojanResnet50 on CIFAR10, CIFAR100, SVHN and GTSRB.
```
python train.py --epochs 300 --datasets_name cifar10 cifar100 svhn gtsrb --model trojan_resnet50 --seed 0 --data_root ./data --save_dir checkpoint
```
