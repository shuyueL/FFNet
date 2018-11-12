# FFNet

This repository provides the implementation for video fast-forward with reinforcement learning, i.e.
FFNet in our paper:

**[FFNet: Video Fast-Forwarding via Reinforcement Learning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Lan_FFNet_Video_Fast-Forwarding_CVPR_2018_paper.pdf)**
<br>


![alt text](https://github.com/shuyueL/FFNet/blob/master/image/model_2.png "FFNet overview")

If you find the codes or other related resources from this repository useful, please cite the following paper:

```
@inproceedings{lan2018ffnet,
  title={FFNet: Video Fast-Forwarding via Reinforcement Learning},
  author={Lan, Shuyue and Panda, Rameswar and Zhu, Qi and Roy-Chowdhury, Amit K},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6771--6780},
  year={2018}
}
```

## Environment

- Windows or Linux
- NVIDIA GPU with compute capability 3.5+
- Python 3.5
- Tensorflow

## Data
The original data we used in paper are available from the following websites
* Tour20 dataset: https://vcg.ece.ucr.edu/
* TVSum dataset: https://github.com/yalesong/tvsum

## Codes
### Testing
We offer a testing example with a pre-trained model in the ./model directory. Download this repository and run the following command:
```
python nn_test.py
```
The fast-forward result will be in the ./output directory.
### Training
If you want to train the model on your own data, you can find the script for training in nn_train.py. For more details, please refer to our paper.
