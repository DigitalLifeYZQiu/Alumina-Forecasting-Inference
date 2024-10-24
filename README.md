# Alumina Forcasting Inference API

This is an Inference API repo for alumina forecasting mission.

## Usage

1. Install Python 3.10 with necessary requirements. If you are using `Anaconda`, here is an example:

```shell
conda create -n alumina python=3.10 jupyter notebook
pip install -r requirements.txt
```

2. Prepare Checkpoint. We offer a checkpoint trained by TimeXer in route `inference_checkpoint`. Feel free to add your own checkpoint in this route.
3. Prepare Data. The [Alumina Dataset](https://cloud.tsinghua.edu.cn/d/773cc0263fa542ad8a02/) used for training is here. Download it and place into a preserved local path.
4. For other technichal details, please refer to an intricate tutorial in `inference_example.ipynbb`.

## Contact

If you have any questions or suggestions, feel free to contact us:

- Yunzhong Qiu (Master student, qiuyz24@mails.tsinghua.edu.cn)
