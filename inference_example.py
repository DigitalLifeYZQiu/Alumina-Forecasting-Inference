# 依赖库导入
import argparse
import torch
import random
import numpy as np
import nni
import logging
from factory.alumina_MS_forecasting import AluminaTransformerMSForecasting

# 参数配置
params={
    # 需要配置的参数
    'inference_csv_path': '../TSFrameworkData/alumina_testset_new.csv',
    'inference_ckpt_path': 'inference_checkpoint/checkpoint-example.pth',
    'feature_cols': '5效5闪到调配槽后循环母液Nk,5效5闪到调配槽后循环母液ak/Rp,6效出料密度,5闪出料密度,57效循环母液进料温度,57效循环母液进料密度,冷凝器出汽压力,5效5闪蒸发母液去循环母液调配槽总流量,2效蒸发器汽温,2效蒸发器汽室压力,2效原液换热前温度,3效蒸发器汽温,3效蒸发器汽室压力,3效原液换热前温度,4效蒸发器汽温,4效蒸发器汽室压力,4效原液换热前温度,5效蒸发器汽温,5效蒸发器汽室压力,5效原液换热前温度,5效循环母液进料流量,6效蒸发器汽温,6效蒸发器汽室压力,6效原液换热前温度,原液闪蒸器循环母液进料流量,原液闪蒸器出料温度,原液闪蒸器蒸汽压力,7效原液换热前温度,7效蒸发器汽温,7效蒸发器汽室压力,1闪出料温度,2闪出料温度,3闪出料温度,4闪出料温度,5闪出料温度,5闪去循环母液调配槽密度,5闪去循环母液调配槽流量,新蒸汽进料压力,新蒸汽进料流量,新蒸汽进料温度,循环上水温度,循环上水流量,循环下水温度,原液闪蒸器循环母液进料阀开度,1效出料阀门开度,1闪出料阀开度,3闪出料阀开度,4闪出料阀开度,2效到1效泵开度,4效到3效泵开度,5效到4效泵开度,6效出料出料泵开度,1效蒸发器汽室压力,1效原液换热前温度,1效出料温度,1效蒸发器汽温',
    'target_cols': '1效蒸发器汽温',
    # 其他可以使用默认参数
    'use_nni': True,
    'stage': 'train_only',
    'model_id': 'alumina_timexer',
    'log_path': 'log',
    'checkpoints': './checkpoints/allfeature/timexer',
    'key': None,
    'task_name': 'Forecast',
    'features': 'MS',
    'embed': 'timeF',
    'freq': 'h',
    'datetime_col': 'dtime',
    'timestamp_feature': 'none',
    'random_features': False,
    'random_features_num': 10,
    'seq_len': 30,
    'label_len': 2,
    'pred_len': 30,
    'interval': 900,
    'model_name': 'alumina_timexer',
    'd_model': 128,
    'n_heads': 8,
    'e_layers': 1,
    'd_layers': 1,
    'd_ff': 512,
    'hidden_dim': 128,
    'factor': 1,
    'dropout': 0.05,
    'activation': 'gelu',
    'output_attention': False,
    'optimizer': 'Adam',
    'fix_seed': -1,
    'num_workers': 10,
    'train_epochs': 50,
    'batch_size': 64,
    'patience': 7,
    'lr': 0.0001,
    'gpu': 0,
    'label_loss_rate': 0.5,
    'reg_loss_rate': 0.5,
    'use_norm': 1,
    'patch_len': 16,
    'target_num': 1,
    'use_gpu': True,
    'hpo': 'nni'
}
args = argparse.Namespace(**params)
print(args)



# 使用配置参数导入模型
def load_model(args, AluminaTransformerMSForecasting):
    forecasting_model = AluminaTransformerMSForecasting(args)
    return forecasting_model

forecasting_model = load_model(args, AluminaTransformerMSForecasting)

# 将输入的list转换为tensor形式
def list_to_input_tensor(input_list):
    input_array = np.array(input_list)
    if len(input_array.shape)<3:
        input_array = np.expand_dims(input_array, axis=0)
        sample = torch.Tensor(input_array).to(forecasting_model.device)
    else:
        sample = torch.Tensor(input_array).to(forecasting_model.device)
    return sample

def output_tensor_to_list(output_tensor):
    if output_tensor.device.type == 'cuda':
        output = output_tensor.detach().to('cpu')
    target_scaler = inference_data.target_scaler
    if len(output.shape)==3:
        output = output.reshape(output.shape[1],output.shape[2])
    output = target_scaler.inverse_transform(output)
    return output.tolist()


# 数据集导入(dataset \ dataloader)
inference_data, inference_loader = forecasting_model._create_data(
    args=args,
    csv_path=args.inference_csv_path,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=1,
    drop_last=True,
    init_scaler=True,
    pattern='test'
)


# 获取数据
#   输入数据：sample（1，30，56）
#   输出数据：outputs（1，30，1）
#   回归标签：label（1，30，1）
index=1
sample_list = inference_data[index][0].tolist()
label_list = inference_data[index][1].tolist()
sample = list_to_input_tensor(sample_list)
label = list_to_input_tensor(label_list)

outputs = forecasting_model.inference(args=args,sample=sample)

predict_outputs = output_tensor_to_list(outputs)
import pdb
pdb.set_trace()
