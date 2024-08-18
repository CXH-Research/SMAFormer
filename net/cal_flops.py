from net import ResUformer, resunet_pp, ResUformer_tiny, ResUformer_base, resT
import torch
import argparse
from calflops import calculate_flops

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='ResUnet Plus Plus',
                        help='model name: (default: arch+timestamp)')

    args = parser.parse_args()

    return args


# 检查模型是否能够创建并输出期望的维度
args = parse_args()
model = ResUformer.ResUformer(args)
# model = ResUformer_base.ResUformer(args)
# model = ResUformer_tiny.ResUformer(args)
# model = resunet_pp.ResUnetPlusPlus(args)
# model = resT.ResTV2(embed_dims=[96, 192, 384, 768], depths=[1, 3, 16, 3])
# model.eval()
model = torch.nn.DataParallel(model).cuda()
x = torch.randn(16, 3, 512, 512)  # 假设输入是256x256的RGB图像

# calculate Flops
flops, macs, params = calculate_flops(model=model,
                                      input_shape=(x.shape[0], x.shape[1], x.shape[2], x.shape[3]),
                                      output_as_string=True,
                                      print_results=True,
                                      print_detailed=True,
                                      output_unit='M'
                                      )
print('%s -- FLOPs:%s  -- MACs:%s   -- Params:%s \n' % (args.model_name, flops, macs, params))
with torch.no_grad():  # 在不计算梯度的情况下执行前向传播
    out = model(x)
print(out.shape)  # 输出预期是与分类头的输出通道数匹配的特征图

'''ResUnet Plus Plus
Total Training Params:                                                  3.63 M  
fwd MACs:                                                               1.13137e+06 M
fwd FLOPs:                                                              2.27106e+06 M
fwd+bwd MACs:                                                           3.39411e+06 M
fwd+bwd FLOPs:                                                          6.81319e+06 M
'''

'''ResUformer Large
Total Training Params:                                                  142.74 M
fwd MACs:                                                               1.63582e+06 MMACs
fwd FLOPs:                                                              3.27801e+06 MFLOPS
fwd+bwd MACs:                                                           4.90746e+06 MMACs
fwd+bwd FLOPs:                                                          9.83404e+06 MFLOPS
'''

'''
Total Training Params:                                                  243.51 M
fwd MACs:                                                               2.04814e+06 MMACs
fwd FLOPs:                                                              4.10312e+06 MFLOPS
fwd+bwd MACs:                                                           6.14441e+06 MMACs
fwd+bwd FLOPs:                                                          1.23094e+07 MFLOPS

'''
'''ResUformer Base, Transformer_layer = 5
Total Training Params:                                                  67.65 M 
fwd MACs:                                                               1.10109e+06 MMACs
fwd FLOPs:                                                              2.20837e+06 MFLOPS
fwd+bwd MACs:                                                           3.30327e+06 MMACs
fwd+bwd FLOPs:                                                          6.6251e+06 MFLOPS
'''

'''ResUformer Tiny
Total Training Params:                                                  1.83 M  
fwd MACs:                                                               575912 MMACs
fwd FLOPs:                                                              1.15835e+06 MFLOPS
fwd+bwd MACs:                                                           1.72773e+06 MMACs
fwd+bwd FLOPs:                                                          3.47504e+06 MFLOPS
'''

'''ResT V2
Total Training Params:                                                  64.25 M 
fwd MACs:                                                               772097 MMACs
fwd FLOPs:                                                              1.54839e+06 MFLOPS
fwd+bwd MACs:                                                           2.31629e+06 MMACs
fwd+bwd FLOPs:                                                          4.64516e+06 MFLOPS
'''