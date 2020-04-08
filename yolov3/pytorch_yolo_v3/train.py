from yolov3.pytorch_yolo_v3.darknet import *

CUDA = True
dn = Darknet('yolov3.cfg')
# dn.load_weights("yolov3.weights")
inp = get_test_input()
# if CUDA:
#     inp = inp.cuda()
a, interms = dn(inp, CUDA=True)
dn.eval()
a_i, interms_i = dn(inp, CUDA=False)

