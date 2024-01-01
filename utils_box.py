import torch
import numpy as np
from shapely.geometry import Polygon
import time
from torch.utils.cpp_extension import load

def timer(func):
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Time elapsed: {(end-start):.4f}s.")
        return result
    return inner

# @timer
def iou(boxes, anchors, inter_only=False):
    'perform iou calculation on rotated boxes and anchors'
    'boxes and anchors are by default torch tensors, extraction and conversion to np.array may be required'
    boxes_np = boxes.detach().cpu().numpy()
    anchors_np = anchors.detach().cpu().numpy()
    table_np = np.zeros((anchors_np.shape[0], boxes_np.shape[0]), dtype=np.float32)
    # print(table_np.shape)
    # print(f"Boxes shape:{boxes.size()}")
    # print(f"Anchors shape:{anchors.size()}")
    # print(f"Boxes np shape:{boxes_np.shape}")
    # print(f"Anchors np shape:{anchors_np.shape}")
    boxes_polygon = [get_polygon(list(box)) for box in boxes_np]
    anchors_polygon = [get_polygon(list(anchor)) for anchor in anchors_np]
    if inter_only:
        # print(iou_table.shape)
        # print(iou_table)
        result = np.array([[compute_iou(b_poly, a_poly) for b_poly in boxes_polygon] for a_poly in anchors_polygon], dtype=np.float32)
    else:
        result = np.array([[compute_intersection(b_poly, a_poly) for b_poly in boxes_polygon] for a_poly in anchors_polygon], dtype=np.float32)
    assert table_np.shape == result.shape
    return torch.tensor(result, dtype=torch.float32)

# @timer
def iou_faster(boxes: torch.Tensor, anchors: torch.Tensor, inter_only=False):
    'perform iou calculation on rotated boxes and anchors'
    'boxes and anchors are by default torch tensors, no conversion to np array'
    table_np = torch.zeros(anchors.size()[0], boxes.size()[0], dtype=torch.float32)
    boxes_polygon = get_polygons_faster(boxes)
    anchors_polygon = get_polygons_faster(anchors)
    if inter_only:
        result = torch.tensor([[compute_iou(b_poly, a_poly) for b_poly in boxes_polygon] for a_poly in anchors_polygon], dtype=torch.float32)
    else:
        result = torch.tensor([[compute_intersection(b_poly, a_poly) for b_poly in boxes_polygon] for a_poly in anchors_polygon], dtype=torch.float32)
    assert table_np.size() == result.size()
    return result


def get_polygon(base: list):
    assert len(base) == 8, "Incorrect input format"
    pt1 = (base[0], base[1])
    pt2 = (base[2], base[3])
    pt3 = (base[4], base[5])
    pt4 = (base[6], base[7])

    return Polygon([pt1, pt2, pt3, pt4])

def get_polygons_faster(base: torch.Tensor):
    'Convert torch tensor info of coordinates (size: (N, 8)), into an array of Polygons (with length N)'
    output_polygons = []
    for box in base:
        assert box.size()[0] == 8, "Incorrect input format"
        pt1 = (box[0].item(), box[1].item())
        pt2 = (box[2].item(), box[3].item())
        pt3 = (box[4].item(), box[5].item())
        pt4 = (box[6].item(), box[7].item())
        output_polygons.append(Polygon([pt1, pt2, pt3, pt4]))

    return output_polygons

def compute_iou(poly1, poly2):
    inter = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - inter
    return inter / union

def compute_intersection(poly1, poly2):
    return poly1.intersection(poly2).area

def load_cpp_extensions(verbose=False):
    cpp_extensions = load(name='cpp_extensions',
                          sources=['csrc/extensions.cpp',
                          'csrc/cuda/decode.cu',
                          'csrc/cuda/decode_rotate.cu',
                          'csrc/cuda/nms.cu',
                          'csrc/cuda/nms_iou.cu'],
                          extra_cflags=['-std=c++14', '-O2', '-Wall'],
                          extra_cuda_cflags=[
                              '-std=c++14', '--extended-lambda', '--expt-extended-lambda', '--use_fast_math', '-Xcompiler', '-Wall,-fno-gnu-unique',
                              '-gencode=arch=compute_60,code=sm_60', '-gencode=arch=compute_61,code=sm_61',
                              '-gencode=arch=compute_70,code=sm_70', '-gencode=arch=compute_72,code=sm_72',
                              '-gencode=arch=compute_75,code=sm_75', '-gencode=arch=compute_80,code=sm_80',
                              '-gencode=arch=compute_86,code=sm_86', '-gencode=arch=compute_86,code=compute_86'
                          ],
                          verbose=verbose,)
    return cpp_extensions

if __name__ == '__main__':
    # boxes = torch.rand(20, 8)
    # anchors = torch.rand(12, 8)
    boxes = torch.tensor([[0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0], [0.1, 0.1, 0.1, 1.5, 1.2, 1.5, 1.2, 0.1], [-1, 0, 0, 1, 1, 0, 0, -1]])
    anchors = torch.tensor([[-2, 0, 0, 2, 2, 0, 0, -2], [0, 0, 0, 1, 1, 1, 1, 0]])
    # print([ele for ele in boxes[0]])
    # print(torch.rand(12, 24, 12).view(-1).size())
    # print(anchors)
    # print(Polygon([(0,0),(0,1),(1,1),(1,0)]).area)
    # print(boxes.size()[-1])
    # print(anchors.size()[-1])
    # print(boxes.size()[-1] == anchors.size()[-1])
    # print(boxes.size() == anchors.size())
    iou_table = iou(boxes.contiguous(), anchors.contiguous())
    print(iou_table)
    best, indices = iou_table.max(1)
    print(best)
    print(indices)
    print(boxes[indices])

    iou_table = iou_faster(boxes.contiguous(), anchors.contiguous())
    print(iou_table)
    best, indices = iou_table.max(1)
    print(best)
    print(indices)
    print(boxes[indices])

    extension = load_cpp_extensions()
    print(extension)

    #print(indices.squeeze())

    #print(table[1].squeeze())
