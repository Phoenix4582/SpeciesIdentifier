#!/usr/bin/env python3
import sys
import os
import shutil
import argparse
import random
import torch.cuda
import torch.distributed
import torch.multiprocessing

import infer, train, utils, video_infer
from model import Model


def parse(args):
    parser = argparse.ArgumentParser(description='ODTK: Object Detection Toolkit.')
    parser.add_argument('--master', metavar='address:port', type=str, help='Address and port of the master worker',
                        default='127.0.0.1:29500')

    subparsers = parser.add_subparsers(help='sub-command', dest='command')
    subparsers.required = True

    devcount = max(1, torch.cuda.device_count())

    parser_train = subparsers.add_parser('train', help='train a network')
    parser_train.add_argument('model', type=str, help='path to output model or checkpoint to resume from')
    parser_train.add_argument('--annotations', metavar='path', type=str, help='path to COCO style annotations',
                              required=True)
    parser_train.add_argument('--images', metavar='path', type=str, help='path to images', default='.')
    parser_train.add_argument('--backbone', action='store', type=str, nargs='+', help='backbone model (or list of)',
                              default=['ResNet50FPN'])
    parser_train.add_argument('--classes', metavar='num', type=int, help='number of classes', default=80)
    parser_train.add_argument('--batch', metavar='size', type=int, help='batch size', default=2 * devcount)
    parser_train.add_argument('--resize', metavar='scale', type=int, help='resize to given size', default=800)
    parser_train.add_argument('--max-size', metavar='max', type=int, help='maximum resizing size', default=1333)
    parser_train.add_argument('--jitter', metavar='min max', type=int, nargs=2, help='jitter size within range',
                              default=[640, 1024])
    parser_train.add_argument('--iters', metavar='number', type=int, help='number of iterations to train for',
                              default=90000)
    parser_train.add_argument('--milestones', action='store', type=int, nargs='*',
                              help='list of iteration indices where learning rate decays', default=[60000, 80000])
    parser_train.add_argument('--schedule', metavar='scale', type=float,
                              help='scale schedule (affecting iters and milestones)', default=1)
    parser_train.add_argument('--full-precision', help='train in full precision', action='store_true')
    parser_train.add_argument('--lr', metavar='value', help='learning rate', type=float, default=0.01)
    parser_train.add_argument('--warmup', metavar='iterations', help='numer of warmup iterations', type=int,
                              default=1000)
    parser_train.add_argument('--gamma', metavar='value', type=float,
                              help='multiplicative factor of learning rate decay', default=0.1)
    parser_train.add_argument('--override', help='override model', action='store_true')
    parser_train.add_argument('--val-annotations', metavar='path', type=str,
                              help='path to COCO style validation annotations')
    parser_train.add_argument('--val-images', metavar='path', type=str, help='path to validation images')
    parser_train.add_argument('--post-metrics', metavar='url', type=str, help='post metrics to specified url')
    parser_train.add_argument('--fine-tune', metavar='path', type=str, help='fine tune a pretrained model')
    parser_train.add_argument('--logdir', metavar='logdir', type=str, help='directory where to write logs')
    parser_train.add_argument('--val-iters', metavar='number', type=int,
                              help='number of iterations between each validation', default=8000)
    parser_train.add_argument('--augment-rotate', help='use four-fold rotational augmentation', action='store_true')
    parser_train.add_argument('--augment-free-rotate', type=float, metavar='value value', nargs=2, default=[0, 0],
                              help='rotate images by an arbitrary angle, between min and max (in degrees)')
    parser_train.add_argument('--augment-brightness', metavar='value', type=float,
                              help='adjust the brightness of the image.', default=0.002)
    parser_train.add_argument('--augment-contrast', metavar='value', type=float,
                              help='adjust the contrast of the image.', default=0.002)
    parser_train.add_argument('--augment-hue', metavar='value', type=float,
                              help='adjust the hue of the image.', default=0.0002)
    parser_train.add_argument('--augment-saturation', metavar='value', type=float,
                              help='adjust the saturation of the image.', default=0.002)
    parser_train.add_argument('--optim', metavar='value', type=str, help='optimizer selection (<adam>, <adamW> or <SGD>)',
                              default='SGD')
    parser_train.add_argument('--lr-scheduler', metavar='value', type=str, help='lr scheduler selection (<plateau> or <lambda>)',
                              default='lambda')
    parser_train.add_argument('--regularization-l2', metavar='value', type=float, help='L2 regularization for optim',
                              default=0.0001)
    parser_train.add_argument('--rotated-bbox', help='detect rotated bounding boxes [x, y, w, h, theta]',
                              action='store_true')
    parser_train.add_argument('--anchor-ious', metavar='value value', type=float, nargs=2,
                              help='anchor/bbox overlap threshold', default=[0.4, 0.5])
    parser_train.add_argument('--absolute-angle', help='regress absolute angle (rather than -45 to 45 degrees.',
                              action='store_true')
    parser_train.add_argument('--score-thr', help='score threshold for validation', type=float, default=0.3)
    parser_train.add_argument('--shuffle', help='shuffle the dataloader during training', action='store_true')

    parser_infer = subparsers.add_parser('infer', help='run inference')
    parser_infer.add_argument('model', type=str, help='path to model')
    parser_infer.add_argument('--images', metavar='path', type=str, help='path to images', default='.')
    parser_infer.add_argument('--annotations', metavar='annotations', type=str,
                              help='evaluate using provided annotations')
    parser_infer.add_argument('--output', metavar='file', type=str, nargs='+',
                              help='save detections to specified JSON file(s)', default=['detections.json'])
    parser_infer.add_argument('--batch', metavar='size', type=int, help='batch size', default=2 * devcount)
    parser_infer.add_argument('--resize', metavar='scale', type=int, help='resize to given size', default=800)
    parser_infer.add_argument('--max-size', metavar='max', type=int, help='maximum resizing size', default=1333)
    parser_infer.add_argument('--full-precision', help='inference in full precision', action='store_true')
    parser_infer.add_argument('--rotated-bbox', help='inference using a rotated bounding box model',
                              action='store_true')
    parser_infer.add_argument('--score-thr', help='score threshold for inference', type=float, default=0.3)

    parser_video = subparsers.add_parser('video', help='perform model inference on videos <.mp4>')
    parser_video.add_argument('model', type=str, help='path to model')
    parser_video.add_argument('--source', metavar='path', type=str, help='path to one video(with VIDEO EXTENSION), or parent directory to multiple videos', default='.')
    parser_video.add_argument('--output', metavar='file', type=str, nargs='+',
                              help='save detections to specified JSON file(s)', default=['detections.json'])
    parser_video.add_argument('--batch', metavar='size', type=int, help='batch size', default=2 * devcount)
    parser_video.add_argument('--resize', metavar='scale', type=int, help='resize to given size', default=800)
    parser_video.add_argument('--max-size', metavar='max', type=int, help='maximum resizing size', default=1333)
    parser_video.add_argument('--full-precision', help='inference in full precision', action='store_true')
    parser_video.add_argument('--rotated-bbox', help='inference using a rotated bounding box model',
                              action='store_true')
    parser_video.add_argument('--freq', help='frame capture frequency', type=int, default=5)
    parser_video.add_argument('--score-thr', help='score threshold for inference', type=float, default=0.3)
    parser_video.add_argument('--nms-thr', help='nms threshold for inference', type=float, default=0.1)
    parser_video.add_argument('--iou-thr', help='iou threshold for inference', type=float, default=0.7)
    parser_video.add_argument('--show-box', help='choose if show bounding box in videos', action='store_true')
    parser_video.add_argument('--show-track', help='choose if show tracklets in videos', action='store_true')
    parser_video.add_argument('--save-details', help='choose if save tracklet details as csv', action='store_true')

    return parser.parse_args(args)


def load_model(args, verbose=False):
    if args.command != 'train' and not os.path.isfile(args.model):
        raise RuntimeError('Model file {} does not exist!'.format(args.model))

    model = None
    state = {}
    _, ext = os.path.splitext(args.model)

    if args.command == 'train' and (not os.path.exists(args.model) or args.override):
        if verbose: print('Initializing model...')
        model = Model(backbones=args.backbone, classes=args.classes, rotated_bbox=args.rotated_bbox,
                      anchor_ious=args.anchor_ious)
        model.initialize(args.fine_tune)
        # Freeze unused params from training
        model.freeze_unused_params()
        if verbose: print(model)

    elif ext == '.pth' or ext == '.torch':
        if verbose: print('Loading model from {}...'.format(os.path.basename(args.model)))
        model, state = Model.load(filename=args.model, rotated_bbox=args.rotated_bbox)
        # Freeze unused params from training
        model.freeze_unused_params()
        if verbose: print(model)

    elif (args.command == 'infer' or args.command == 'video'):
        model = None

    else:
        raise RuntimeError('Invalid model format "{}"!'.format(ext))

    state['path'] = args.model
    return model, state


def worker(rank, args, world, model, state):
    'Per-device distributed worker'

    if torch.cuda.is_available():
        os.environ.update({
            'MASTER_PORT': args.master.split(':')[-1],
            'MASTER_ADDR': ':'.join(args.master.split(':')[:-1]),
            'WORLD_SIZE': str(world),
            'RANK': str(rank),
            'CUDA_DEVICE': str(rank)
        })

        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

        if (args.command != 'export') and (args.batch % world != 0):
            raise RuntimeError('Batch size should be a multiple of the number of GPUs')

    if model and model.angles is not None:
        args.rotated_bbox = True

    if args.command == 'train':
        train.train(model, state, args.images, args.annotations,
                    args.val_images or args.images, args.val_annotations, args.resize, args.max_size, args.jitter,
                    args.batch, int(args.iters * args.schedule), args.val_iters, args.lr, args.warmup,
                    [int(m * args.schedule) for m in args.milestones], args.gamma, rank, world=world, shuffle=args.shuffle,
                    mixed_precision=not args.full_precision,
                    metrics_url=args.post_metrics, logdir=args.logdir, verbose=(rank == 0),
                    rotate_augment=args.augment_rotate, augment_brightness=args.augment_brightness,
                    augment_contrast=args.augment_contrast, augment_hue=args.augment_hue, augment_saturation=args.augment_saturation,
                    optim=args.optim, lr_scheduler=args.lr_scheduler,
                    regularization_l2=args.regularization_l2, rotated_bbox=args.rotated_bbox, absolute_angle=args.absolute_angle, score_threshold=args.score_thr)

    elif args.command == 'infer':
        if model is None:
            if rank == 0: print('Loading CUDA engine from {}...'.format(os.path.basename(args.model)))
            # model = Engine.load(args.model)
            # FIND A WAY TO LOAD MODEL
            model, state = Model.load(filename=args.model, rotated_bbox=args.rotated_bbox)
            # model.eval()

        infer.infer(model, args.images, args.output, args.resize, args.max_size, args.batch,
                    annotations=args.annotations, mixed_precision=not args.full_precision,
                    is_master=(rank == 0), world=world,
                    verbose=(rank == 0), rotated_bbox=args.rotated_bbox, score_threshold=args.score_thr)

    elif args.command == 'video':
        if model is None:
            if rank == 0: print('Loading CUDA engine from {}...'.format(os.path.basename(args.model)))
            # model = Engine.load(args.model)
            # FIND A WAY TO LOAD MODEL
            model, state = Model.load(filename=args.model, rotated_bbox=args.rotated_bbox)
            # model.eval()

        if not os.path.isdir(args.source):
            video_infer.video_infer(model, args.source, args.output, args.resize, args.max_size, args.batch,
                        annotations=None, mixed_precision=not args.full_precision,
                        is_master=(rank == 0), world=world,
                        verbose=(rank == 0), rotated_bbox=args.rotated_bbox, freq=args.freq, score_threshold=args.score_thr,
                        nms_threshold=args.nms_thr, iou_threshold=args.iou_thr, show_box=args.show_box, show_track=args.show_track,
                        save_details=args.save_details)
        else:
            videos = [os.path.join(args.source, mp4) for mp4 in os.listdir(args.source) if mp4.endswith(".mp4")]
            video_names = [mp4[:-4] for mp4 in os.listdir(args.source) if mp4.endswith(".mp4")]
            if os.path.exists(os.path.join(args.source, "infer_jsons")):
                shutil.rmtree(os.path.join(args.source, "infer_jsons"))
            os.mkdir(os.path.join(args.source, "infer_jsons"))
            json_names = [[os.path.join(args.source, "infer_jsons", f"infer_{name}.json")] for name in video_names]
            for video, json_name in zip(videos, json_names):
                video_infer.video_infer(model, video, json_name, args.resize, args.max_size, args.batch,
                            annotations=None, mixed_precision=not args.full_precision,
                            is_master=(rank == 0), world=world,
                            verbose=(rank == 0), rotated_bbox=args.rotated_bbox, freq=args.freq, score_threshold=args.score_thr,
                            nms_threshold=args.nms_thr, iou_threshold=args.iou_thr, show_box=args.show_box, show_track=args.show_track,
                            save_details=args.save_details)


def main(args=None):
    'Entry point for the command'

    args = parse(args or sys.argv[1:])

    model, state = load_model(args, verbose=True)
    if model: model.share_memory()

    world = torch.cuda.device_count()
    if world <= 1:
        worker(0, args, 1, model, state)
    else:
        torch.multiprocessing.spawn(worker, args=(args, world, model, state), nprocs=world)


if __name__ == '__main__':
    main()
