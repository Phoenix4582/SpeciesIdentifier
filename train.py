from statistics import mean
from math import isfinite
import torch
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
# from apex import amp, optimizers
# from apex.parallel import DistributedDataParallel as ADDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from backbones.layers import convert_fixedbn_model

from data import DataIterator, RotatedDataIterator
# from .dali import DaliDataIterator
from utils import ignore_sigint, post_metrics, Profiler
from infer import infer


def train(model, state, path, annotations, val_path, val_annotations, resize, max_size, jitter, batch_size, iterations,
          val_iterations, lr, warmup, milestones, gamma, rank=0, world=1, shuffle=True, mixed_precision=True, verbose=True, metrics_url=None, logdir=None, rotate_augment=False, augment_brightness=0.0,
          augment_contrast=0.0, augment_hue=0.0, augment_saturation=0.0, optim='adam', lr_scheduler='lambda', regularization_l2=0.0001, rotated_bbox=False,
          absolute_angle=False, score_threshold=0.3):
    'Train the model on the given dataset'

    # Prepare model
    nn_model = model
    stride = model.stride

    model = convert_fixedbn_model(model)
    if torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last).cuda()

    # Setup optimizer and schedule
    if optim == 'adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=regularization_l2)
    elif optim == 'adamW':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=regularization_l2)
    else:
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=regularization_l2, momentum=0.9)


    is_master = rank==0

    if world > 1:
        model = DDP(model, device_ids=[rank])
    model.train()

    if 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])

    if lr_scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=int(0.25*iterations), threshold=1e-5, verbose=True)
    else:
        def schedule(train_iter):
            if warmup and train_iter <= warmup:
                return 0.9 * train_iter / warmup + 0.1
            return gamma ** len([m for m in milestones if m <= train_iter])

        scheduler = LambdaLR(optimizer, schedule)

    if 'scheduler' in state:
        scheduler.load_state_dict(state['scheduler'])

    # Prepare dataset
    if verbose: print('Preparing dataset...')
    if rotated_bbox:
        data_iterator = RotatedDataIterator(path, jitter, max_size, batch_size, shuffle, stride,
                                            world, annotations, training=True, rotate_augment=rotate_augment,
                                            augment_brightness=augment_brightness,
                                            augment_contrast=augment_contrast, augment_hue=augment_hue,
                                            augment_saturation=augment_saturation, absolute_angle=absolute_angle)
    else:
        data_iterator = DataIterator(
            path, jitter, max_size, batch_size, shuffle, stride,
            world, annotations, training=True, rotate_augment=rotate_augment, augment_brightness=augment_brightness,
            augment_contrast=augment_contrast, augment_hue=augment_hue, augment_saturation=augment_saturation)
    if verbose: print(data_iterator)

    if verbose:
        print('    device: {} {}'.format(
            world, 'cpu' if not torch.cuda.is_available() else 'GPU' if world == 1 else 'GPUs'))
        print('     batch: {}, precision: {}'.format(batch_size, 'mixed' if mixed_precision else 'full'))
        print(' BBOX type:', 'rotated' if rotated_bbox else 'axis aligned')
        print('Training model for {} iterations...'.format(iterations))

    # Create TensorBoard writer
    if is_master and logdir is not None:
        from torch.utils.tensorboard import SummaryWriter
        if verbose:
            print('Writing TensorBoard logs to: {}'.format(logdir))
        writer = SummaryWriter(log_dir=logdir)

    scaler = GradScaler(enabled=mixed_precision)
    profiler = Profiler(['train', 'fw', 'bw'])
    iteration = state.get('iteration', 0)
    while iteration < iterations:
        cls_losses, box_losses = [], []
        for i, (data, target) in enumerate(data_iterator):
            if iteration>=iterations:
                break

            # Forward pass
            profiler.start('fw')

            optimizer.zero_grad()
            with autocast(enabled=mixed_precision):
                cls_loss, box_loss = model([data.contiguous(memory_format=torch.channels_last), target])
            del data
            profiler.stop('fw')

            # Backward pass
            profiler.start('bw')
            # if with_apex:
            #     with amp.scale_loss(cls_loss + box_loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            #     optimizer.step()
            # else:
            scaler.scale(cls_loss + box_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if lr_scheduler == 'plateau':
                scheduler.step(cls_loss)
            else:
                scheduler.step()

            # Reduce all losses
            cls_loss, box_loss = cls_loss.mean().clone(), box_loss.mean().clone()
            if world > 1:
                torch.distributed.all_reduce(cls_loss)
                torch.distributed.all_reduce(box_loss)
                cls_loss /= world
                box_loss /= world
            if is_master:
                cls_losses.append(cls_loss)
                box_losses.append(box_loss)

            if is_master and not isfinite(cls_loss + box_loss):
                raise RuntimeError('Loss is diverging!\n{}'.format(
                    'Try lowering the learning rate.'))

            del cls_loss, box_loss
            profiler.stop('bw')

            iteration += 1
            profiler.bump('train')
            if is_master and (profiler.totals['train'] > 60 or iteration == iterations):
                focal_loss = torch.stack(list(cls_losses)).mean().item()
                box_loss = torch.stack(list(box_losses)).mean().item()
                learning_rate = optimizer.param_groups[0]['lr']
                if verbose:
                    msg = '[{:{len}}/{}]'.format(iteration, iterations, len=len(str(iterations)))
                    msg += ' focal loss: {:.5f}'.format(focal_loss)
                    msg += ', box loss: {:.5f}'.format(box_loss)
                    msg += ', {:.3f}s/{}-batch'.format(profiler.means['train'], batch_size)
                    msg += ' (fw: {:.3f}s, bw: {:.3f}s)'.format(profiler.means['fw'], profiler.means['bw'])
                    msg += ', {:.1f} im/s'.format(batch_size / profiler.means['train'])
                    msg += ', lr: {:.2g}'.format(learning_rate)
                    print(msg, flush=True)

                if is_master and logdir is not None:
                    writer.add_scalar('focal_loss', focal_loss, iteration)
                    writer.add_scalar('box_loss', box_loss, iteration)
                    writer.add_scalar('learning_rate', learning_rate, iteration)
                    del box_loss, focal_loss

                if metrics_url:
                    post_metrics(metrics_url, {
                        'focal loss': mean(cls_losses),
                        'box loss': mean(box_losses),
                        'im_s': batch_size / profiler.means['train'],
                        'lr': learning_rate
                    })

                # Save model weights
                state.update({
                    'iteration': iteration,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                })
                with ignore_sigint():
                    nn_model.save(state)

                profiler.reset()
                del cls_losses[:], box_losses[:]

            if val_annotations and (iteration == iterations or iteration % val_iterations == 0):
                stats = infer(model, val_path, None, resize, max_size, batch_size, annotations=val_annotations,
                            mixed_precision=mixed_precision, is_master=is_master, world=world, is_validation=True, verbose=False, rotated_bbox=rotated_bbox, score_threshold=score_threshold)
                model.train()
                if is_master and logdir is not None and stats is not None:
                    writer.add_scalar(
                        'Validation_Precision/mAP', stats[0], iteration)
                    writer.add_scalar(
                        'Validation_Precision/mAP@0.50IoU', stats[1], iteration)
                    writer.add_scalar(
                        'Validation_Precision/mAP@0.75IoU', stats[2], iteration)
                    writer.add_scalar(
                        'Validation_Recall/mAR', stats[3], iteration)
                    writer.add_scalar(
                        'Validation_Recall/mAR@0.50IoU', stats[4], iteration)
                    writer.add_scalar(
                        'Validation_Recall/mAR@0.75IoU', stats[5], iteration)

            if (iteration==iterations and not rotated_bbox) or (iteration>iterations and rotated_bbox):
                break

    if is_master and logdir is not None:
        writer.close()
