"""
Enhanced Training Script with Comprehensive Utilities

This is an example of how to integrate the new utilities into the training pipeline.
It demonstrates:
- Configuration management
- Enhanced logging
- Metrics tracking
- Visualization
- Automatic report generation
"""

import os
import sys
import torch
import numpy as np
import datetime
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

# Import new utilities
from utils import (
    ExperimentConfig,
    TrainingConfig,
    create_experiment_config,
    save_experiment_config,
    ClassificationMetrics,
    MetricsTracker,
    TrainingVisualizer,
    create_experiment_logger,
    ReportGenerator
)

import provider
import importlib


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Enhanced Training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampling')
    
    # New parameters for enhanced features
    parser.add_argument('--use_tensorboard', action='store_true', default=False, help='use TensorBoard logging')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='use Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='pointnet', help='W&B project name')
    parser.add_argument('--generate_report', action='store_true', default=True, help='generate experiment report')
    
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def test(model, loader, num_class, metrics_tracker):
    """
    Enhanced test function with metrics tracking
    """
    classifier = model.eval()
    metrics = ClassificationMetrics(num_class)
    
    for j, (points, target) in tqdm(enumerate(loader), total=len(loader), desc="Testing"):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()
        
        points = points.transpose(2, 1)
        with torch.no_grad():
            pred, _ = classifier(points)
        
        metrics.update(pred, target)
    
    return metrics.compute()


def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    
    '''CONFIGURATION'''
    config = create_experiment_config(args, task='classification')
    save_experiment_config(config, str(exp_dir))
    
    '''LOGGING'''
    exp_logger, tb_logger, wb_logger = create_experiment_logger(
        log_dir=str(exp_dir),
        experiment_name=args.model,
        use_tensorboard=args.use_tensorboard,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        config=config.config
    )
    
    exp_logger.log_hyperparameters(vars(args))
    
    '''DATA LOADING'''
    exp_logger.info('Loading dataset...')
    data_path = 'data/modelnet40_normal_resampled/'
    
    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                                   shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, 
                                                  shuffle=False, num_workers=10)
    
    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    
    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)
    
    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()
    
    # Log model summary
    exp_logger.log_model_summary(classifier, (args.batch_size, 3 if not args.use_normals else 6, args.num_point))
    
    '''OPTIMIZER'''
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    
    '''METRICS TRACKING'''
    metrics_tracker = MetricsTracker()
    train_metrics = ClassificationMetrics(num_class)
    
    '''VISUALIZATION'''
    visualizer = TrainingVisualizer(str(exp_dir / 'visualizations'))
    
    best_instance_acc = 0.0
    best_class_acc = 0.0
    
    '''TRAINING'''
    exp_logger.info('Starting training...')
    
    for epoch in range(args.epoch):
        exp_logger.info(f'Epoch {epoch + 1}/{args.epoch}')
        
        # Reset training metrics
        train_metrics.reset()
        classifier = classifier.train()
        scheduler.step()
        
        # Training loop
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), desc="Training"):
            optimizer.zero_grad()
            
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()
            
            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            
            train_metrics.update(pred, target)
            
            loss.backward()
            optimizer.step()
        
        # Compute training metrics
        train_results = train_metrics.compute()
        train_results['loss'] = loss.item()
        train_results['learning_rate'] = scheduler.get_last_lr()[0]
        
        # Log training metrics
        exp_logger.log_metrics(epoch + 1, train_results, phase='train')
        if tb_logger:
            for key, value in train_results.items():
                tb_logger.log_scalar(f'train/{key}', value, epoch + 1)
        if wb_logger:
            wb_logger.log({f'train/{k}': v for k, v in train_results.items()}, step=epoch + 1)
        
        metrics_tracker.add_train_metrics(epoch + 1, train_results)
        
        # Testing
        with torch.no_grad():
            test_results = test(classifier.eval(), testDataLoader, num_class, metrics_tracker)
        
        # Log test metrics
        exp_logger.log_metrics(epoch + 1, test_results, phase='val')
        if tb_logger:
            for key, value in test_results.items():
                tb_logger.log_scalar(f'val/{key}', value, epoch + 1)
        if wb_logger:
            wb_logger.log({f'val/{k}': v for k, v in test_results.items()}, step=epoch + 1)
        
        metrics_tracker.add_val_metrics(epoch + 1, test_results)
        
        # Save best model
        instance_acc = test_results['overall_accuracy']
        class_acc = test_results['average_class_accuracy']
        
        if instance_acc >= best_instance_acc:
            best_instance_acc = instance_acc
            best_epoch = epoch + 1
            
            savepath = str(checkpoints_dir) + '/best_model.pth'
            state = {
                'epoch': best_epoch,
                'instance_acc': instance_acc,
                'class_acc': class_acc,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            exp_logger.log_checkpoint_saved(savepath, test_results)
    
    # Save final metrics
    metrics_tracker.save(str(exp_dir / 'metrics.json'))
    
    # Generate visualizations
    exp_logger.info('Generating visualizations...')
    visualizer.plot_training_curves(
        metrics_tracker.train_metrics,
        metrics_tracker.val_metrics,
        metric_keys=['overall_accuracy', 'average_class_accuracy']
    )
    
    # Generate report
    if args.generate_report:
        exp_logger.info('Generating experiment report...')
        report_gen = ReportGenerator(str(exp_dir / 'reports'))
        experiment_data = {
            'name': args.log_dir or timestr,
            'path': str(exp_dir),
            'config': config.config,
            'metrics': metrics_tracker.train_metrics + metrics_tracker.val_metrics,
            'best_metrics': metrics_tracker.get_best_metrics()
        }
        report_gen.generate_report(experiment_data)
    
    # Close loggers
    if tb_logger:
        tb_logger.close()
    if wb_logger:
        wb_logger.finish()
    
    exp_logger.info('Training completed!')
    exp_logger.info(f'Best instance accuracy: {best_instance_acc:.4f}')
    exp_logger.info(f'Best class accuracy: {best_class_acc:.4f}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
