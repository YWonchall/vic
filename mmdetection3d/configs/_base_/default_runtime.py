checkpoint_config = dict(interval=1)
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/workspace/vic-competition/mmdetection3d/work-dirs/exam-c/veh-car/train/'
load_from = None
resume_from = "/workspace/vic-competition/dair-v2x/configs/vic3d/late-fusion-pointcloud/pointpillars/vic3d_latefusion_inf_pointpillars_596784ad6127866fcfb286301757c949.pth"
workflow = [('train', 1)]
