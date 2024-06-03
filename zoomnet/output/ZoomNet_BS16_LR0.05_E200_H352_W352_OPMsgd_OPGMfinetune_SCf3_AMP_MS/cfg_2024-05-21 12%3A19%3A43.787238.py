has_test = True
base_seed = 0
deterministic = True
log_interval = dict(txt=20, tensorboard=0)
load_from = ''
resume_from = ''
model_name = 'ZoomNet'
experiment_tag = ''
train = dict(
    batch_size=16,
    num_workers=2,
    use_amp=True,
    num_epochs=200,
    num_iters=30000,
    epoch_based=True,
    lr=0.05,
    optimizer=dict(
        mode='sgd',
        set_to_none=True,
        group_mode='finetune',
        cfg=dict(momentum=0.9, weight_decay=0.0005, nesterov=False)),
    grad_acc_step=1,
    sche_usebatch=True,
    scheduler=dict(
        warmup=dict(num_iters=0, initial_coef=0.01, mode='linear'),
        mode='f3',
        cfg=dict(lr_decay=0.9, min_coef=None)),
    save_num_models=1,
    ms=dict(
        enable=True,
        extra_scales=[
            0.6363636363636364, 0.7272727272727273, 0.8181818181818182,
            0.9090909090909091, 1.0
        ]),
    grad_clip=dict(enable=False, mode='value', cfg=dict()),
    ema=dict(
        enable=False, cmp_with_origin=True, force_cpu=False, decay=0.9998))
test = dict(
    batch_size=12,
    num_workers=2,
    eval_func='default_test',
    clip_range=None,
    tta=dict(
        enable=False,
        reducation='mean',
        cfg=dict(
            HorizontalFlip=dict(),
            VerticalFlip=dict(),
            Rotate90=dict(angles=[0, 90, 180, 270]),
            Scale=dict(
                scales=[0.75, 1, 1.5],
                interpolation='bilinear',
                align_corners=False),
            Add=dict(values=[0, 10, 20]),
            Multiply=dict(factors=[1, 2, 5]),
            FiveCrops=dict(crop_height=224, crop_width=224),
            Resize=dict(
                sizes=[0.75, 1, 1.5],
                original_size=224,
                interpolation='bilinear',
                align_corners=False))),
    show_bar=False)
use_custom_worker_init = False
datasets = dict(
    train=dict(
        dataset_type='msi_sod_tr',
        shape=dict(h=352, w=352),
        path=dict(
            dutstr=dict(
                root='/mnt/disk1/roger/ZoomNet/Training_dataset',
                image=dict(path='img', suffix='.jpg'),
                mask=dict(path='label_img', suffix='.png'))),
        interp_cfg=dict()),
    test=dict(
        dataset_type='msi_sod_te',
        shape=dict(h=352, w=352),
        path=dict({
            'pascal-s':
            dict(
                root=
                '/mnt/disk1/yang/BiRefNet/datasets/SOD/Private_Public_dataset',
                image=dict(path='im', suffix='.jpg'),
                mask=dict(path='gt', suffix='.png'))
        }),
        interp_cfg=dict()))
use_ddp = False
proj_root = '/mnt/disk1/roger/ZoomNet'
exp_name = 'ZoomNet_BS16_LR0.05_E200_H352_W352_OPMsgd_OPGMfinetune_SCf3_AMP_MS'
output_dir = '/mnt/disk1/roger/ZoomNet/output'
path = dict(
    output_dir='/mnt/disk1/roger/ZoomNet/output',
    pth_log=
    '/mnt/disk1/roger/ZoomNet/output/ZoomNet_BS16_LR0.05_E200_H352_W352_OPMsgd_OPGMfinetune_SCf3_AMP_MS',
    tb=
    '/mnt/disk1/roger/ZoomNet/output/ZoomNet_BS16_LR0.05_E200_H352_W352_OPMsgd_OPGMfinetune_SCf3_AMP_MS/tb',
    save=
    '/mnt/disk1/roger/ZoomNet/output/ZoomNet_BS16_LR0.05_E200_H352_W352_OPMsgd_OPGMfinetune_SCf3_AMP_MS/pre',
    pth=
    '/mnt/disk1/roger/ZoomNet/output/ZoomNet_BS16_LR0.05_E200_H352_W352_OPMsgd_OPGMfinetune_SCf3_AMP_MS/pth',
    final_full_net=
    '/mnt/disk1/roger/ZoomNet/output/ZoomNet_BS16_LR0.05_E200_H352_W352_OPMsgd_OPGMfinetune_SCf3_AMP_MS/pth/checkpoint_final.pth',
    final_state_net=
    '/mnt/disk1/roger/ZoomNet/output/ZoomNet_BS16_LR0.05_E200_H352_W352_OPMsgd_OPGMfinetune_SCf3_AMP_MS/pth/state_final.pth',
    tr_log=
    '/mnt/disk1/roger/ZoomNet/output/ZoomNet_BS16_LR0.05_E200_H352_W352_OPMsgd_OPGMfinetune_SCf3_AMP_MS/tr_2024-05-21.txt',
    te_log=
    '/mnt/disk1/roger/ZoomNet/output/ZoomNet_BS16_LR0.05_E200_H352_W352_OPMsgd_OPGMfinetune_SCf3_AMP_MS/te_2024-05-21.txt',
    trans_log=
    '/mnt/disk1/roger/ZoomNet/output/ZoomNet_BS16_LR0.05_E200_H352_W352_OPMsgd_OPGMfinetune_SCf3_AMP_MS/trans_2024-05-21.txt',
    cfg_copy=
    '/mnt/disk1/roger/ZoomNet/output/ZoomNet_BS16_LR0.05_E200_H352_W352_OPMsgd_OPGMfinetune_SCf3_AMP_MS/cfg_2024-05-21 12:19:43.787238.py',
    excel=
    '/mnt/disk1/roger/ZoomNet/output/ZoomNet_BS16_LR0.05_E200_H352_W352_OPMsgd_OPGMfinetune_SCf3_AMP_MS/results.xlsx',
    trainer_copy=
    '/mnt/disk1/roger/ZoomNet/output/ZoomNet_BS16_LR0.05_E200_H352_W352_OPMsgd_OPGMfinetune_SCf3_AMP_MS/trainer_2024-05-21 12:19:43.787242.txt'
)
