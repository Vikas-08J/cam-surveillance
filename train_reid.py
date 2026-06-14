import torch
import torchreid


def main():

    # =========================
    # DEVICE (GPU / CPU)
    # =========================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # =========================
    # DATA MANAGER
    # =========================
    datamanager = torchreid.data.ImageDataManager(
        root='reid-data',            # your dataset root
        sources='market1501',        # IMPORTANT
        targets='market1501',
        height=256,
        width=128,
        batch_size_train=32,         # good for GPU
        batch_size_test=64,
        workers=4                   # GPU parallel loading
    )

    # =========================
    # MODEL
    # =========================
    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=datamanager.num_train_pids,
        loss='softmax',
        pretrained=True
    )

    model = model.to(device)

    # =========================
    # OPTIMIZER
    # =========================
    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=0.0003
    )

    # =========================
    # SCHEDULER
    # =========================
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=20
    )

    # =========================
    # ENGINE
    # =========================
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    # =========================
    # TRAINING
    # =========================
    print("🚀 Starting training...")

    engine.run(
        save_dir='log/osnet',
        max_epoch=30,
        eval_freq=-1,     # 🔥 completely disables evaluation (fix your error)
        print_freq=10
    )

    # =========================
    # FORCE SAVE MODEL
    # =========================
    torch.save(model.state_dict(), 'log/osnet/my_model.pth')
    print("✅ Model saved at log/osnet/my_model.pth")


# =========================
# WINDOWS FIX (MANDATORY)
# =========================
if __name__ == '__main__':
    main()