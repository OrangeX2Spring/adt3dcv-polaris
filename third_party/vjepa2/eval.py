import os
from logging import getLogger
from math import ceil
from torch.nn import functional as F
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from decord import VideoReader, cpu
from app.vjepa_behavior1k.transforms import make_transforms
from app.vjepa_behavior1k.utils import init_video_model, load_checkpoint, load_pretrained
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
device="cuda:0"
parquet_path = "/mnt/4T/ADT3DCV2026/data/test/data/task-0049/episode_00490450.parquet"
df = pd.read_parquet(parquet_path)


timestamps = df['timestamp'].values

target_timestamps = np.array([680, 680.25])
indices_state = np.searchsorted(timestamps, target_timestamps)
full_state = df['observation.state'].iloc[indices_state].values
full_state = np.stack(full_state, axis=0)
states = np.concatenate([
            full_state[:, 253:256],
            full_state[:, 236:240],
            full_state[:, 158:165],
            ((full_state[:, 193] + full_state[:, 194]) * 10).reshape(-1, 1),
            full_state[:, 197:204],
            ((full_state[:, 232] + full_state[:, 233]) * 10).reshape(-1, 1),
        ], axis=-1)
states = torch.from_numpy(states).to(device)
actions = (states[1:] - states[:-1])

# fps4_timestamps = np.arange(timestamps[0], timestamps[-1], 0.25)
# indices_4fps = np.searchsorted(timestamps, fps4_timestamps)
# all_state = df['observation.state'].iloc[indices_4fps].values
# all_state = np.stack(all_state, axis=0)
# diff_states  = np.concatenate([
#             all_state[:, 253:256],
#             all_state[:, 236:240],
#             all_state[:, 158:165],
#             ((all_state[:, 193] + all_state[:, 194]) * 10).reshape(-1, 1),
#             all_state[:, 197:204],
#             ((all_state[:, 232] + all_state[:, 233]) * 10).reshape(-1, 1),
#         ], axis=-1)
# diff_states = torch.from_numpy(diff_states).to(device)
# all_actions = diff_states[1:] - diff_states[:-1]
# action_std = torch.std(all_actions, dim=0)
# action_mean = torch.mean(all_actions, dim=0)
    
# print("\n📊 统计完成！")
# print(f"Action 均值 (shape [23]):\n{action_mean}")
# print(f"Action 标准差 (shape [23]):\n{action_std}")

transform = make_transforms(
    random_horizontal_flip=False,
    random_resize_aspect_ratio=[0.75, 1.35],
    random_resize_scale=[1.777, 1.777],
    reprob=0.0,
    auto_augment=False,
    motion_shift=False,
    crop_size=256,
)
vpath = "/mnt/4T/ADT3DCV2026/data/test/videos/task-0049/observation.images.rgb.head/episode_00490450.mp4"
vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
vfps = vr.get_avg_fps() 
vlen = len(vr)

indices_video = ((target_timestamps - timestamps[0]) * vfps).astype(np.int64)
buffer = vr.get_batch(indices_video).asnumpy()
buffer = transform(buffer).unsqueeze(0).to(device)


encoder, predictor = init_video_model(
        uniform_power=True,
        device="cuda:0",
        patch_size=16,
        max_num_frames=512,
        tubelet_size=2,
        model_name="vit_giant_xformers",
        crop_size=256,
        pred_depth=24,
        pred_num_heads=16,
        pred_embed_dim=1024,
        action_embed_dim=23,
        pred_is_frame_causal=True,
        use_extrinsics=False,
        use_sdpa=True,
        use_silu=False,
        use_pred_silu=False,
        wide_silu=False,
        use_rope=True,
        use_activation_checkpointing=True,
    )

encoder, predictor, _, _, _ = load_checkpoint(
        r_path="./outputs/behavior1k/vitg16-256px-8f/latest.pt",
        encoder=encoder,
        predictor=predictor,
        target_encoder=None
    )

def forward_target(c, normalize_reps=True):
    B, C, T, H, W = c.size()
    c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    h = encoder(c)
    h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
    if normalize_reps:
        h = F.layer_norm(h, (h.size(-1),))
    return h
def calculate_energy_normalized_mse(z_pred, z_target):
    """
    z_pred: 预测的未来潜在特征, shape: [B, D] 或 [D]
    z_target: 真实的目标潜在特征, shape: [B, D] 或 [D]
    """
    # 1. 确保是浮点数并展平（如果是时空特征 [C, H, W] 则展平为 [C*H*W]）
    z_pred_flat = z_pred.reshape(z_pred.shape[0], -1) if z_pred.dim() > 1 else z_pred.flatten().unsqueeze(0)
    z_target_flat = z_target.reshape(z_target.shape[0], -1) if z_target.dim() > 1 else z_target.flatten().unsqueeze(0)
    
    # 2. 对特征进行 L2 归一化
    z_pred_norm = F.normalize(z_pred_flat, p=2, dim=-1)
    z_target_norm = F.normalize(z_target_flat, p=2, dim=-1)
    
    # 3. 计算均方误差 (MSE) 作为能量
    energy = torch.mean((z_pred_norm - z_target_norm) ** 2, dim=-1)
    
    return energy.item() if energy.numel() == 1 else energy


def plot_curves(alpha_range, all_curves, save_path="vjepa_energy_landscape.png"):
    """
    alpha_range: 扰动系数的一维数组, shape: [num_steps] (例如从 -0.5 到 0.5)
    all_curves: 包含多条曲线能量值的列表或数组, shape: [num_directions, num_steps]
    save_path: 图片保存路径
    """
    # 1. 设置科研风格的画图主题和大小
    plt.style.use('seaborn-v0_8-whitegrid') # 使用美观的网格主题
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300) # 高清分辨率
    
    num_directions = len(all_curves)
    
    # 2. 引入渐变色板（比如从深蓝到紫色的渐变，看起来非常 Meta 风格）
    colors = plt.cm.plasma(np.linspace(0.1, 0.8, num_directions))
    
    # 3. 循环画出每一条随机方向的切片曲线
    for i, energies in enumerate(all_curves):
        ax.plot(alpha_range, energies, 
                color=colors[i], 
                linewidth=2.0, 
                alpha=0.7, 
                label=f'Direction {i+1}')
        
    # 4. 核心灵魂：在 X=0 处画一条垂直虚线，代表 Ground Truth 动作位置
    ax.axvline(x=0.0, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Ground Truth Action')
    
    # 5. 美化坐标轴和标签
    ax.set_title('V-JEPA 2-AC 23D Action Energy Landscape (Line Sweep)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(r'Action Perturbation Scale ($\alpha$)', fontsize=12, labelpad=10)
    ax.set_ylabel('V-JEPA Energy (Prediction Loss)', fontsize=12, labelpad=10)
    
    # 6. 微调坐标轴范围和刻度，让画面更好看
    ax.set_xlim(alpha_range.min(), alpha_range.max())
    # 动态调整 Y 轴，给顶部留出 15% 的空隙放图例
    y_min, y_max = np.min(all_curves), np.max(all_curves)
    ax.set_ylim(y_min - (y_max - y_min)*0.05, y_max + (y_max - y_min)*0.15)
    
    # 7. 优化图例展示（只显示前几条和 GT，防止图例太多炸掉）
    handles, labels = ax.get_legend_handles_labels()
    # 抽样显示图例：显示第1条、最后一条和红色虚线
    if num_directions > 2:
        selected_handles = [handles[0], handles[-2], handles[-1]]
        selected_labels = [labels[0], labels[-2], labels[-1]]
    else:
        selected_handles, selected_labels = handles, labels
        
    ax.legend(selected_handles, selected_labels, loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    
    # 8. 紧凑布局并保存/显示
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# def step_predictor(_z, _a, _s):
#         _z = predictor(_z, _a, _s)[:, -tokens_per_frame:]
#         if normalize_reps:
#             _z = F.layer_norm(_z, (_z.size(-1),))
#         _s = compute_new_pose(_s[:, -1:], _a[:, -1:])
#         return _z, _s
h = forward_target(buffer)
z0 = h[0, :256]
z1 = h[0, 256:]


def visualize_frame_features(
    h,
    h_hat,
    images,
    tokens_per_frame=256,
    method="norm",   # "norm" or "pca"
    alpha=0.5
):
    """
    h: [1, 2*tokens_per_frame, D]
    images: np_clips[0] -> [T, H, W, 3]
    """

    assert h.dim() == 3
    B, N, D = h.shape
    assert B == 1

    z0 = h[0, :tokens_per_frame]
    z1 = h[0, tokens_per_frame:]

    # -----------------------------
    # 1. feature reduction
    # -----------------------------
    if method == "norm":
        feat0 = torch.norm(z0, dim=-1).detach().numpy()
        feat1 = torch.norm(z1, dim=-1).detach().numpy()
        feat_pred = torch.norm(h_hat, dim=-1).cpu().detach().numpy()
    elif method == "pca":
        z = torch.cat([z0, z1, h_hat.squeeze(0)], dim=0).cpu().detach().numpy()
        pca = PCA(n_components=1)
        feat = pca.fit_transform(z).squeeze()
        feat0 = feat[:tokens_per_frame]
        feat1 = feat[tokens_per_frame:2*tokens_per_frame]
        feat_pred = feat[2*tokens_per_frame:]

    else:
        raise ValueError("method must be 'norm' or 'pca'")

    # -----------------------------
    # 2. reshape to token grid
    # -----------------------------
    grid_size = int(np.sqrt(tokens_per_frame))
    assert grid_size * grid_size == tokens_per_frame, "tokens_per_frame must be square"

    feat0_map = feat0.reshape(grid_size, grid_size)
    feat1_map = feat1.reshape(grid_size, grid_size)
    feat_pred_map = feat_pred.reshape(grid_size, grid_size)
    # -----------------------------
    # 3. upsample to image size
    # -----------------------------
    H, W = images.shape[1], images.shape[2]

    feat0_up = cv2.resize(feat0_map, (W, H))
    feat1_up = cv2.resize(feat1_map, (W, H))
    feat_pred_up = cv2.resize(feat_pred_map, (W,H))
    # normalize for visualization
    def norm(x):
        x = x - x.min()
        x = x / (x.max() + 1e-8)
        return x

    feat0_up = norm(feat0_up)
    feat1_up = norm(feat1_up)
    feat_pred_up = norm(feat_pred_up)
    # -----------------------------
    # 4. plot overlay
    # -----------------------------
    plt.figure(figsize=(15, 4.5))

    plt.subplot(1, 3, 1)
    plt.imshow(images[0])
    plt.imshow(feat0_up, cmap="jet", alpha=alpha)
    plt.title("Frame 0 feature map")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(images[1])
    plt.imshow(feat1_up, cmap="jet", alpha=alpha)
    plt.title("Frame 1 feature map")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(images[1])
    plt.imshow(feat_pred_up, cmap="jet", alpha=alpha)
    plt.title("Frame 1 Feature (PREDICTED)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return feat0_up, feat1_up

np_buffer = buffer.permute(0,2,3,4,1).cpu().numpy()
h_hat = predictor(z0.unsqueeze(0), actions.unsqueeze(0).float(), states[0].unsqueeze(0).unsqueeze(0).float())
feat0, feat1 = visualize_frame_features(
    h=h,
    h_hat=h_hat,
    images=np_buffer[0],
    tokens_per_frame=256,
    method="pca",   # 或 "pca"
    alpha=0.5
)
diff = torch.norm(h_hat - z1, dim=-1)

diff_map = diff.reshape(16,16)

plt.imshow(diff_map.cpu().detach().numpy())
plt.colorbar()
plt.title("Token Difference")
plt.show()










num_directions = 10  # 随机生成 10 个不同的高维扰动方向
num_steps = 50       # 每个方向上切片采样的精细度
alpha_range = torch.linspace(-0.5, 0.5, num_steps, device=device) # 扰动系数从 -0.5 到 0.5

all_curves = []

# 3. 开始切片扫描
for i in range(num_directions):
    # 随机生成一个 23 维的方向向量，并进行单位化（L2归一化）
    V = torch.randn(size=(23,), device=device)
    V = V / torch.linalg.norm(V, ord=2)
    
    energies = []
    for alpha in alpha_range:
        # 在 GT 动作上加上微小的扰动，组合成一个新的 23维 动作
        A_perturbed = actions + alpha * V   # shape: [23]
        A_perturbed_batched = A_perturbed.unsqueeze(0).float()
        with torch.no_grad():
            # 将扰动后的动作喂给 VJEPA 的 Predictor 预测未来
            z_next_pred = predictor(z0.unsqueeze(0), A_perturbed_batched, states[0].unsqueeze(0).unsqueeze(0).float())
            
            # 计算预测状态与目标状态之间的能量（距离/损失）
            energy = calculate_energy_normalized_mse(z_next_pred, z1.unsqueeze(0))
        energies.append(energy)
        
    all_curves.append(energies)
alpha_range_cpu = alpha_range.cpu().numpy()
all_curves_cpu = [
    c.cpu().numpy() if isinstance(c, torch.Tensor) else np.array(c) 
    for c in all_curves
]
# 4. 画图
# X轴是 alpha_range，Y轴是 energies，把 10 条线画在同一张图上
plot_curves(alpha_range_cpu, all_curves_cpu)
