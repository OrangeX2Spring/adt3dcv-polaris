import cv2
import numpy as np
import pandas as pd
import os


def save_frames_every_n(video_path, output_dir, interval=20,
                         crop_top=48, crop_bottom=176, crop_left=0, crop_right=224):
    if not os.path.isfile(video_path):
        print(f"⚠️ 视频不存在，跳过: {video_path}")
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return False

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(output_dir, video_name)
    os.makedirs(save_dir, exist_ok=True)

    frame_idx = 0
    saved_count = 0
    last_frame = None

    # ✅ 关键：全程顺序 read()，不使用 cap.set() 跳帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        last_frame = frame  # 持续更新，循环结束时即为最后一帧

        if frame_idx % interval == 0:
            crop = frame[crop_top:crop_bottom, crop_left:crop_right]
            out_path = os.path.join(save_dir, f"{video_name}_frame{frame_idx:05d}.jpg")
            cv2.imwrite(out_path, crop)
            saved_count += 1

        frame_idx += 1

    cap.release()

    # 补存最后一帧
    if last_frame is not None and (frame_idx - 1) % interval != 0:
        crop = last_frame[crop_top:crop_bottom, crop_left:crop_right]
        out_path = os.path.join(save_dir, f"{video_name}_frame{frame_idx-1:05d}_last.jpg")
        cv2.imwrite(out_path, crop)
        saved_count += 1

    print(f"✅ {video_path} 完成，共保存 {saved_count} 张 -> {save_dir}")
    return True


def process_csv(csv_path, video_dir, output_dir,
                 episode_col="episode", success_col="success",
                 filename_pattern="episode_{episode}.mp4",
                 interval=20):
    """
    读取 CSV，筛选 success=True 的行，根据 episode 拼出 mp4 文件名并抽帧

    filename_pattern 支持两种写法：
      - "episode_{episode}.mp4"      -> 不补零，如 episode_5.mp4
      - "episode_{episode:02d}.mp4"  -> 补零两位，如 episode_05.mp4
    """
    df = pd.read_csv(csv_path)

    # 兼容 success 列是布尔值 或 字符串 "True"/"False"
    if df[success_col].dtype == object:
        mask = df[success_col].astype(str).str.strip().str.lower() == "true"
    else:
        mask = df[success_col] == True  # noqa: E712

    success_df = df[mask]
    print(f"共找到 {len(success_df)} 条 success=True 的记录")

    os.makedirs(output_dir, exist_ok=True)

    for _, row in success_df.iterrows():
        episode = row[episode_col]
        # 尝试转成 int，方便补零格式化；转不了就保持原样
        try:
            episode_val = int(episode)
        except (ValueError, TypeError):
            episode_val = episode

        video_filename = filename_pattern.format(episode=episode_val)
        video_path = os.path.join(video_dir, video_filename)

        save_frames_every_n(video_path, output_dir, interval=interval)


# --- 使用示例 ---
if __name__ == "__main__":
    csv_file = "/workspace/polaris/runs/pi05/DROID-FoodBussing/eval_results.csv"
    video_dir = "/workspace/polaris/runs/pi05/DROID-FoodBussing"
    output_dir = "/workspace/polaris/runs/test/downsampled_frames"

    process_csv(
        csv_path=csv_file,
        video_dir=video_dir,
        output_dir=output_dir,
        episode_col="episode",              # 👈 改成 CSV 中 episode 列的实际列名
        success_col="success",              # 👈 改成 CSV 中 success 列的实际列名
        filename_pattern="episode_{episode}.mp4",  # 👈 如果文件名是补零的（如 episode_05.mp4），改成 "episode_{episode:02d}.mp4"
        interval=20,
    )