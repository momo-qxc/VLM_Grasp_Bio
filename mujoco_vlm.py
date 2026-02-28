"""
VLM 机器人抓取 — CLI 入口点（无 GUI）。
运行方式: conda activate vlm_graspnet_RRT && python mujoco_vlm.py
"""
import os, sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))

from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv
from task_executor import TaskExecutor


def _log(level, msg):
    print(f"[{level}] {msg}")


def _ask(question):
    return input(f"  {question}\n  > ").strip()


if __name__ == '__main__':
    env = UR5GraspEnv()
    env.reset()

    executor = TaskExecutor(
        env, log_fn=_log, ask_fn=_ask,
        headless=False, render_callback=None,
    )

    print("\n" + "=" * 60)
    print("  VLM 机器人抓取 — CLI 模式（smart）")
    print("  输入自然语言指令，例如：把培养皿放到显微镜右边")
    print("  输入 q 退出")
    print("=" * 60)

    while True:
        user_input = input("\n请输入指令: ").strip()
        if not user_input:
            continue
        if user_input.lower() == 'q':
            break
        try:
            executor.execute_smart_task(user_input)
            print("\n[OK] 任务完成")
        except Exception as e:
            import traceback
            print(f"\n[X] 任务异常: {e}")
            traceback.print_exc()

    env.close()
