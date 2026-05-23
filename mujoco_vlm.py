"""
VLM 机器人抓取 — CLI 入口点（无 GUI）。
运行方式: conda activate vlm_graspnet_RRT && python mujoco_vlm.py
"""
import os, sys
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))

from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv
from task_executor import TaskExecutor
from config import get_config_manager


def _log(level, msg):
    print(f"[{level}] {msg}")


def _ask(question):
    return input(f"  {question}\n  > ").strip()


def _show_models():
    """显示所有可用模型"""
    mgr = get_config_manager()
    models = mgr.get_all_models()
    current = mgr.get_active_model_name()

    print("\n" + "=" * 60)
    print("可用模型列表:")
    print("=" * 60)
    for i, model_name in enumerate(models.keys(), 1):
        marker = "✓" if model_name == current else " "
        print(f"  [{i}] {marker} {model_name}")
    print("=" * 60)
    return list(models.keys())


def _switch_model():
    """切换模型"""
    model_list = _show_models()
    mgr = get_config_manager()

    while True:
        choice = input("\n请选择模型编号 (或按 Enter 取消): ").strip()
        if not choice:
            print("已取消")
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(model_list):
                model_name = model_list[idx]
                success = mgr.set_active_model(model_name)
                if success:
                    print(f"\n✅ 已切换到模型: {model_name}")
                    return
                else:
                    print(f"\n❌ 切换失败")
                    return
            else:
                print("无效的编号，请重新输入")
        except ValueError:
            print("请输入数字")


def _parse_args():
    parser = argparse.ArgumentParser(description="VLM Grasp CLI")
    parser.add_argument(
        "--scene",
        choices=["normal", "obstacle"],
        default=None,
        help="启动场景: normal=普通场景, obstacle=障碍场景",
    )
    parser.add_argument(
        "--planner",
        choices=["default", "rrt"],
        default=None,
        help="规划模式: default=默认算法, rrt=RRT避障算法",
    )
    return parser.parse_args()


def _pick_scene(scene_arg):
    if scene_arg in ("normal", "obstacle"):
        return scene_arg
    while True:
        s = input("请选择场景 [1=普通场景, 2=障碍场景] (默认1): ").strip()
        if s in ("", "1"):
            return "normal"
        if s == "2":
            return "obstacle"
        print("输入无效，请输入 1 或 2。")


def _pick_planner(planner_arg):
    if planner_arg in ("default", "rrt"):
        return planner_arg
    while True:
        s = input("请选择规划 [1=默认算法, 2=RRT避障算法] (默认1): ").strip()
        if s in ("", "1"):
            return "default"
        if s == "2":
            return "rrt"
        print("输入无效，请输入 1 或 2。")


def _resolve_scene_config(scene_key):
    if scene_key == "obstacle":
        scene_name = "障碍场景"
        scene_file = os.path.join(
            ROOT_DIR, "manipulator_grasp", "assets", "scenes", "scene_obstacle.xml"
        )
    else:
        scene_name = "普通场景"
        scene_file = os.path.join(
            ROOT_DIR, "manipulator_grasp", "assets", "scenes", "scene.xml"
        )
    return scene_name, scene_file


def _resolve_planner_config(planner_key):
    if planner_key == "rrt":
        return "RRT避障算法"
    return "默认算法"


if __name__ == '__main__':
    args = _parse_args()
    scene_key = _pick_scene(args.scene)
    planner_key = _pick_planner(args.planner)
    scene_name, scene_file = _resolve_scene_config(scene_key)
    planner_mode = _resolve_planner_config(planner_key)

    env = UR5GraspEnv(scene_file=scene_file)
    env.reset()

    executor = TaskExecutor(
        env, log_fn=_log, ask_fn=_ask,
        headless=False, render_callback=None,
    )

    # 显示当前模型
    mgr = get_config_manager()
    current_model = mgr.get_active_model_name()

    print("\n" + "=" * 60)
    print("  VLM 机器人抓取 — CLI 模式（smart）")
    print(f"  当前场景: {scene_name} | 规划模式: {planner_mode}")
    print(f"  当前模型: {current_model}")
    print("=" * 60)
    print("  输入自然语言指令，例如：把培养皿放到显微镜右边")
    print("  输入 /model 切换模型")
    print("  输入 q 退出")
    print("=" * 60)

    while True:
        user_input = input("\n请输入指令: ").strip()
        if not user_input:
            continue
        if user_input.lower() == 'q':
            break

        # 处理命令
        if user_input.startswith('/'):
            cmd = user_input[1:].lower()
            if cmd == 'model':
                _switch_model()
                current_model = mgr.get_active_model_name()
                print(f"\n当前模型: {current_model}")
            elif cmd == 'help':
                print("\n可用命令:")
                print("  /model  - 切换视觉模型")
                print("  /help   - 显示帮助")
                print("  q       - 退出程序")
            else:
                print(f"\n未知命令: {user_input}")
                print("输入 /help 查看可用命令")
            continue

        # 执行任务
        try:
            executor.execute_smart_task(
                user_input,
                planner_mode=planner_mode,
                scene_name=scene_name,
            )
            print("\n[OK] 任务完成")
        except Exception as e:
            import traceback
            print(f"\n[X] 任务异常: {e}")
            traceback.print_exc()

    env.close()
