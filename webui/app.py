# -*- coding: UTF-8 -*-
"""
FedTracker WebUI - 深色主题界面
"""

import argparse
import os
import os.path as osp
import sys
from datetime import datetime

import gradio as gr

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from modules.utils import (
    find_model_dirs,
    find_leaked_models,
    has_trace_data,
    get_default_output_dir,
)
from modules.generation import generate_images, save_images
from modules.tracing import simulate_client_leak, identify_owner


PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULT_DIR = osp.join(PROJECT_ROOT, "result")
LEAK_TEST_DIR = osp.join(PROJECT_ROOT, "leak_test")

DARK_CSS = """
:root {
    --bg-primary: #0f0f0f;
    --bg-secondary: #1a1a1a;
    --bg-tertiary: #232323;
    --text-primary: #ffffff;
    --text-secondary: #a0a0a0;
    --accent: #7c3aed;
}
body { background: var(--bg-primary) !important; color: var(--text-primary) !important; }
.gradio-container { max-width: 100% !important; padding: 0 !important; }
#header { background: var(--bg-secondary); padding: 20px 40px; text-align: center; border-bottom: 1px solid #333; }
#header h1 { margin: 0; font-size: 28px; background: linear-gradient(135deg, #7c3aed, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
#header p { color: var(--text-secondary); margin: 5px 0; }
"""


def create_ui():
    model_dirs = find_model_dirs(RESULT_DIR)
    default_model = model_dirs[0] if model_dirs else None

    # 过滤出有trace_data的模型
    models_with_trace = [
        m for m in model_dirs if has_trace_data(osp.join(RESULT_DIR, m))
    ]

    # 获取泄漏模型列表
    leaked_models = find_leaked_models(LEAK_TEST_DIR)

    with gr.Blocks(css=DARK_CSS, theme=gr.themes.Base()) as demo:
        # 顶部标题
        gr.HTML("""
            <div id="header">
                <h1>FedTracker 水印追踪系统</h1>
                <p>联邦学习扩散模型所有权验证与可追溯平台</p>
            </div>
        """)

        # 首页
        with gr.Row(visible=True) as home_row:
            with gr.Column():
                gr.HTML(
                    '<div style="text-align: center; padding: 40px 20px;"><h2 style="color: #fff;">欢迎使用 FedTracker</h2><p style="color: #a0a0a0;">选择功能开始使用</p></div>'
                )

                with gr.Row():
                    with gr.Column():
                        gr.HTML(
                            '<div style="background: #1a1a1a; border: 1px solid #333; border-radius: 16px; padding: 30px; text-align: center;"><div style="font-size: 40px;">🎨</div><h3 style="color: #fff;">图片生成</h3><p style="color: #a0a0a0; font-size: 14px;">从训练好的模型生成图片</p></div>'
                        )
                        gen_btn = gr.Button("开始生成 →", size="lg", variant="primary")

                    with gr.Column():
                        gr.HTML(
                            '<div style="background: #1a1a1a; border: 1px solid #333; border-radius: 16px; padding: 30px; text-align: center;"><div style="font-size: 40px;">🔍</div><h3 style="color: #fff;">泄漏模拟</h3><p style="color: #a0a0a0; font-size: 14px;">模拟客户端模型泄漏</p></div>'
                        )
                        leak_btn = gr.Button("开始模拟 →", size="lg", variant="primary")

                    with gr.Column():
                        gr.HTML(
                            '<div style="background: #1a1a1a; border: 1px solid #333; border-radius: 16px; padding: 30px; text-align: center;"><div style="font-size: 40px;">🎯</div><h3 style="color: #fff;">所有者识别</h3><p style="color: #a0a0a0; font-size: 14px;">识别泄漏模型所有者</p></div>'
                        )
                        identify_btn = gr.Button(
                            "开始识别 →", size="lg", variant="primary"
                        )

        # 图片生成页面
        with gr.Row(visible=False) as gen_row:
            with gr.Column(scale=1, min_width=280):
                gr.HTML('<h3 style="color: #fff;">⚙️ 模型设置</h3>')
                model_dropdown = gr.Dropdown(
                    choices=model_dirs if model_dirs else ["暂无可用模型"],
                    value=default_model,
                    label="选择模型",
                    interactive=True,
                    allow_custom_value=False,
                )

                with gr.Accordion("高级设置", open=False):
                    class_label = gr.Number(value=0, label="类标签")
                    num_images = gr.Number(value=4, label="生成数量")
                    num_steps = gr.Number(value=1000, label="推理步数")
                    seed = gr.Number(value=42, label="随机种子")
                    trigger_check = gr.Checkbox(label="生成水印图片", value=False)
                    device = gr.Radio(
                        choices=["cuda", "cpu"], value="cuda", label="计算设备"
                    )

                generate_btn = gr.Button("🚀 开始生成", size="lg", variant="primary")
                back1_btn = gr.Button("← 返回首页", size="lg")

            with gr.Column(scale=2):
                gr.HTML('<h3 style="color: #fff;">📷 生成结果</h3>')
                output_gallery = gr.Gallery(
                    label="图片", show_label=False, columns=4, rows=2, height=350
                )
                status_output = gr.Textbox(lines=2, interactive=False, label="状态")
                output_path = gr.Textbox(
                    value="保存路径：./webui/static/outputs/",
                    interactive=False,
                    show_label=False,
                )

        # 泄漏模拟页面
        with gr.Row(visible=False) as leak_row:
            with gr.Column(scale=1, min_width=280):
                gr.HTML('<h3 style="color: #fff;">⚙️ 模拟配置</h3>')
                leak_model = gr.Dropdown(
                    choices=models_with_trace
                    if models_with_trace
                    else ["暂无可用模型"],
                    value=models_with_trace[0] if models_with_trace else None,
                    label="源模型",
                    interactive=True,
                    allow_custom_value=False,
                )
                client_idx = gr.Number(
                    value=0,
                    label="客户端索引",
                    interactive=True,
                    precision=0,
                )
                leak_output = gr.Textbox(
                    value="leaked_model_client_0.pth", label="输出文件名"
                )

                simulate_btn = gr.Button("🎭 开始模拟", size="lg", variant="primary")
                back2_btn = gr.Button("← 返回首页", size="lg")

            with gr.Column(scale=2):
                gr.HTML('<h3 style="color: #fff;">📋 模拟结果</h3>')
                leak_result = gr.Textbox(lines=10, interactive=False, label="结果")

        # 所有者识别页面
        with gr.Row(visible=False) as identify_row:
            with gr.Column(scale=1, min_width=280):
                gr.HTML('<h3 style="color: #fff;">⚙️ 识别配置</h3>')
                leaked_model = gr.Dropdown(
                    choices=leaked_models if leaked_models else ["暂无泄漏模型"],
                    value=leaked_models[0] if leaked_models else "暂无泄漏模型",
                    label="泄漏模型",
                    interactive=True,
                    allow_custom_value=False,
                )
                refresh_btn = gr.Button("🔄 刷新列表", size="sm")
                source_model = gr.Dropdown(
                    choices=models_with_trace
                    if models_with_trace
                    else ["暂无可用模型"],
                    value=models_with_trace[0] if models_with_trace else "暂无可用模型",
                    label="源模型",
                    interactive=True,
                    allow_custom_value=False,
                )

                identify2_btn = gr.Button("🎯 开始识别", size="lg", variant="primary")
                back3_btn = gr.Button("← 返回首页", size="lg")

            with gr.Column(scale=2):
                gr.HTML('<h3 style="color: #fff;">🏆 识别结果</h3>')
                identify_result = gr.Textbox(lines=8, interactive=False, label="结果")
                confidence = gr.Number(label="置信度", interactive=False)

        # 页面切换
        def show_generation():
            return [
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            ]

        def show_leak():
            return [
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
            ]

        def show_identify():
            return [
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
            ]

        def show_home():
            return [
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            ]

        gen_btn.click(
            show_generation,
            inputs=[],
            outputs=[home_row, gen_row, leak_row, identify_row],
        )
        leak_btn.click(
            show_leak, inputs=[], outputs=[home_row, gen_row, leak_row, identify_row]
        )
        identify_btn.click(
            show_identify,
            inputs=[],
            outputs=[home_row, gen_row, leak_row, identify_row],
        )

        back1_btn.click(
            show_home, inputs=[], outputs=[home_row, gen_row, leak_row, identify_row]
        )
        back2_btn.click(
            show_home, inputs=[], outputs=[home_row, gen_row, leak_row, identify_row]
        )
        back3_btn.click(
            show_home, inputs=[], outputs=[home_row, gen_row, leak_row, identify_row]
        )

        def refresh_leaked():
            return gr.Dropdown(choices=find_leaked_models(LEAK_TEST_DIR))

        refresh_btn.click(refresh_leaked, [], [leaked_model])

        def do_generate(model_name, class_lbl, num_img, steps, sd, trigger, dev):
            if not model_name:
                return None, "❌ 请选择模型", ""

            model_dir = osp.join(RESULT_DIR, model_name)
            output_base = get_default_output_dir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = osp.join(output_base, f"gen_{timestamp}")

            trigger_class = None
            if trigger:
                args_path = osp.join(model_dir, "args.txt")
                if osp.exists(args_path):
                    with open(args_path, "r") as f:
                        for line in f:
                            if "trigger_class" in line and "=" in line:
                                try:
                                    trigger_class = eval(line.split("=", 1)[1].strip())
                                except:
                                    trigger_class = int(line.split("=", 1)[1].strip())
                                break
                if trigger_class is None:
                    trigger_class = num_img

            images, error = generate_images(
                model_dir,
                int(class_lbl),
                int(num_img),
                int(steps),
                int(sd),
                trigger_class if trigger else None,
                dev,
            )

            if error:
                return None, f"❌ 失败：{error}", ""

            if images:
                os.makedirs(output_dir, exist_ok=True)
                paths = save_images(images, output_dir)
                return paths, f"✅ 成功生成 {len(images)} 张", f"保存至：{output_dir}"

            return None, "❌ 未生成图片", ""

        def update_output_name(client_idx_val):
            if client_idx_val is not None:
                return gr.update(value=f"leaked_model_client_{int(client_idx_val)}.pth")
            return gr.update(value="leaked_model_client_0.pth")

        def do_simulate(model, client, output_name):
            if not model:
                return "❌ 请选择源模型"
            if client is None:
                return "❌ 请选择客户端"

            os.makedirs(LEAK_TEST_DIR, exist_ok=True)
            checkpoint = osp.join(RESULT_DIR, model, "model_final.pth")
            trace_dir = osp.join(RESULT_DIR, model, "trace_data")
            output = osp.join(LEAK_TEST_DIR, output_name)

            result, error = simulate_client_leak(
                checkpoint, trace_dir, int(client), output
            )

            if error:
                return f"❌ 失败：{error}"

            return f"✅ 成功！保存至：{output}"

        def do_identify(leaked, source):
            if not leaked:
                return "❌ 请选择泄漏模型", 0
            if not source:
                return "❌ 请选择源模型", 0

            leaked_path = osp.join(LEAK_TEST_DIR, leaked)
            trace_dir = osp.join(RESULT_DIR, source, "trace_data")
            source_model_dir = osp.join(RESULT_DIR, source)

            client, conf, error = identify_owner(
                leaked_path, trace_dir, source_model_dir
            )

            if error:
                return f"❌ 失败：{error}", 0

            return (
                f"✅ 识别成功！\n\n客户端ID：{client}\n置信度：{conf:.2%}\n\n该模型属于客户端 {client}",
                float(conf),
            )

        generate_btn.click(
            do_generate,
            [
                model_dropdown,
                class_label,
                num_images,
                num_steps,
                seed,
                trigger_check,
                device,
            ],
            [output_gallery, status_output, output_path],
        )
        simulate_btn.click(
            do_simulate, [leak_model, client_idx, leak_output], [leak_result]
        )
        client_idx.change(update_output_name, [client_idx], [leak_output])
        identify2_btn.click(
            do_identify, [leaked_model, source_model], [identify_result, confidence]
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="FedTracker WebUI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = create_ui()
    print(f"\n{'=' * 60}")
    print("FedTracker WebUI - 深色主题")
    print(f"{'=' * 60}")
    print(f"本地访问：http://localhost:{args.port}")
    if args.share:
        print("公网链接将自动生成...")
    print(f"{'=' * 60}\n")

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
