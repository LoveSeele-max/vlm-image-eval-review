"""Open the saved local demonstration index; never calls a model."""
import webbrowser
from pathlib import Path


def main():
    private = Path(__file__).resolve().parent / "private"
    page = private / "project_navigation/index.html"
    if not page.is_file():
        page = private / "h_stage_20261109/demo.html"
    if not page.is_file():
        raise SystemExit("本机演示材料不存在。请先查看 README；代码仓库不包含原素材及私有演示证据。")
    webbrowser.open(page.as_uri())
    print("已打开本机项目演示索引；未调用模型。")


if __name__ == "__main__":
    main()
