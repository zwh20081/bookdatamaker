"""演示数据集导出功能

展示如何使用 DatasetManager 和 CLI 导出命令
"""

from pathlib import Path
import tempfile
from bookdatamaker.dataset import DatasetManager


def demonstrate_dataset_manager():
    """演示 DatasetManager 的基本使用"""
    
    print("=== DatasetManager 使用示例 ===\n")
    
    # 创建临时数据库
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    print(f"创建数据库: {db_path}\n")
    
    # 使用上下文管理器
    with DatasetManager(db_path) as dm:
        # 添加一些示例数据
        print("添加数据集条目...")
        entries_data = [
            ("这本书的主题是什么？", "这本书介绍深度学习的基础知识。"),
            ("什么是神经网络？", "神经网络是由多个层次组成的计算模型。"),
            ("反向传播的作用是什么？", "反向传播用于训练神经网络，计算梯度并更新权重。"),
            ("卷积神经网络的特点？", "卷积神经网络主要用于图像处理，具有局部连接和权重共享的特性。"),
            ("什么是激活函数？", "激活函数为神经网络引入非线性，常见的有ReLU、Sigmoid等。"),
        ]
        
        for input_text, output_text in entries_data:
            entry_id = dm.add_entry(input_text, output_text)
            print(f"  ✓ 添加条目 #{entry_id}")
        
        # 查询统计
        count = dm.count_entries()
        print(f"\n总条目数: {count}\n")
        
        # 获取所有条目
        print("所有条目:")
        all_entries = dm.get_all_entries()
        for entry in all_entries:
            print(f"  ID {entry['id']}: {entry['input'][:30]}...")
        
        print()
        
        # 导出为不同格式
        temp_dir = Path(tempfile.mkdtemp())
        
        print("导出数据集...")
        
        # JSONL
        jsonl_path = temp_dir / "dataset.jsonl"
        exported = dm.export_jsonl(str(jsonl_path))
        size = jsonl_path.stat().st_size
        print(f"  ✓ JSONL: {exported} 条目, {size} 字节")
        
        # Parquet
        parquet_path = temp_dir / "dataset.parquet"
        exported = dm.export_parquet(str(parquet_path))
        size = parquet_path.stat().st_size
        print(f"  ✓ Parquet: {exported} 条目, {size} 字节")
        
        # CSV
        csv_path = temp_dir / "dataset.csv"
        exported = dm.export_csv(str(csv_path))
        size = csv_path.stat().st_size
        print(f"  ✓ CSV: {exported} 条目, {size} 字节")
        
        # JSON
        json_path = temp_dir / "dataset.json"
        exported = dm.export_json(str(json_path))
        size = json_path.stat().st_size
        print(f"  ✓ JSON: {exported} 条目, {size} 字节")
        
        print(f"\n所有文件保存在: {temp_dir}")
        
        # 显示 JSONL 内容
        print("\nJSONL 文件内容预览:")
        print("-" * 60)
        with jsonl_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if i <= 2:  # 只显示前两条
                    print(line.strip())
                else:
                    break
        print("-" * 60)
    
    # 清理
    Path(db_path).unlink()
    print(f"\n清理临时文件: {db_path}")


def show_cli_commands():
    """显示 CLI 命令示例"""
    
    print("\n\n=== CLI 命令示例 ===\n")
    
    print("1. 启动 MCP 服务器（指定数据库）:")
    print("   $ bookdatamaker mcp-server combined.txt -d dataset.db")
    print()
    
    print("2. 停止服务器后，导出数据集:")
    print()
    
    print("   导出为 JSONL（默认）:")
    print("   $ bookdatamaker export-dataset dataset.db -o output.jsonl")
    print()
    
    print("   导出为 Parquet:")
    print("   $ bookdatamaker export-dataset dataset.db -o output.parquet -f parquet")
    print()
    
    print("   导出为 CSV（包含元数据）:")
    print("   $ bookdatamaker export-dataset dataset.db -o output.csv -f csv --include-metadata")
    print()
    
    print("   导出为 JSON:")
    print("   $ bookdatamaker export-dataset dataset.db -o output.json -f json")
    print()
    
    print("3. 使用导出的数据:")
    print("""
   Python 读取 JSONL:
   ```python
   import json
   
   with open('output.jsonl', 'r') as f:
       for line in f:
           entry = json.loads(line)
           print(entry['prompt'], '→', entry['response'])
   ```
   
   Python 读取 Parquet:
   ```python
   import pandas as pd
   
   df = pd.read_parquet('output.parquet')
   print(df.head())
   ```
   
   Python 读取 CSV:
   ```python
   import pandas as pd
   
   df = pd.read_csv('output.csv')
   print(df.info())
   ```
    """)


def show_workflow():
    """显示完整工作流程"""
    
    print("\n\n=== 完整工作流程 ===\n")
    
    print("""
第一步：提取文档
    $ bookdatamaker extract book.pdf -o extracted_text
    
第二步：启动 MCP 服务器
    $ bookdatamaker mcp-server extracted_text/combined.txt -d dataset.db
    
    输出:
    Loading document from: extracted_text/combined.txt
    Dataset database: dataset.db
    ✓ Loaded 150 pages
    ✓ Total lines: 3542
    ✓ Total paragraphs: 287
    
    Starting MCP server...
    Use Ctrl+C to stop the server

第三步：LLM 通过 MCP 导航并提交数据
    [LLM 自动执行]
    - 调用 get_current_paragraph 获取内容
    - 分析并生成问答对
    - 调用 submit_dataset 提交
    - 数据实时保存到 SQLite

第四步：停止服务器
    [按 Ctrl+C]
    
第五步：导出数据集
    $ bookdatamaker export-dataset dataset.db -o output.jsonl
    
    输出:
    Loading dataset from: dataset.db
    Found 287 entries
    Exporting to: output.jsonl
    Format: JSONL
    Exported 287 entries to output.jsonl
    File size: 125.47 KB

第六步：使用数据集
    - 训练模型
    - 评估性能
    - 数据分析
    """)


def show_advantages():
    """显示 SQLite 存储的优势"""
    
    print("\n\n=== SQLite 存储的优势 ===\n")
    
    advantages = [
        ("实时保存", "每次提交立即写入数据库，避免数据丢失"),
        ("事务安全", "ACID 保证，数据一致性和完整性"),
        ("高效查询", "可以使用 SQL 查询和分析数据"),
        ("灵活导出", "支持 JSONL、Parquet、CSV、JSON 多种格式"),
        ("多次导出", "可以多次导出为不同格式，无需重新生成"),
        ("元数据支持", "自动记录创建时间，支持添加自定义元数据"),
        ("可扩展性", "支持大规模数据集，性能稳定"),
        ("标准格式", "SQLite 是广泛支持的标准格式，便于集成"),
    ]
    
    for i, (title, desc) in enumerate(advantages, 1):
        print(f"{i}. {title}")
        print(f"   {desc}")
        print()


if __name__ == "__main__":
    try:
        demonstrate_dataset_manager()
        show_cli_commands()
        show_workflow()
        show_advantages()
        
        print("\n" + "=" * 60)
        print("查看 MCP_NAVIGATION.md 获取更多详细信息")
        print("=" * 60)
        
    except ImportError as e:
        print(f"错误: {e}")
        print("请确保已安装所有依赖: pip install -r requirements.txt")
