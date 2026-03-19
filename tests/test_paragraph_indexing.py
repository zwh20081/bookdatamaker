"""测试 PageManager 段落索引功能"""

from pathlib import Path
import tempfile
import pytest
from bookdatamaker.utils import PageManager


def test_paragraph_indexing():
    """测试段落索引构建"""
    # 创建测试文本（带段落）
    test_content = """[PAGE_001]
第一段第一行
第一段第二行

第二段第一行
第二段第二行
第二段第三行

[PAGE_002]
第三段第一行

第四段第一行
第四段第二行"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        temp_file = Path(f.name)

    try:
        # 加载文档
        pm = PageManager.from_combined_file(temp_file)

        # 验证基本统计
        assert pm.get_total_pages() == 2
        assert pm.total_lines > 0
        assert pm.total_paragraphs >= 4  # 至少4个段落

        # 验证段落号获取
        para_0 = pm.get_paragraph_number(0)  # 第一行应该在第一段
        assert para_0 is not None
        assert para_0 >= 0

        # 验证段落信息
        para_info = pm.get_paragraph_info(0)
        assert para_info is not None
        assert "start_line" in para_info
        assert "end_line" in para_info
        assert "content" in para_info
        assert "page_number" in para_info

        print(f"✓ 总段落数: {pm.total_paragraphs}")
        print(f"✓ 第一段信息: {para_info}")

    finally:
        temp_file.unlink()


def test_unified_response_format():
    """测试统一的响应格式"""
    test_content = """[PAGE_001]
第一段内容

第二段内容"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        temp_file = Path(f.name)

    try:
        pm = PageManager.from_combined_file(temp_file)

        # 测试 get_line 返回格式
        line_0 = pm.get_line(0)
        assert line_0 is not None

        # 获取行的段落和页码信息
        para_num = pm.get_paragraph_number(0)
        page_num, _ = pm.line_to_page.get(0, (None, None))

        assert para_num is not None
        assert page_num is not None

        print(f"✓ 第0行: 段落{para_num}, 页码{page_num}, 内容: {line_0}")

        # 测试统计信息包含段落
        stats = pm.get_statistics()
        assert "total_paragraphs" in stats
        assert "paragraphs" in stats
        assert stats["total_paragraphs"] > 0

        print(f"✓ 统计信息包含 {stats['total_paragraphs']} 个段落")

    finally:
        temp_file.unlink()


def test_get_paragraph_info():
    """测试获取段落详细信息"""
    test_content = """[PAGE_001]
段落一行一
段落一行二

段落二行一

[PAGE_002]
段落三行一
段落三行二"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        temp_file = Path(f.name)

    try:
        pm = PageManager.from_combined_file(temp_file)

        # 获取第一个段落的信息
        para_0_info = pm.get_paragraph_info(0)
        assert para_0_info is not None
        assert para_0_info["line_count"] >= 2  # 至少2行
        assert "段落一" in para_0_info["content"]

        print(f"✓ 段落0: 第{para_0_info['start_line']}-{para_0_info['end_line']}行")
        print(f"  页码: {para_0_info['page_number']}")
        print(f"  行数: {para_0_info['line_count']}")

        # 获取第二个段落
        para_1_info = pm.get_paragraph_info(1)
        assert para_1_info is not None
        assert "段落二" in para_1_info["content"]

        print(f"✓ 段落1: 第{para_1_info['start_line']}-{para_1_info['end_line']}行")

    finally:
        temp_file.unlink()


def test_search_with_paragraph():
    """测试搜索结果包含段落号"""
    test_content = """[PAGE_001]
深度学习是机器学习的分支

神经网络是基础

[PAGE_002]
深度学习应用广泛"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        temp_file = Path(f.name)

    try:
        pm = PageManager.from_combined_file(temp_file)

        # 搜索"深度学习"
        results = pm.search_text("深度学习")
        assert len(results) >= 2  # 应该找到至少2个结果

        # 为每个结果添加段落号（模拟 MCP 服务器的行为）
        for result in results:
            para_num = pm.get_paragraph_number(result["line_number"])
            result["paragraph_number"] = para_num

            print(f"✓ 找到 '深度学习' 在:")
            print(f"  行号: {result['line_number']}")
            print(f"  段落号: {result['paragraph_number']}")
            print(f"  页码: {result['page_number']}")
            print(f"  内容: {result['content']}")

    finally:
        temp_file.unlink()


if __name__ == "__main__":
    print("=== 测试 PageManager 段落索引功能 ===\n")

    try:
        test_paragraph_indexing()
        print("\n✓ 段落索引测试通过\n")
    except AssertionError as e:
        print(f"\n✗ 段落索引测试失败: {e}\n")

    try:
        test_unified_response_format()
        print("\n✓ 统一响应格式测试通过\n")
    except AssertionError as e:
        print(f"\n✗ 统一响应格式测试失败: {e}\n")

    try:
        test_get_paragraph_info()
        print("\n✓ 获取段落信息测试通过\n")
    except AssertionError as e:
        print(f"\n✗ 获取段落信息测试失败: {e}\n")

    try:
        test_search_with_paragraph()
        print("\n✓ 搜索包含段落号测试通过\n")
    except AssertionError as e:
        print(f"\n✗ 搜索包含段落号测试失败: {e}\n")

    print("=== 所有测试完成 ===")
