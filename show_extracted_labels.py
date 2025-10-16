#!/usr/bin/env python3
"""
显示每个content_id抽取到的具体标签内容
"""

import json
import os
from pathlib import Path
from datetime import datetime

def show_extracted_labels():
    """显示所有抽取的标签详细内容"""
    temp_dir = Path("temp_outputs")
    
    if not temp_dir.exists():
        print("temp_outputs目录不存在")
        return
    
    json_files = list(temp_dir.glob("*.json"))
    
    if not json_files:
        print("没有找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个抽取结果文件\n")
    print("=" * 80)
    
    # 按时间排序，显示最新的结果
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    for i, json_file in enumerate(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取基本信息
            content_id = data.get('request', {}).get('content_id', 'Unknown')
            model_name = data.get('response', {}).get('model_name', 'Unknown')
            tokens_used = data.get('response', {}).get('tokens_used', 0)
            processing_time = data.get('response', {}).get('processing_time', 0)
            
            # 获取文件修改时间
            file_time = datetime.fromtimestamp(json_file.stat().st_mtime)
            
            print(f"\n📄 文件 {i+1}: {json_file.name}")
            print(f"🆔 Content ID: {content_id}")
            print(f"🤖 Model: {model_name}")
            print(f"⏱️  Processing Time: {processing_time:.2f}s")
            print(f"🎯 Tokens Used: {tokens_used}")
            print(f"📅 Generated: {file_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 检查是否成功解析
            parsed_labels = data.get('response', {}).get('parsed_labels', {})
            raw_content = data.get('response', {}).get('raw_content', '')
            
            if not parsed_labels or all(not v or (isinstance(v, list) and len(v) == 0) for k, v in parsed_labels.items() if k != 'content_id'):
                print("❌ 标签解析失败或为空")
                if raw_content:
                    print(f"📝 Raw Content Length: {len(raw_content)} characters")
                    if not raw_content.strip().endswith('}'):
                        print("⚠️  Raw content appears to be truncated")
                print("-" * 80)
                continue
            
            print("\n✅ 成功抽取的标签:")
            
            # 显示各类标签
            categories = [
                ('talking_angles', '💬 话题角度'),
                ('scenarios', '🎬 使用场景'),
                ('skin_types', '🧴 肌肤类型'),
                ('skin_concerns', '😰 肌肤问题'),
                ('product_categories', '🛍️  产品类别'),
                ('ingredients', '🧪 成分'),
                ('benefits', '✨ 功效'),
                ('brands', '🏷️  品牌'),
                ('stance', '👍 情感立场'),
                ('compliance_flags', '⚠️  合规标记'),
                ('quality_flags', '🔍 质量标记')
            ]
            
            for key, display_name in categories:
                labels = parsed_labels.get(key, [])
                
                if key == 'stance' and labels:
                    # stance是单个对象，不是列表
                    print(f"\n{display_name}:")
                    print(f"  • {labels['label']} (置信度: {labels['confidence']:.2f})")
                    print(f"    证据: \"{labels['evidence']['text']}\" (来源: {labels['evidence']['source']})")
                elif isinstance(labels, list) and labels:
                    print(f"\n{display_name}: ({len(labels)} 个)")
                    for label in labels:
                        if isinstance(label, dict):
                            if 'raw' in label:  # brands格式
                                print(f"  • {label['raw']} (置信度: {label['confidence']:.2f})")
                            else:  # 其他标签格式
                                print(f"  • {label['label']} (置信度: {label['confidence']:.2f})")
                            
                            if 'evidence' in label:
                                evidence = label['evidence']
                                print(f"    证据: \"{evidence['text']}\" (来源: {evidence['source']})")
            
            # 显示notes
            notes = parsed_labels.get('notes')
            if notes:
                print(f"\n📝 备注: {notes}")
            
            print("-" * 80)
            
        except Exception as e:
            print(f"❌ 读取文件 {json_file.name} 失败: {e}")
            print("-" * 80)
    
    print(f"\n📊 总结: 共处理了 {len(json_files)} 个文件")

if __name__ == "__main__":
    show_extracted_labels()