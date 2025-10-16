#!/usr/bin/env python3
"""
查看所有抽取的标签结果
"""

import json
import os
from pathlib import Path

def view_extracted_labels():
    """查看temp_outputs目录下所有保存的标签抽取结果"""
    temp_dir = Path("temp_outputs")
    
    if not temp_dir.exists():
        print("temp_outputs目录不存在")
        return
    
    json_files = list(temp_dir.glob("*.json"))
    
    if not json_files:
        print("没有找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件:\n")
    
    for json_file in sorted(json_files):
        print(f"=== {json_file.name} ===")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取基本信息
            content_id = data.get('request', {}).get('content_id', 'Unknown')
            model_name = data.get('response', {}).get('model_name', 'Unknown')
            tokens_used = data.get('response', {}).get('tokens_used', 0)
            processing_time = data.get('response', {}).get('processing_time', 0)
            
            print(f"Content ID: {content_id}")
            print(f"Model: {model_name}")
            print(f"Tokens: {tokens_used}")
            print(f"Processing Time: {processing_time:.2f}s")
            
            # 检查raw_content是否被截断
            raw_content = data.get('response', {}).get('raw_content', '')
            if raw_content:
                print(f"Raw Content Length: {len(raw_content)} characters")
                if not raw_content.strip().endswith('}'):
                    print("⚠️  Raw content appears to be truncated")
                else:
                    print("✅ Raw content appears complete")
            
            # 查看parsed_labels
            parsed_labels = data.get('response', {}).get('parsed_labels', {})
            if parsed_labels:
                print("\nParsed Labels:")
                for category, labels in parsed_labels.items():
                    if category == 'content_id':
                        continue
                    if isinstance(labels, list) and labels:
                        print(f"  {category}: {len(labels)} items")
                        for label in labels[:3]:  # 只显示前3个
                            if isinstance(label, dict) and 'label' in label:
                                confidence = label.get('confidence', 0)
                                print(f"    - {label['label']} (confidence: {confidence})")
                        if len(labels) > 3:
                            print(f"    ... and {len(labels) - 3} more")
                    elif labels and not isinstance(labels, list):
                        print(f"  {category}: {labels}")
                    else:
                        print(f"  {category}: empty")
            else:
                print("\n❌ No parsed labels found")
            
            print("\n" + "-" * 50 + "\n")
            
        except Exception as e:
            print(f"❌ 读取文件失败: {e}")
            print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    view_extracted_labels()