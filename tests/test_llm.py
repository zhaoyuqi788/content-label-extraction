"""LLM客户端测试模块

测试LLM连接和基本功能。
"""

import unittest
import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.llm_client import (
    LLMClientManager, 
    DashScopeClient, 
    OpenAICompatibleClient,
    create_llm_client
)
from src.schemas import LLMExtractionRequest, ContentLabels
from src.utils import load_config


class TestLLMConnection(unittest.TestCase):
    """测试LLM连接"""
    
    def setUp(self):
        """设置测试环境"""
        # 加载配置
        config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
        self.config = load_config(str(config_path))
        
        # 测试配置
        self.test_config = {
            'provider': 'dashscope',
            'model_name': 'qwen-plus',
            'api_key_env': 'DASHSCOPE_API_KEY',
            'temperature': 0.2,
            'top_p': 0.9,
            'max_output_tokens': 1024,
            'parallel_requests': 8,
            'rate_limit_qps': 2,
            'retry_attempts': 3,
            'retry_delay': 1.0
        }
    
    def test_create_dashscope_client(self):
        """测试创建DashScope客户端"""
        client = DashScopeClient(self.test_config)
        
        self.assertEqual(client.model_name, 'qwen-plus')
        self.assertEqual(client.temperature, 0.2)
        self.assertEqual(client.base_url, "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    def test_create_openai_client(self):
        """测试创建OpenAI兼容客户端"""
        openai_config = self.test_config.copy()
        openai_config['provider'] = 'openai_compatible'
        openai_config['base_url'] = 'https://api.openai.com/v1'
        
        client = OpenAICompatibleClient(openai_config)
        
        self.assertEqual(client.model_name, 'qwen-plus')
        self.assertEqual(client.base_url, 'https://api.openai.com/v1')
    
    def test_create_llm_client_manager(self):
        """测试创建LLM客户端管理器"""
        manager = LLMClientManager(self.test_config)
        
        self.assertIsInstance(manager.client, DashScopeClient)
        self.assertEqual(manager.client.model_name, 'qwen-plus')
    
    def test_create_llm_client_convenience_function(self):
        """测试便捷创建函数"""
        manager = create_llm_client(self.test_config)
        
        self.assertIsInstance(manager, LLMClientManager)
        self.assertIsInstance(manager.client, DashScopeClient)
    
    def test_unsupported_provider(self):
        """测试不支持的提供商"""
        invalid_config = self.test_config.copy()
        invalid_config['provider'] = 'invalid_provider'
        
        with self.assertRaises(ValueError) as context:
            LLMClientManager(invalid_config)
        
        self.assertIn('不支持的LLM提供商', str(context.exception))
    
    @patch('src.llm_client.get_api_key')
    def test_api_key_loading(self, mock_get_api_key):
        """测试API密钥加载"""
        mock_get_api_key.return_value = 'test-api-key'
        
        client = DashScopeClient(self.test_config)
        
        mock_get_api_key.assert_called_with('DASHSCOPE_API_KEY')
        self.assertEqual(client.api_key, 'test-api-key')


class TestLLMClientMock(unittest.TestCase):
    """使用Mock测试LLM客户端功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.test_config = {
            'provider': 'dashscope',
            'model_name': 'qwen-plus',
            'api_key_env': 'DASHSCOPE_API_KEY',
            'temperature': 0.2,
            'top_p': 0.9,
            'max_output_tokens': 1024,
            'parallel_requests': 8,
            'rate_limit_qps': 2,
            'retry_attempts': 3,
            'retry_delay': 1.0
        }
    
    @patch('src.llm_client.get_api_key')
    @patch('httpx.AsyncClient')
    def test_dashscope_api_call_success(self, mock_client_class, mock_get_api_key):
        """测试DashScope API调用成功"""
        # Mock API密钥
        mock_get_api_key.return_value = 'test-api-key'
        
        # Mock HTTP响应
        mock_response = Mock()
        mock_response.json.return_value = {
            'output': {
                'text': '{"brands": [{"label": "兰蔻", "confidence": 0.9}]}'
            },
            'usage': {
                'total_tokens': 150
            }
        }
        mock_response.raise_for_status.return_value = None
        
        # Mock HTTP客户端
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client
        
        # 测试API调用
        async def test_call():
            client = DashScopeClient(self.test_config)
            messages = [{"role": "user", "content": "测试内容"}]
            response = await client.generate_response(messages)
            
            self.assertIn('output', response)
            self.assertEqual(response['usage']['total_tokens'], 150)
        
        # 运行异步测试
        asyncio.run(test_call())
    
    @patch('src.llm_client.get_api_key')
    @patch('httpx.AsyncClient')
    def test_openai_api_call_success(self, mock_client_class, mock_get_api_key):
        """测试OpenAI兼容API调用成功"""
        # Mock API密钥
        mock_get_api_key.return_value = 'test-api-key'
        
        # Mock HTTP响应
        mock_response = Mock()
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': '{"brands": [{"label": "兰蔻", "confidence": 0.9}]}'
                }
            }],
            'usage': {
                'total_tokens': 150
            }
        }
        mock_response.raise_for_status.return_value = None
        
        # Mock HTTP客户端
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client
        
        # 测试API调用
        async def test_call():
            openai_config = self.test_config.copy()
            openai_config['provider'] = 'openai_compatible'
            openai_config['base_url'] = 'https://api.openai.com/v1'
            
            client = OpenAICompatibleClient(openai_config)
            messages = [{"role": "user", "content": "测试内容"}]
            response = await client.generate_response(messages)
            
            self.assertIn('choices', response)
            self.assertEqual(response['usage']['total_tokens'], 150)
        
        # 运行异步测试
        asyncio.run(test_call())
    
    def test_token_usage_extraction(self):
        """测试token使用量提取"""
        # 测试DashScope格式
        dashscope_response = {
            'usage': {'total_tokens': 150}
        }
        client = DashScopeClient(self.test_config)
        tokens = client.get_token_usage(dashscope_response)
        self.assertEqual(tokens, 150)
        
        # 测试OpenAI格式
        openai_response = {
            'usage': {'total_tokens': 200}
        }
        openai_config = self.test_config.copy()
        openai_config['provider'] = 'openai_compatible'
        openai_client = OpenAICompatibleClient(openai_config)
        tokens = openai_client.get_token_usage(openai_response)
        self.assertEqual(tokens, 200)
    
    @patch('src.llm_client.get_api_key')
    def test_extract_labels_integration(self, mock_get_api_key):
        """测试标签提取集成功能"""
        mock_get_api_key.return_value = 'test-api-key'
        
        # Mock LLM客户端的generate_response方法
        with patch.object(DashScopeClient, 'generate_response') as mock_generate:
            mock_generate.return_value = {
                'output': {
                    'text': '{"brands": [{"label": "兰蔻", "confidence": 0.9}]}'
                },
                'usage': {
                    'total_tokens': 150
                }
            }
            
            async def test_extraction():
                manager = LLMClientManager(self.test_config)
                
                request = LLMExtractionRequest(
                    content_id="test_001",
                    text="兰蔻小黑瓶精华液效果很好",
                    source_mapping={"兰蔻小黑瓶精华液效果很好": "body"}
                )
                
                response = await manager.extract_labels(request)
                
                self.assertEqual(response.content_id, "test_001")
                self.assertEqual(response.tokens_used, 150)
                self.assertIsInstance(response.extracted_labels, ContentLabels)
                self.assertGreater(response.processing_time, 0)
            
            # 运行异步测试
            asyncio.run(test_extraction())


class TestLLMRealConnection(unittest.TestCase):
    """测试真实LLM连接（需要有效API密钥）"""
    
    def setUp(self):
        """设置测试环境"""
        # 检查是否有API密钥环境变量
        self.has_dashscope_key = os.getenv('DASHSCOPE_API_KEY') is not None
        self.has_openai_key = os.getenv('OPENAI_API_KEY') is not None
        
        self.test_config = {
            'provider': 'dashscope',
            'model_name': 'qwen-plus',
            'api_key_env': 'DASHSCOPE_API_KEY',
            'temperature': 0.2,
            'top_p': 0.9,
            'max_output_tokens': 100,  # 减少token使用
            'parallel_requests': 1,
            'rate_limit_qps': 1,
            'retry_attempts': 2,
            'retry_delay': 1.0
        }
    
    @unittest.skipUnless(os.getenv('DASHSCOPE_API_KEY'), "需要DASHSCOPE_API_KEY环境变量")
    def test_real_dashscope_connection(self):
        """测试真实DashScope连接"""
        async def test_connection():
            try:
                manager = LLMClientManager(self.test_config)
                
                request = LLMExtractionRequest(
                    content_id="test_real_001",
                    text="兰蔻小黑瓶",
                    source_mapping={"兰蔻小黑瓶": "body"}
                )
                
                response = await manager.extract_labels(request)
                
                # 验证响应
                self.assertEqual(response.content_id, "test_real_001")
                self.assertIsInstance(response.extracted_labels, ContentLabels)
                self.assertGreater(response.tokens_used, 0)
                self.assertGreater(response.processing_time, 0)
                
                print(f"✅ DashScope连接测试成功!")
                print(f"   - 使用token: {response.tokens_used}")
                print(f"   - 处理时间: {response.processing_time:.3f}秒")
                print(f"   - 模型: {response.model_name}")
                
            except Exception as e:
                self.fail(f"DashScope连接测试失败: {e}")
        
        # 运行异步测试
        asyncio.run(test_connection())
    
    @unittest.skipUnless(os.getenv('OPENAI_API_KEY'), "需要OPENAI_API_KEY环境变量")
    def test_real_openai_connection(self):
        """测试真实OpenAI连接"""
        async def test_connection():
            try:
                openai_config = self.test_config.copy()
                openai_config.update({
                    'provider': 'openai_compatible',
                    'api_key_env': 'OPENAI_API_KEY',
                    'base_url': 'https://api.openai.com/v1',
                    'model_name': 'gpt-3.5-turbo'
                })
                
                manager = LLMClientManager(openai_config)
                
                request = LLMExtractionRequest(
                    content_id="test_real_002",
                    text="兰蔻小黑瓶",
                    source_mapping={"兰蔻小黑瓶": "body"}
                )
                
                response = await manager.extract_labels(request)
                
                # 验证响应
                self.assertEqual(response.content_id, "test_real_002")
                self.assertIsInstance(response.extracted_labels, ContentLabels)
                self.assertGreater(response.tokens_used, 0)
                self.assertGreater(response.processing_time, 0)
                
                print(f"✅ OpenAI连接测试成功!")
                print(f"   - 使用token: {response.tokens_used}")
                print(f"   - 处理时间: {response.processing_time:.3f}秒")
                print(f"   - 模型: {response.model_name}")
                
            except Exception as e:
                self.fail(f"OpenAI连接测试失败: {e}")
        
        # 运行异步测试
        asyncio.run(test_connection())
    
    def test_connection_status_info(self):
        """显示连接状态信息"""
        print("\n=== LLM连接状态检查 ===")
        print(f"DASHSCOPE_API_KEY: {'✅ 已设置' if self.has_dashscope_key else '❌ 未设置'}")
        print(f"OPENAI_API_KEY: {'✅ 已设置' if self.has_openai_key else '❌ 未设置'}")
        
        if not self.has_dashscope_key and not self.has_openai_key:
            print("\n⚠️  提示: 设置环境变量以启用真实连接测试:")
            print("   export DASHSCOPE_API_KEY='your-key'")
            print("   export OPENAI_API_KEY='your-key'")
        
        print("========================\n")


if __name__ == '__main__':
    unittest.main(verbosity=2)