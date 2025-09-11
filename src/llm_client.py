"""LLM客户端模块

支持DashScope和OpenAI兼容接口的大模型调用。
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

from .schemas import LLMExtractionRequest, LLMExtractionResponse, ContentLabels
from .utils import get_api_key


class BaseLLMClient(ABC):
    """LLM客户端基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('model_name', 'qwen-plus')
        self.temperature = config.get('temperature', 0.2)
        self.top_p = config.get('top_p', 0.9)
        self.max_tokens = config.get('max_output_tokens', 1024)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        
    @abstractmethod
    async def generate_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """生成响应
        
        Args:
            messages: 消息列表
            
        Returns:
            Dict: 响应结果
        """
        pass
    
    @abstractmethod
    def get_token_usage(self, response: Dict[str, Any]) -> int:
        """获取token使用量
        
        Args:
            response: API响应
            
        Returns:
            int: token使用量
        """
        pass


class DashScopeClient(BaseLLMClient):
    """DashScope客户端"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = get_api_key(config.get('api_key_env', 'DASHSCOPE_API_KEY'))
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """调用DashScope API
        
        Args:
            messages: 消息列表
            
        Returns:
            Dict: API响应
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "input": {
                "messages": messages
            },
            "parameters": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "result_format": "message"
            }
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error(f"DashScope API调用失败: {e}")
                raise
    
    def get_token_usage(self, response: Dict[str, Any]) -> int:
        """获取DashScope token使用量"""
        usage = response.get('usage', {})
        return usage.get('total_tokens', 0)


class OpenAICompatibleClient(BaseLLMClient):
    """OpenAI兼容客户端"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = get_api_key(config.get('api_key_env', 'OPENAI_API_KEY'))
        self.base_url = config.get('base_url', 'https://api.openai.com/v1')
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """调用OpenAI兼容API
        
        Args:
            messages: 消息列表
            
        Returns:
            Dict: API响应
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens
        }
        
        url = f"{self.base_url}/chat/completions"
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error(f"OpenAI兼容API调用失败: {e}")
                raise
    
    def get_token_usage(self, response: Dict[str, Any]) -> int:
        """获取OpenAI token使用量"""
        usage = response.get('usage', {})
        return usage.get('total_tokens', 0)


class LLMClientManager:
    """LLM客户端管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = self._create_client()
        self.rate_limiter = RateLimiter(
            qps=config.get('rate_limit_qps', 2),
            max_concurrent=config.get('parallel_requests', 8)
        )
        
    def _create_client(self) -> BaseLLMClient:
        """创建LLM客户端
        
        Returns:
            BaseLLMClient: 客户端实例
        """
        provider = self.config.get('provider', 'dashscope')
        
        if provider == 'dashscope':
            return DashScopeClient(self.config)
        elif provider == 'openai_compatible':
            return OpenAICompatibleClient(self.config)
        else:
            raise ValueError(f"不支持的LLM提供商: {provider}")
    
    async def extract_labels(self, request: LLMExtractionRequest) -> LLMExtractionResponse:
        """提取标签
        
        Args:
            request: 抽取请求
            
        Returns:
            LLMExtractionResponse: 抽取响应
        """
        start_time = time.time()
        
        try:
            # 构建消息
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": request.text}
            ]
            
            # 调用API
            async with self.rate_limiter:
                response = await self.client.generate_response(messages)
            
            # 解析响应
            content = self._extract_content(response)
            labels = self._parse_labels(content, request.content_id)
            
            processing_time = time.time() - start_time
            tokens_used = self.client.get_token_usage(response)
            
            return LLMExtractionResponse(
                content_id=request.content_id,
                extracted_labels=labels,
                tokens_used=tokens_used,
                processing_time=processing_time,
                model_name=self.client.model_name
            )
            
        except Exception as e:
            logger.error(f"标签提取失败 content_id={request.content_id}: {e}")
            raise
    
    def _get_system_prompt(self) -> str:
        """获取系统提示"""
        # 这里可以从prompts模块导入
        return """你是美妆内容标注助手。任务：从给定内容中抽取预定义标签。
要求：仅使用给定的枚举标签，返回严格JSON格式。"""
    
    def _extract_content(self, response: Dict[str, Any]) -> str:
        """提取响应内容
        
        Args:
            response: API响应
            
        Returns:
            str: 内容文本
        """
        if 'output' in response:  # DashScope格式
            return response['output']['text']
        elif 'choices' in response:  # OpenAI格式
            return response['choices'][0]['message']['content']
        else:
            raise ValueError("无法解析API响应格式")
    
    def _parse_labels(self, content: str, content_id: str) -> ContentLabels:
        """解析标签内容
        
        Args:
            content: 响应内容
            content_id: 内容ID
            
        Returns:
            ContentLabels: 解析后的标签
        """
        try:
            # 尝试解析JSON
            data = json.loads(content)
            
            # 验证并创建ContentLabels对象
            labels = ContentLabels(
                content_id=content_id,
                **{k: v for k, v in data.items() if k != 'content_id'}
            )
            
            return labels
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败 content_id={content_id}: {e}")
            logger.error(f"原始内容: {content}")
            
            # 返回空标签
            return ContentLabels(content_id=content_id)
        except Exception as e:
            logger.error(f"标签解析失败 content_id={content_id}: {e}")
            return ContentLabels(content_id=content_id)


class RateLimiter:
    """速率限制器"""
    
    def __init__(self, qps: float = 2.0, max_concurrent: int = 8):
        self.qps = qps
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.last_request_time = 0.0
        self.lock = asyncio.Lock()
    
    async def __aenter__(self):
        await self.semaphore.acquire()
        
        async with self.lock:
            now = time.time()
            time_since_last = now - self.last_request_time
            min_interval = 1.0 / self.qps
            
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                await asyncio.sleep(sleep_time)
            
            self.last_request_time = time.time()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.semaphore.release()


class BatchProcessor:
    """批处理器"""
    
    def __init__(self, client_manager: LLMClientManager, batch_size: int = 16):
        self.client_manager = client_manager
        self.batch_size = batch_size
    
    async def process_batch(self, requests: List[LLMExtractionRequest]) -> List[LLMExtractionResponse]:
        """批量处理请求
        
        Args:
            requests: 请求列表
            
        Returns:
            List[LLMExtractionResponse]: 响应列表
        """
        results = []
        
        # 分批处理
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            
            # 并发处理当前批次
            tasks = [
                self.client_manager.extract_labels(request)
                for request in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"批处理失败 content_id={batch[j].content_id}: {result}")
                    # 创建错误响应
                    error_response = LLMExtractionResponse(
                        content_id=batch[j].content_id,
                        extracted_labels=ContentLabels(content_id=batch[j].content_id),
                        tokens_used=0,
                        processing_time=0.0,
                        model_name=self.client_manager.client.model_name
                    )
                    results.append(error_response)
                else:
                    results.append(result)
            
            logger.info(f"完成批次 {i//self.batch_size + 1}/{(len(requests)-1)//self.batch_size + 1}")
        
        return results


def create_llm_client(config: Dict[str, Any]) -> LLMClientManager:
    """便捷函数：创建LLM客户端
    
    Args:
        config: 配置字典
        
    Returns:
        LLMClientManager: 客户端管理器
    """
    return LLMClientManager(config)