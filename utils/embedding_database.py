import numpy as np
import faiss
import json
import os
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional

class EmbeddingDatabase:
    def __init__(self, model_name: str = '../bge-large-en-v1.5', dimension: int = None):
        """
        初始化嵌入数据库
        
        :param model_name: 使用的句子转换器模型名称
        :param dimension: 可选的嵌入维度（自动检测模型维度）
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = dimension or self.model.get_sentence_embedding_dimension()
        self.index = None
        self.strings = []  # 存储所有字符串
        
    def create_index(self, strings: List[str]):
        """
        创建新的数据库并添加字符串
        
        :param strings: 要添加到数据库的字符串列表
        """
        # 检查输入
        if not strings:
            raise ValueError("字符串列表不能为空")
            
        # 创建FAISS索引
        self.index = faiss.IndexFlatIP(self.dimension)  # 内积索引（余弦相似度）
        self.strings = []
        
        # 添加字符串
        self.add_strings(strings)
    
    def add_strings(self, strings: List[str]):
        """
        向数据库中添加新字符串
        
        :param strings: 要添加的字符串列表
        """
        if not self.index:
            # raise RuntimeError("数据库未初始化，请先调用create_index或load_database")
            self.create_index(strings)
            return
            
        if not strings:
            return
            
        # 编码新字符串
        new_embeddings = self.model.encode(strings, convert_to_numpy=True)
        new_embeddings = np.array(new_embeddings).astype('float32')
        
        # 归一化向量（使内积等于余弦相似度）
        faiss.normalize_L2(new_embeddings)
        
        # 添加到索引
        self.index.add(new_embeddings)
        
        # 更新内部存储
        self.strings.extend(strings)
    
    def save_database(self, file_prefix: str):
        """
        保存数据库到本地文件
        
        :param file_prefix: 文件前缀（将创建两个文件：.faiss 和 .json）
        """
        if not self.index:
            raise RuntimeError("数据库未初始化，无法保存")
            
        # 保存FAISS索引
        faiss_file = f"{file_prefix}.faiss"
        faiss.write_index(self.index, faiss_file)
        
        # 保存元数据
        meta_file = f"{file_prefix}.json"
        metadata = {
            "dimension": self.dimension,
            "strings": self.strings,
            "model_name": self.model.get_sentence_embedding_dimension()
        }
        
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_database(self, file_prefix: str):
        """
        从本地文件加载数据库
        
        :param file_prefix: 文件前缀（将加载 .faiss 和 .json 文件）
        """
        # 加载FAISS索引
        faiss_file = f"{file_prefix}.faiss"
        if not os.path.exists(faiss_file):
            raise FileNotFoundError(f"FAISS文件不存在: {faiss_file}")
        self.index = faiss.read_index(faiss_file)
        
        # 加载元数据
        meta_file = f"{file_prefix}.json"
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"元数据文件不存在: {meta_file}")
            
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        
        # 验证维度匹配
        if metadata["dimension"] != self.dimension:
            # 如果维度不匹配，尝试重新初始化模型
            try:
                self.model = SentenceTransformer(metadata["model_name"])
                self.dimension = metadata["dimension"]
            except:
                raise ValueError(f"维度不匹配: 数据库维度 {metadata['dimension']}, 当前模型维度 {self.dimension}")
        
        # 加载字符串
        self.strings = metadata["strings"]
    
    def search_similar(self, query: str, top_k: int = 3, min_similarity: float = 0.8) -> List[Tuple[str, float]]:
        """
        检索与查询字符串最相似的top-k个字符串（余弦相似度 > min_similarity）
        
        :param query: 查询字符串
        :param top_k: 返回的结果数量
        :param min_similarity: 最小余弦相似度阈值 (0.0-1.0)
        :return: 包含(字符串, 相似度)的列表
        """
        if not self.index:
            raise RuntimeError("数据库未初始化")
            
        # 限制top_k不超过数据库大小
        top_k = min(top_k, len(self.strings))
        if top_k <= 0:
            return []
        
        # 编码查询字符串
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # 搜索相似字符串
        distances, indices = self.index.search(query_embedding, top_k)
        
        # 组织结果 - 只返回相似度 > min_similarity 的结果
        results = []
        for i in range(top_k):
            idx = indices[0, i]
            if idx < 0:  # 无效索引
                continue
                
            similarity = distances[0, i]  # 内积即余弦相似度
            
            # 应用相似度阈值
            if similarity < min_similarity:
                continue  # 跳过不满足阈值的项
                
            results.append((self.strings[idx], similarity))
        
        # 如果结果不足top_k，尝试搜索更多可能项
        if len(results) < top_k and top_k < len(self.strings):
            # 搜索更多候选项（最多搜索数据库大小的10%）
            max_search = min(100, max(50, len(self.strings) // 10))
            distances, indices = self.index.search(query_embedding, max_search)
            
            # 收集所有满足条件的项
            for i in range(max_search):
                if len(results) >= top_k:
                    break  # 已经收集足够的结果
                    
                idx = indices[0, i]
                if idx < 0:
                    continue
                    
                similarity = distances[0, i]
                if similarity >= min_similarity:
                    # 确保不重复添加
                    if self.strings[idx] not in [r[0] for r in results]:
                        results.append((self.strings[idx], similarity))
        
        # 按相似度降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 如果结果仍然超过top_k，截断
        return results[:top_k]

    def get_database_size(self) -> int:
        """返回数据库中的字符串数量"""
        return len(self.strings) if self.index else 0

    def clear_database(self):
        """清空数据库"""
        self.index = None
        self.strings = []
    
    def find_most_similar(self, query: str, min_similarity: float = 0.8) -> Optional[Tuple[str, float]]:
        """
        查找与查询字符串最相似的一个字符串（余弦相似度 > min_similarity）
        如果没有满足条件的项，返回None
        
        :param query: 查询字符串
        :param min_similarity: 最小余弦相似度阈值
        :return: (最相似的字符串, 相似度) 或 None
        """
        results = self.search_similar(query, top_k=1, min_similarity=min_similarity)
        return results[0] if results else None