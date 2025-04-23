# 使用vllm异步引擎，从而支持张量和数据同时并行
import logging
from vllm import AsyncEngineArgs,AsyncLLMEngine,SamplingParams
import pandas as pd
import os
import uuid
import asyncio
class VllmAsyncEngine:
    def __init__(self,config:dict):
        self.config = config
        self.model_path = config.get("model_path")
        self.temperature = config.get("temperature",0.6)
        self.top_p = config.get("top_p",0.9)
        self.max_tokens = config.get("max_tokens",8192)
        self.top_k = config.get("top_k",-1)
        self.tensor_parallel_size = config.get("tensor_parallel_size",1)
        self.pipeline_parallel_size = config.get("pipeline_parallel_size",8)
        self.max_model_len = config.get("max_model_len",16384)
        self.real_model_path = self.model_path
        self.engine = self.__init__engine()
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            top_k=self.top_k,
        )

    def __init__engine(self):
        logging.info(f"Initializing engine with model path: {self.real_model_path}")
        engine_args = AsyncEngineArgs(
            model=self.real_model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            max_model_len=self.max_model_len
        )
        return AsyncLLMEngine.from_engine_args(engine_args)

    async def run_query(self,id,query:str):
        outputs = self.engine.generate(query,self.sampling_params,id)
        async for output in outputs:
            final_output = output
        responses = []
        for output in final_output.outputs:
            responses.append(output.text)
        return responses,id
    
    async def process(self,queries:list):
        tasks = [asyncio.create_task(self.run_query(index,q)) for index,q in enumerate(queries)]
        results = []
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
        # sort results by id
        results = sorted(results,key=lambda x:x[1])
        return [result[0][0] for result in results]
    
    def run(self,queries:list):
        results = asyncio.run(self.process(queries))
        return results



    

        

