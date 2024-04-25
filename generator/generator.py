import pandas as pd
class GraphGenerator():
    def __init__(self, proposals:pd.DataFrame, 
                 members:pd.DataFrame,
                 votes:pd.DataFrame,
                 start_time, window_size, window_step):
        ###统计三个df的信息, 为similarity计算作准备.
        pass
    def similarity(self, u, v):
        pass
    def __iter__(self):
        pass
    def __next__(self):
        pass
    def __len__(self):
        pass
    def __getitem__(self, idx_or_slice):
        pass
    
"""
example usage:

gen = GraphGenerator(proposals, members, votes, start_time, window_size, window_step)
for (start, end) in sliding_windows:
    g_list = gen[start:end]##end might exceed the length of the generator. Handle the corner case.
    # do something with g_list
    
可能的加速方法: 多线程.

大致的生成方法:

将每个vote用similarity翻译成一条
(source target weight)记录(按时间排序), 这天然形成edge mask
然后直接滑动窗口去取就行了.
"""