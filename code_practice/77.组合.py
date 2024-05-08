#
# @lc app=leetcode.cn id=77 lang=python3
#
# [77] 组合
#

from typing import List

# @lc code=start
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        self.res = [] # 符合要求的结果 外部变量
        self.path = [] # 路径  外部变量
        self.backtracking(n,k,1)
        return self.res

    def backtracking(self,n:int,k:int,start:int):
        """递归遍历,无返回值，回溯思想，用一个backtracking函数配合外部变量来实现，这叫「遍历」的思维模式。
            递归三要素：
                1.确定递归函数的参数和返回类型
                2.确定终止条件
                3.确定单层递归逻辑

        Args:
            n (int): _description_
            k (int): _description_
            start (int): _description_
        """
        if k == len(self.path):  # 终止条件
            self.res.append(self.path[:])
        for i in range(start,n+1): 
            self.path.append(i)  # 选择
            self.backtracking(n,k,i+1)
            self.path.pop() # 撤销选择
        

# @lc code=end

