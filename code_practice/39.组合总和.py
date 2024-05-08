#
# @lc app=leetcode.cn id=39 lang=python3
#
# [39] 组合总和
#
from typing import List

# @lc code=start
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        self.res = [] # 符合要求的结合的集合 外部变量
        self.path = [] # 路径 外部变量
        self.backtracking(candidates,target,0)
        return self.res

    def backtracking(self,candidates:list,target:int,start:int):
        """递归遍历,无返回值，回溯思想，用一个backtracking函数配合外部变量来实现，这叫「遍历」的思维模式。
            递归三要素：
                1.确定递归函数的参数和返回类型
                2.确定终止条件
                3.确定单层递归逻辑

        Args:
            candidates (list): _description_
            target (int): _description_
            start (int): start控制是否可以复选
        """
        if target == 0:  # 终止条件
            self.res.append(self.path[:])
            return
        if target < 0:
            return
        for i in range(start,len(candidates)): 
            target -= candidates[i]  # 做选择
            self.path.append(candidates[i]) # 做选择
            self.backtracking(candidates,target,i)
            target += candidates[i] # 撤销选择
            self.path.pop() # 撤销选择

# @lc code=end

