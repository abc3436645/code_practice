#
# @lc app=leetcode.cn id=40 lang=python3
#
# [40] 组合总和 II
#
from typing import List

# @lc code=start
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        self.res = [] # 外部变量
        self.path = [] # 外部变量
        candidates.sort()
        self.backtracking(candidates,target,0)
        return self.res

    def backtracking(self,candidates:List,target:int,start:int):
        """递归遍历,无返回值，回溯思想，用一个backtracking函数配合外部变量来实现，这叫「遍历」的思维模式。
            递归三要素：
                1.确定递归函数的参数和返回类型
                2.确定终止条件
                3.确定单层递归逻辑

        Args:
            candidates (List): _description_
            target (int): _description_
            start (int): _description_
        """
        if target == 0:  # 终止条件
            self.res.append(self.path[:])
            return
        if target < 0: # 终止条件
            return
        for i in range(start,len(candidates)):
            if i >start and candidates[i] == candidates[i-1]:  # 剪枝
                    continue
            self.path.append(candidates[i]) # 回溯 做选择
            target -= candidates[i] # 回溯 做选择
            self.backtracking(candidates,target,i+1)  # 控制元素是否可以重复，和index相关的话代表不可重复，反之可重复
            target += candidates[i] # 回溯 撤销选择
            self.path.pop() # 回溯 撤销选择

        
# @lc code=end

