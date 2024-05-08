#
# @lc app=leetcode.cn id=90 lang=python3
#
# [90] 子集 II
#
from typing import List

# @lc code=start
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        self.res = [] # 外部变量
        self.path = [] # 外部变量
        nums.sort()  
        self.backtracking(nums,0)
        return self.res
    
    def backtracking(self,nums:List,start:int):
        """递归遍历,无返回值，回溯思想，用一个backtracking函数配合外部变量来实现，这叫「遍历」的思维模式。
            递归三要素：
                1.确定递归函数的参数和返回类型
                2.确定终止条件
                3.确定单层递归逻辑

        Args:
            nums (List): _description_
            start (int): _description_
        """
        self.res.append(self.path[:]) # base case
        for i in range(start,len(nums)):
            if i > start and nums[i] == nums[i-1]: # 控制重复不可复选
                continue
            self.path.append(nums[i]) # 回溯 做选择
            self.backtracking(nums,i+1) 
            self.path.pop() # 回溯 撤销选择
# @lc code=end

