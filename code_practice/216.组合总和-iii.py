#
# @lc app=leetcode.cn id=216 lang=python3
#
# [216] 组合总和 III
#
from typing import List
# @lc code=start
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        self.res = [] # 外部变量
        self.path = [] # 外部变量
        self.backtracking(k,n,1)
        return self.res

    def backtracking(self, k, n, start):
        """递归遍历,无返回值，回溯思想，用一个backtracking函数配合外部变量来实现，这叫「遍历」的思维模式。
            递归三要素：
                1.确定递归函数的参数和返回类型
                2.确定终止条件
                3.确定单层递归逻辑

        Args:
            k (_type_): _description_
            n (_type_): _description_
            start (_type_): _description_
        """
        if sum(self.path) > n or len(self.path) >n: # 单层逻辑
            return
        if sum(self.path) == n and len(self.path) == k: # 单层逻辑
            self.res.append(self.path[:])
            return
        for i in range(start, 10): 
            self.path.append(i) # 回溯 做选择
            self.backtracking(k, n, i + 1)
            self.path.pop() # 回溯 撤销选择
# @lc code=end

