#
# @lc app=leetcode.cn id=46 lang=python3
#
# [46] 全排列
#
from typing import List
# @lc code=start
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        self.res = []
        self.path = []
        self.used = [False] * len(nums)
        self.backtracking(nums)
        return self.res

    def backtracking(self,nums:list):
        if len(self.path) == len(nums):
            self.res.append(self.path[:])
            return
        for i in range(len(nums)):
            if self.used[i]:
                continue
            self.path.append(nums[i])
            self.used[i] = True
            self.backtracking(nums)
            self.path.pop()
            self.used[i] = False
# @lc code=end

