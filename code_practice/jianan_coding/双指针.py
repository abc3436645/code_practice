# leetcode 209 长度最小的子数组

"""
画图、画图、画图
双指针，思考窗口变化, while条件就是从满足要求不断变成不满足要求，同时对于满足要求的情况不断更新res
while条件也可以从不满足要求到满足要求
满足单调性才可以使用双指针
"""

from typing import Counter, List

from code_practice.dynamic_programming import length_of_LIS


class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]):
        n = len(nums)
        res = n + 1  # inf 答案初始化
        s = 0
        left = 0
        for right, x in enumerate(nums):  # right从0...n-1  x:nums[right]
            s += x
            while s - nums[left] > target:
                s -= nums[left]
                left += 1
            if s >= target:
                res = min(res, right - left + 1)
        return res if res <= n else 0

    def minSubArrayLen2(self, target: int, nums: List[int]):
        n = len(nums)
        res = n + 1
        s = 0
        left = 0
        for right, x in enumerate(nums):
            s += x
            # 单调性
            while s >= target:
                res = min(res, right - left + 1)
                s -= nums[left]
                left += 1
        return res if res <= n else 0


# leetcode 713 乘积小于k的子数组个数
class Solution:
    def numSubarrayProductLessThanK(self,nums:List[int],k:int):
        # 子数组的数目计算 
        # 左端点left 右端点为right,子数组个数是以右端点right的个数
        # [left,right] [left+1,right] ... [right,right] 个数为 right-left+1 
        if k <= 1:
            return 0
        ans = 0
        prod = 1 
        left = 0
        for right,x in enumerate(nums):
            prod *= x 
            while prod >=k:
                prod /= nums[left]
                left += 1 
            ans += right - left + 1 

        return ans 

# leetcode 3 无重复字符的最长子串
class Solution:
    def lengthOfLongestSubstring(self,s:str):
        # 如何判断子串是否有重复元素，利用哈希表来实现
        ans = 0 
        cnt = Counter() # hashmap char int
        left = 0 
        for right, c in  enumerate(s):
            cnt[c] += 1 
            while cnt[c] > 1: 
                cnt[s[left]] -= 1 
                left += 1 

            ans = max(ans,right-left+1) # 字符个数

        return ans 




