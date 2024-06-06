# leetcode 3 无重复字符的最长子串
from typing import List


"""滑动窗口
输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
"""
from collections import Counter

class Solution:
    def lengthOfLongestSubstring(self,s:str):
        res = 0
        cnt = Counter() # hashmap c:1
        left = 0
        for right,c in enumerate(s):
            cnt[c] += 1 
            while cnt[c] > 1: # 当滑动窗口不满足的情况，肯定是新加入的c导致了cnt个数大于1,移动左指针
                cnt[s[left]] -= 1 
                left += 1 
            res = max(res,right-left+1)

        return res
    

# leetcode 209 长度最小的字数组
"""滑动窗口
输入：target = 7, nums = [2,3,1,2,4,3]
输出：2
解释：子数组 [4,3] 是该条件下的长度最小的子数组。
"""
class Solution:
    def minSubArrayLen(self,target:int,nums:List[int]):
        n = len(nums)
        res = n+1  # 初始化答案为一个比较大的数
        nums_sum = 0 
        left = 0 
        for right, x in  enumerate(nums):
            nums_sum += x 
            while nums_sum - nums[left] >= target:
                nums_sum -= nums[left]
                left += 1 
            if nums_sum >= target:
                res = min(res,right-left+1)

        return res if res <=n else 0 
    
    def minSubArrayLen2(self,target:int,nums:List[int]):
        n = len(nums)
        res = n+1 
        nums_sum = 0 
        left = 0 
        for right, x in  enumerate(nums):
            nums_sum += x
            while nums_sum >= target:
                ans = min(ans,right-left+1)
                s -= nums[left]
                left += 1 
        return res if res <=n else 0

# leetcode  713 元素乘积小于K的子数组
""" 滑动窗口
输入：nums = [10,5,2,6], k = 100
输出：8
解释：8 个乘积小于 100 的子数组分别为：[10]、[5]、[2],、[6]、[10,5]、[5,2]、[2,6]、[5,2,6]。
需要注意的是 [10,5,2] 并不是乘积小于 100 的子数组。
"""
class Solution:
    def numSubarrayProductLessThanK(self,nums:List[int],k:int):
        # l r 以r为结尾的连续子数组
        # [l,r],[l+1,r]....[r,r]
        # r-l+1个符合要求的子数组
        if k <= 1:
            return 0
        ans = 0 
        prod = 1 
        left = 0 
        for right,x in enumerate(nums):
            prod *= x 
            while prod >= k:
                prod /= nums[left]
                left += 1 
            ans += right - left + 1 

        return ans
