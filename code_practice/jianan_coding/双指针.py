from typing import List


# leetcode167 两数之和 无重复有序数组
class Solution:
    def twoSum(self, nums: List[int], target: int):
        left = 0
        right = len(nums) - 1
        while left < right:
            _sum = nums[left] + nums[right]
            if _sum == target:
                return [left + 1, right + 1]
            elif _sum > target:
                right -= 1
            elif _sum < target:
                left += 1

        return []

class Solution:
    def twoSum(self, nums: List[int],target: int):
        left = 0 
        right = len(nums) -1 
        ans = []
        while left < right:
            _sum = nums[left] + nums[right]
            if _sum > target:
                right -= 1 
            elif _sum < target:
                left += 1
            if _sum == target:
                ans.append([nums[left],nums[right]])
                left += 1 
                right -= 1  
        
        return ans 

# 两数之和 有重复有序数组
class Solution:
    def twoSum(self, nums: List[int], target: int):
        res = []
        lo, hi = 0, len(nums) - 1
        while lo < hi:
            _sum = nums[lo] + nums[hi]
            left = nums[lo]
            right = nums[hi]
            if _sum < target:
                while lo < hi and nums[lo] == left:
                    lo += 1
            elif _sum > target:
                while lo < hi and nums[hi] == right:
                    hi -= 1
            if _sum == target:
                res.append([nums[lo], nums[hi]])
                while lo < hi and nums[lo] == left:
                    lo += 1
                while lo < hi and nums[hi] == right:
                    hi -= 1

        return res


# leetcode15 三数之和
class Solution:
    def threeSum(self,nums:List[int],target:int):
        nums.sort()
        # 三数之和顺序不重要
        # i < j < k 
        # 答案不能重复
        ans = []
        n = len(nums)
        for i in range(n-2):
            x = nums[i]
            if  i > 0 and x == nums[i-1]:
                continue
            if x + nums[i+1] + nums[i+2] > 0:
                break
            if x + nums[-1] + nums[-2] < 0:
                break
            j = i + 1 
            k = n - 1 
            while j < k:
                s = x + nums[j] + nums[k]
                if s > 0:
                    k -= 1 
                elif s < 0:
                    j += 1 
                else:
                    ans.append([x,nums[j],nums[j]])
                    j += 1
                    while j < k and nums[j] == nums[j-1]:
                        j += 1 
                    while j < k and nusm[k] = nums[k+1]:
                        k -= 1 

        return ans 
