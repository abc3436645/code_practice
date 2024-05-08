"""
二分查找，主要是针对于有序数组， 需要对区间进行严格定义并严格按照区间设置进行处理
2种区间：左闭右闭[left, right] 左闭右开[left, right)

二分查找记忆：
    搜索一个元素时，搜索区间两端闭
    while条件带等号，否则需要打补丁
    if相等就返回，其他事情甭操心
    mid必须加减一，因为区间两端闭
    while结束就凉凉，凄凄惨惨返-1

    搜索左右边界时，搜索区间要阐明
    左闭右开最常见，其余逻辑便自明
    while要用小于号，这次才能不漏掉
    if相等别返回，利用mid锁边界
    mid加一或减一？ 要看区间开或闭
    while结束不算完，因为还没有返回
    索引可能出边界，if检查保平安
"""
from typing import List

def binarySearch(nums:List[int], target:int):
    left = 0, 
    right = ... # 左闭右开或者左闭右闭

    while(...): 
        mid = left + (right - left) // 2;
        if (nums[mid] == target):
            ... # TODO
        elif (nums[mid] < target):
            left = ... # TODO
        elif (nums[mid] > target):
            right = ... # TODO
    return ...


# 704二分查找
"""
输入: nums = [-1,0,3,5,9,12], target = 9     
输出: 4       
解释: 9 出现在 nums 中并且下标为 4    

输入: nums = [-1,0,3,5,9,12], target = 2     
输出: -1        
解释: 2 不存在 nums 中因此返回 -1        

"""
class Solution:
    def search(self,nums:List[int],target:int):
        left, right = 0, len(nums)-1  # 左闭右闭区间
        while left <= right:
            mid = left + (right-left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid -1 

        return -1


class Solution:
    def search(self,nums:List[int],target:int):
        left,right = 0,len(nums)  # 左闭右开区间
        while left < right:
            mid = left + (right-left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid
        return -1 


# 二分查找寻找左侧边界
class Solution:
    def left_bound(self, nums:List[int], target:int):
        left, right = 0, len(nums)-1  # 左闭右闭区间
        while left <= right:
            mid = left + (right-left) // 2 
            if target < nums[mid]:
                left = mid + 1 
            elif target > nums[mid]:
                right = mid - 1
            elif target == nums[mid]:
                right = mid - 1 # 收缩右侧边界

        if left >= len(nums) or left < 0: # 判断越界
            return -1 
        
        return left if target == nums[left] else -1
        

# 二分查找寻找右侧边界
class Solution:
    def right_bound(self, nums:List[int], target:int):
        left, right = 0, len(nums)-1  # 左闭右闭区间
        while left <= right:
            mid = left + (right-left) // 2
            if target < nums[mid]:
                left = mid + 1  
            elif target > nums[mid]:
                right = mid - 1
            elif target == nums[mid]:
                left = mid + 1  # 收缩左侧边界
        if left - 1 < 0 or left - 1 >= len(nums): # 判断越界
            return -1
        return left - 1 if nums[left - 1] == target else -1
    

# 二分查找左侧边界，左闭右开区间解法
class Solution:
    def left_bound(self,nums:List[int],target:int):
        left, right = 0,len(nums) # 左闭右开
        while left < right:
            mid = left + (right-left) // 2 
            if target < nums[mid]:
                left = mid + 1
            elif target > nums[mid]:
                right = mid 
            elif target == nums[mid]:
                right = mid

        if left < 0 or left >= len(nums):
            return -1
        
        return left
    
# 二分查找寻找右侧边界,左闭右开区间解法
class Solution:
    def right_bound(self,nums:List[int],target:int):
        left, right = 0, len(nums)
        while left < right:
            mid = left + (right - left) // 2 
            if target < nums[mid]:
                left = mid+1
            elif target > nums[mid]:
                right = mid
            elif target == nums[mid]:
                left = mid + 1
        
        if left < 0 or left >= len(nums):
            return -1

        return left - 1   