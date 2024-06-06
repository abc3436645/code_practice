from typing import List
from typing_extensions import IntVar


# leetcode 78 子集 元素无重复不可复选
class Solution:
    def __init__(self):
        self.res = []
        self.path = []

    def subsets(self, nums: List[int]):
        if not nums:
            return []
        self.backtracking(nums, 0)

        return self.res

    def backtracking(self, nums: List, start: int):
        # 前序遍历，每个节点都是一个子集
        self.res.append(self.path[:])
        for i in range(start, len(nums)):
            self.path.append(nums[i])
            self.backtracking(nums, i + 1)
            self.path.pop()


# leetcode 77 组合（元素无重不可复选）
class Solution:
    def __init__(self) -> None:
        self.res = []
        self.path = []

    def combine(self, n: int, k: int):
        self.backtracking(n, k, 1)
        return self.res

    def backtracking(self, n: int, k: int, start: int):
        if len(self.path) == k:
            self.res.append(self.path[:])
            return

        for i in range(start, n + 1):
            self.path.append(i)
            self.backtracking(n, k, i + 1)
            self.path.pop()


# leetcode 90 子集 元素可重复不可复选
class Solution:
    def __init__(self):
        self.res = []
        self.path = []

    def subsetsWithDup(self, nums: List[int]):
        nums.sort()
        self.backtracking(nums, 0)
        return self.res

    def backtracking(self, nums: List[int], start: int):
        self.res.append(self.path[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue

            self.path.append(nums[i])
            self.backtracking(nums, i + 1)
            self.path.pop()


# leetocde 39 子集/组合（元素无重可复选）
class Solution:
    def __init__(self) -> None:
        self.res = []
        self.path = []
        self.pathSum = 0

    def combinationSum(self, nums: List[int], target: int):
        if len(nums) == 0:
            return self.res
        self.backtracking(nums, 0, target)
        return self.res

    def backtracking(self, nums: List[int], start: int, target: int):
        if self.pathSum == target:
            self.res.append(self.path[:])
            return self.res

        if self.pathSum > target:
            return

        for i in range(start, len(nums)):
            self.path.append(nums[i])
            self.pathSum += nums[i]
            # 同一元素可重复使用，注意参数
            self.backtracking(nums, i, target)
            self.pathSum -= nums[i]
            self.path.pop()
