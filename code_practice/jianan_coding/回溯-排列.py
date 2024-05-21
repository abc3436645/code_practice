from typing import List

class NtreeNode():
    def __init__(self,val:int,children:List[]) -> None:
        self.val = val
        self.chilidren = children


# N叉树递归遍历
def traversal(root:NtreeNode):
    for child in root.chilidren:
        # do_something() # 前序遍历
        traversal(child)
        # do_something() # 后序遍历

""" 回溯代码框架
for 选择 in 选择列表:
    # 做选择
    将该选择从选择列表移除
    路径.add(选择)
    backtrack(路径, 选择列表)
    # 撤销选择
    路径.remove(选择)
    将该选择再加入选择列表
"""

# leetcode 46 全排列 元素无重复不可复选
class Solution:
    def __init__(self):
        self.res = []
        self.path = []
        self.used = []

    def permute(self,nums:List[int]):
        self.used = [False] * len(nums)
        self.backtracking(nums)

        return self.res

    def backtracking(self,nums:List[int]):
        if len(self.path) == len(nums):
            self.path.append(self.path[:])
            return 
        
        for i, num in  enumerate(nums):
            if self.used[i]:
                continue
            self.path.append(num)
            self.used[i] = True
            self.backtracking(nums)
            self.used[i] = False
            self.path.pop()

# leetcode 47 全排列二 元素可重复不可复选
class Solution:
    def __init__(self) -> None:
        self.res = []
        self.path = []
        self.used = []

    def permuteUnique(self,nums:List[int]):
        nums.sort()
        self.used = [False] * len*(nums)
        self.backtracking(nums)
        return self.res

    def backtracking(self,nums:List[int]):
        if len(self.path) == len(nums):
            self.res.append(self.path[:])
            return
        
        for i,nums in enumerate(nums):
            if self.used[i]:
                continue
            if i > 0 and nums[i] == nums[i-1] and not self.used[i-1]:
                continue
            self.path.append(nums)
            self.used[i] = True
            self.backtracking(nums)
            self.used[i] = False
            self.path.pop()