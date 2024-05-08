#
# @lc app=leetcode.cn id=22 lang=python3
#
# [22] 括号生成
#
from typing import List
# @lc code=start
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        if n == 0:
            return []
        self.res = []
        self.path = []
        self.backtrack(n, n)
        return self.res 
    
    def backtrack(self, left, right):
        """递归遍历,无返回值，回溯思想，用一个backtracking函数配合外部变量来实现，这叫「遍历」的思维模式。
            递归三要素：
                1.确定递归函数的参数和返回类型
                2.确定终止条件
                3.确定单层递归逻辑

        Args:
            left (_type_): _description_
            right (_type_): _description_
        """
        if left == 0 and right == 0: # base case
            self.res.append(''.join(self.path))
            return
        if left > 0:
            self.path.append('(')  # 回溯，选择
            self.backtrack(left - 1, right)
            self.path.pop() # 回溯 撤销选择
        if right > left:
            self.path.append(')') # 回溯，选择
            self.backtrack(left, right - 1)
            self.path.pop() # 回溯，撤销选择
# @lc code=end

