#
# @lc app=leetcode.cn id=17 lang=python3
#
# [17] 电话号码的字母组合
#
from typing import List
# @lc code=start
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        self.mapping = ["", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"]
        self.res = [] # 外部变量
        self.path = [] # 外部变量
        if not digits:
            return []
        self.backtracking(digits,0)
        return self.res
    
    def backtracking(self, digits:str,index:int):
        """递归遍历,无返回值，回溯思想，用一个backtracking函数配合外部变量来实现，这叫「遍历」的思维模式。
            递归三要素：
                1.确定递归函数的参数和返回类型
                2.确定终止条件
                3.确定单层递归逻辑
        Args:
            digits (str): _description_
            index (int): index是记录遍历第几个数字了，就是用来遍历digits的（题目中给出数字字符串），同时index也表示树的深度。
        """
        if index == len(digits):
            self.res.append("".join(self.path))
            return
        digit = int(digits[index])
        letter = self.mapping[digit]
        for i in range(len(letter)):
            self.path.append(letter[i])
            self.backtracking(digits,index+1)
            self.path.pop()
       
# @lc code=end

