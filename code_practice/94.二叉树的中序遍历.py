# -*- coding: utf-8 -*-
# @lc app=leetcode.cn id=94 lang=python3
#
# [94] 二叉树的中序遍历
#

# @lc code=start
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def inorderTraversal(self, root: TreeNode):
        self.res = [] # 外部变量
        if root is None:
            return []
        self.traverse(root)
        return self.res
    
    # def traverse(self,node:TreeNode,res:list):
    #     if node is None:
    #         return 
    #     self.traverse(node.left,res)
    #     res.append(node.val)
    #     self.traverse(node.right,res)

    def traverse(self,node:TreeNode):
        """二叉树中序遍历, 遍历思路
        1. 递归三要素：
            1.1 确定递归函数的参数和返回值
            1.2 确定终止条件
            1.3 确定单层递归逻辑
        2. 使用一个 traverse 函数配合外部变量来实现

        Args:
            node (TreeNode): _description_
        """
        if node is None:
            return
        self.traverse(node.left)
        self.res.append(node.val)
        self.traverse(node.right)

class Solution2:
    def inorderTraversal(self, root: TreeNode):
        """二叉树中序遍历, 分解问题思路

        Args:
            root (TreeNode): _description_

        Returns:
            _type_: _description_
        """
        res = []
        if root is None:
            return res
        res.extend(self.inorderTraversal(root.left))  # 左子树
        res.append(root.val) # 根节点
        res.extend(self.inorderTraversal(root.right)) # 右子树
        return res

# @lc code=end

