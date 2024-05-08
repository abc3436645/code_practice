#
# @lc app=leetcode.cn id=144 lang=python3
#
# [144] 二叉树的前序遍历
#

# @lc code=start
# Definition for a binary tree node.

from typing import List

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        result = self.traversal(root)
        return result

    def traversal(self,cur: TreeNode):
        if not cur : 
            return []

        left = self.traversal(cur.left)
        right = self.traversal(cur.right) 

        result = [cur.val] + left + right

        return result

# @lc code=end

"""二叉树前序遍历，迭代解法
"""
def preorderTraversal(root: TreeNode) -> List[int]:
    if not root:
        return []
    
    stack = [root]  # 用栈模拟递归
    result = []

    while stack:
        cur = stack.pop()
        result.append(cur.val)

        if cur.right:
            stack.append(cur.right)
        if cur.left:
            stack.append(cur.left)

    return result


"""二叉树中序遍历，迭代解法
"""
def inorderTraversal(root: TreeNode) -> List[int]:
    if not root:
        return []
    
    stack = []  # 不能将root节点提前入栈
    result = []
    cur = root

    while stack or cur:
        # 先迭代访问最底层的左子树结点
        if cur:
            stack.append(cur)
            cur = cur.left
        # 到达最左结点后处理栈顶结点 
        else:
            cur = stack.pop()
            result.append(cur.val)
            # 取出栈顶结点后，访问其右子树
            cur = cur.right

    return result