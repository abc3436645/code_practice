from typing import List

"""有些题目，你按照拍脑袋的方式去做，可能发现需要在递归代码中调用其他递归函数计算字数的信息。
一般来说，出现这种情况时你可以考虑用后序遍历的思维方式来优化算法，
利用后序遍历传递子树的信息，避免过高的时间复杂度。
"""

class TreeNode:
    def __init__(self,val,left=None,right=None) -> None:
        self.val = val 
        self.left = left
        self.right = right
        
# leetcode 110 平衡二叉树
class Solution:
    def isBalanced(self,root:TreeNode):
        """_summary_
            一般的拍脑袋思路是，遍历二叉树，然后对每一个节点计算左右的最大高度。
            但是计算一棵二叉树的最大深度也需要递归遍历这棵树的所有节点，如果对每个节点都算一遍最大深度，时间复杂度是比较高的。
            所以最好的解法是反过来思考，只计算一次最大深度，计算的过程中在后序遍历位置顺便判断二叉树是否平衡：
            对于每个节点，先算出来左右子树的最大高度，然后在后序遍历的位置根据左右子树的最大高度判断平衡性。
        Args:
            root (TreeNode): _description_
        """
        self.balance = True
        self.maxDepth(root)

        return self.balance
    
    def maxDepth(self,root:TreeNode):
        if not root:
            return 0
        left_max_depth = self.maxDepth(root.left)
        right_max_depth = self.maxDepth(root.right)
        if abs(left_max_depth-right_max_depth) > 1:
            self.balance = False
            
        return max(left_max_depth,right_max_depth) + 1 