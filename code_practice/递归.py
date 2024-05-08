#   二叉树解题的两种思维模式：遍历和分解问题
#   1. 是否可以通过遍历一遍二叉树得到答案？如果可以，用一个traverse函数配合外部变量来实现，这叫遍历的思维模式。
#   2. 是否可以定义一个递归函数，通过子问题（子树）的答案推导出原问题的答案？
#         如果可以，写出这个递归函数的定义，并充分利用这个函数的返回值，这叫分解问题的思维模式。
#   3. 无论使用哪种思维模式，都需要思考，**如果单独抽出一个二叉树节点，它需要做什么事情？
#         需要在什么时候（前/中/后序位置）做**？其他的节点不用你操心，递归函数会帮你在所有节点上执行相同的操作。


# 递归三要素
"""
    1. 确定递归函数的参数和返回类型
        1.1 遍历思维模式
            用一个traverse函数配合外部变量，其他局部遍历基本上都是递归函数入参
        1.2 分解问题思维模式
            使用func()和定义函数返回值，充分利用这个函数的返回值
    2. 确定终止条件
    3. 确定单层递归逻辑
        如果单独抽出一个二叉树节点，它需要做什么事情？需要在什么时候（前/中/后序位置）做？
"""


# 二叉树遍历
# 中序遍历

class TreeNode():
    def __init__(self,val,left:None,right:None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def preorder(self,root:TreeNode):
        self.result = []  # 外部变量
        self.traverse(root)
        return self.result
    
    # 遍历思维模式,使用traverse()和外部变量，无返回值
    def traverse(self,root:TreeNode):
        if not root:
            return 
        self.result.append(root.val)
        self.result.append(self.traverse(root.left))
        self.result.append(self.traverse(root.right))


class Solution:
    def preorder(self,root:TreeNode):
        # 分解问题思路：
        # 函数定义：输入为root，输出以该节点为根节点的树的前序遍历
        if not root:
            return []
        left_res = self.preorder(root.left)
        right_res = self.preorder(root.right)

        return [root.val] + left_res + right_res


    def preorder(self,root:TreeNode):
        # 分解问题思路
        # 函数定义：输入节点为root，返回以该节点为根节点的树的前序遍历结果
        result = [] 
        if not root:
            return result
        result.append(root.val)
        result.extend(self.preorder(root.left))
        result.extend(self.preorder(root.right))

        return result
    


# 二叉树最大路径
class Solution:
    """
       每一条二叉树的「直径」长度，就是一个节点的左右子树的最大深度之和
       分解问题思路，定义一个函数并充分利用这个函数的定义进行求解
    """
    def binary_tree_diameter(self,root:TreeNode):
        self.max_diameter = 0 
        self.max_depth(root)
        return self.max_diameter

    def max_depth(self,root:TreeNode): 
        if not root:
            return 0
        left_depth = self.max_depth(root.left)
        right_depth = self.max_depth(root.right)
        diameter = left_depth + right_depth  # 顺带求出每颗子树的深度
        self.max_diameter = max(self.max_diameter,diameter) # 更新最大深度

        return max(left_depth,right_depth) + 1
