class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None
    

class Node:
    def __init__(self, val=0,children=None):
        self.val = val
        self.children = children



def preorder_tree(root:TreeNode,res:list):
    """
        二叉树前序遍历，使用递归
        先遍历根节点，左孩子，又孩子
    
    递归三要素:
        1.确定递归函数参数和返回值，明确每次递归的返回值是什么进而确定递归函数的返回类型
        2.确定递归函数终止条件
        3.确定单层递归的逻辑

    Args:
        root (TreeNode): 二叉树根节点
        res (list):  遍历数组res
    """
    if root is None:
        return []
    res.append(root.val)
    preorder_tree(root.left,res)
    preorder_tree(root.right,res)

def preorder(root:TreeNode):
    res = []
    preorder_tree(root,res)
    return res

def mid_order_tree(root:TreeNode,res:list):
    if root is None:
        return []
    mid_order_tree(root.left,res)
    res.append(root.val)
    mid_order_tree(root.right)

def mid_order(root:TreeNode):
    res = []
    mid_order_tree(root,res)
    return res







    
    