from typing import List

# 数组遍历框架，典型的线性迭代结构
def traverse(array:list):
    for i in range(len(array)):
        # 迭代遍历
        pass


# 链表遍历，兼具迭代和递归结构
class ListNode():
    def __init__(self,val):
        self.val = val
        self.next = None

def traverse(head:ListNode):
    p = head 
    while p is not None:
        # 迭代访问p.val
        p = p.next

def traverse(head:ListNode):
    # 递归遍历p.val   
    
    # 前序遍历
    traverse(head.next)
    # 后序遍历

# 二叉树遍历，典型递归结构
class TreeNode():
    def __init__(self,val,left=None,right=None):
        self.val = val
        self.left = left
        self.right = right


def traverse(root:TreeNode):
    # 前序遍历
    traverse(root.left)
    # 中序遍历    
    traverse(root.right)
    # 后序遍历

# N叉树遍历，递归结构
class TreeNode():
    def __init__(self,val,children:List[TreeNode]):
        self.val = val
        self.children = children
    
def traverse(root:TreeNode):
    for child in root.children:
        traverse(child)

