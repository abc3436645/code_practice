import re
from typing import List

# 链表ListNode
class ListNode:
    def __init__(self, val, next):
        self.val = val
        self.next = next


# 二叉树TreeNode
class TreeNode:
    def __init__(self, val, left, right):
        self.val = val
        self.left = left
        self.right = right

""" 二叉树解题框架
**二叉树的所有问题，就是让你在前中后序位置注入巧妙的代码逻辑，去达到自己的目的，你只需要单独思考每一个节点应该做什么，其他的不用你管，抛给二叉树遍历框架，递归会在所有节点上做相同的操作**。
**二叉树递归遍历**: **如果需要遍历整棵树，递归函数就不能有返回值。如果需要遍历某一条固定路线，递归函数就一定要有返回值！，因为遇到符合条件的就要及时返回**。

- 二叉树解题的两种思维模式：**遍历和分解问题**
    - **是否可以通过遍历一遍二叉树得到答案**？如果可以，用一个`traverse`函数配合外部变量来实现，这叫「**遍历**」的思维模式。
    - **是否可以定义一个递归函数，通过子问题（子树）的答案推导出原问题的答案**？如果可以，写出这个递归函数的定义，并**充分利用这个函数的返回值**，这叫「**分解问题**」的思维模式。

    无论使用哪种思维模式，都需要思考，**如果单独抽出一个二叉树节点，它需要做什么事情？需要在什么时候（前/中/后序位置）做**？其他的节点不用你操心，递归函数会帮你在所有节点上执行相同的操作。


递归遍历三要素
    1. 确定递归函数的参数和返回值： 确定哪些参数是递归的过程中需要处理的，那么就在递归函数里加上这个参数，
        并且还要明确每次递归的返回值是什么进而确定递归函数的返回类型。
    2. 确定终止条件： 写完了递归算法, 运行的时候，经常会遇到栈溢出的错误，就是没写终止条件或者终止条件写的不对，
        操作系统也是用一个栈的结构来保存每一层递归的信息，如果递归没有终止，操作系统的内存栈必然就会溢出。
    3. 确定单层递归的逻辑： 确定每一层递归需要处理的信息。在这里也就会重复调用自己来实现递归的过程。
"""

# N叉树
class NTreeNode:
    def __init__(self, val:int, children:List[TreeNode]):
        self.val = val
        self.children = children


# 数组arr遍历
def traverse(arr):
    """
    数组迭代遍历
    """
    for i in range(len(arr)):
        # do_something()
        pass

def traverse(arr,i):
    """
    数据递归遍历
    """
    if i == len(arr):
        return
    # do_something()
    traverse(arr,i+1)
    # do_something()

# 链表head遍历
def traverse(head: ListNode):
    """
    链表迭代遍历
    """
    if not head:
        return
    p = head
    while p:
        # do_something()
        p = p.next


def traverse(head: ListNode):
    """
    链表递归遍历
    """
    if not head:
        return
    # do_something() 前序遍历
    traverse(head.next)
    # do_something() 逆序遍历


# 二叉树遍历
def traverse(root: TreeNode):
    if not root:
        return
    # do_something() 前序遍历
    traverse(root.left)
    # do_something() 中序遍历
    traverse(root.right)
    # do_something() 后序遍历


def traverse(root: TreeNode):
    """
    二叉树迭代遍历
    """
    if not root:
        return []
    res = []
    stack = []
    stack.append(root)
    while stack:
        node = stack.pop()
        # do_something() 前序遍历
        res.append(node.val)
        if node.left:
            stack.append(node.left)
        # do_something() 中序遍历
        if node.right:
            stack.append(node.right)
        # do_something() 后序遍历


# N叉树递归遍历
def traverse(root: NTreeNode):
    if not root:
        return
    for child in root.children:
        traverse(child)
