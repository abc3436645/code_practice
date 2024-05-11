from code_practice.data_structure import traverse


class TreeNode:
    def __init__(self,val,left=None,right=None) -> None:
        self.val = val 
        self.left = left
        self.right = right


# leetcode 230 二叉搜索树中第K小的元素
class Solution:
    def kthSmallest(self,root:TreeNode,k:int):
        self.res = 0
        self.rank = 0
        self.traverse(root,k)

        return self.res

    def traverse(self,root:TreeNode,k:int):
        if not root:
            return 
        self.traverse(root.left,k)
        self.rank += 1
        if self.rank == k:
            self.res = root.val 
            return
        self.traverse(root.right,k)


class Solution:
    def kthSmallest(self,root:TreeNode,k:int):
        pass


# leetcode 538 将二叉搜索树转换为累加树
# 解题思路：BST 的每个节点左小右大，这似乎是一个有用的信息，既然累加和是计算大于等于当前值的所有元素之和，那么每个节点都去计算右子树的和，不就行了吗？
# 这是不行的。对于一个节点来说，确实右子树都是比它大的元素，但问题是它的父节点也可能是比它大的元素呀？这个没法确定的，我们又没有触达父节点的指针，所以二叉树的通用思路在这里用不了。
# 其实，正确的解法很简单，还是利用 BST 的中序遍历特性。
# 这段代码可以降序打印 BST 节点的值，如果维护一个外部累加变量 sum，然后把 sum 赋值给 BST 中的每一个节点，不就将 BST 转化成累加树了吗？
class Solution:
    def convertBST(self,root:TreeNode):
        """
        遍历解法：使用外部变量+ traverse()函数，traverse()函数无返回值，思考当前节点需要做什么，并且利用二叉搜索树相关特性
        """
        self.path_sum = 0
        self.traverse(root)
        return root

    def traverse(self,root:TreeNode):
        if not root:
            return 

        self.traverse(root.right)
        self.path_sum += root.val
        root.val = self.path_sum
        self.traverse(root.left)



"""
BST 相关的问题，要么利用 BST 左小右大的特性提升算法效率，要么利用中序遍历的特性满足题目的要求，也就这么些事儿吧。
"""
# 在二叉搜索树上寻找之为target的的节点
class Solution:
    def findInBST(self,root:TreeNode,target:int):
        if not root:
            return -1 
        if root.val == target:
            do_something()
        if root.val < target:
            self.findInBST(root.right)
        if root.val > target:
            self.findInBST(root.left)


# leetcode 98
class Solution:
    def isValidBST(self,root:TreeNode):
        return self.isValidBSTHelper(root,None,None)
    def isValidBSTHelper(self,root:TreeNode,min_node:TreeNode,max_node:TreeNode):
        """
        分解问题思路：定义func，并且充分利用这个定义从子问题推导原始问题,思考当前节点需要做什么
        函数定: 输入根节点root以及以当前根节点的最小节点和最大节点，判断该树是否是二叉搜索树
        """
        if not root:
            return True
        if min_node and root.val <= min_node.val:
            return False
        if max_node and max_node.val <= root.val:
            return False
        return self.isValidBSTHelper(root.left,min_node,root) and self.isValidBSTHelper(root.right,root,max_node)
        

