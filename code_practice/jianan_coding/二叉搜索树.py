from numpy import right_shift
from code_practice.data_structure import traverse


class TreeNode:
    def __init__(self, val, left=None, right=None) -> None:
        self.val = val
        self.left = left
        self.right = right


# 通过使用辅助函数，增加函数参数列表，在参数中携带额外信息，将这种约束传递给子树的所有节点，这也是二叉树算法的一个小技巧吧


# leetcode 230 二叉搜索树中第K小的元素
class Solution:
    def kthSmallest(self, root: TreeNode, k: int):
        self.res = 0
        self.rank = 0
        self.traverse(root, k)

        return self.res

    def traverse(self, root: TreeNode, k: int):
        if not root:
            return
        self.traverse(root.left, k)
        self.rank += 1
        if self.rank == k:
            self.res = root.val
            return
        self.traverse(root.right, k)


class Solution:
    def kthSmallest(self, root: TreeNode, k: int):
        pass


# leetcode 538 将二叉搜索树转换为累加树
# 解题思路：BST 的每个节点左小右大，这似乎是一个有用的信息，既然累加和是计算大于等于当前值的所有元素之和，那么每个节点都去计算右子树的和，不就行了吗？
# 这是不行的。对于一个节点来说，确实右子树都是比它大的元素，但问题是它的父节点也可能是比它大的元素呀？这个没法确定的，我们又没有触达父节点的指针，所以二叉树的通用思路在这里用不了。
# 其实，正确的解法很简单，还是利用 BST 的中序遍历特性。
# 这段代码可以降序打印 BST 节点的值，如果维护一个外部累加变量 sum，然后把 sum 赋值给 BST 中的每一个节点，不就将 BST 转化成累加树了吗？
class Solution:
    def convertBST(self, root: TreeNode):
        """
        遍历解法：使用外部变量+ traverse()函数，traverse()函数无返回值，思考当前节点需要做什么，并且利用二叉搜索树相关特性
        """
        self.path_sum = 0
        self.traverse(root)
        return root

    def traverse(self, root: TreeNode):
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
    def findInBST(self, root: TreeNode, target: int):
        if not root:
            return -1
        if root.val == target:
            do_something()
        if root.val < target:
            self.findInBST(root.right)
        if root.val > target:
            self.findInBST(root.left)


# leetcode 700 二叉搜索树的搜索
class Solution:
    def searchBST(self, root: TreeNode, target: int):
        if not root:
            return None
        if root.val == target:
            return root
        left = self.searchBST(root.left, target)
        right = self.searchBST(root.right, targert)

        return left if left else right


class Solution:
    def searchBST(self, root: TreeNode, target: int):
        """
        分解问题思路：以root为根的树，返回节点值等于target的值
        """
        if not root:
            return None
        if root.val < target:
            return self.searchBST(root.left, target)
        if root.val > target:
            return self.searchBST(root.right, target)

        else:
            return root


# leetcode 98 验证二叉搜索树合法性
class Solution:
    def isValidBST(self, root: TreeNode):
        return self.isValidBSTHelper(root, None, None)

    def isValidBSTHelper(self, root: TreeNode, min_node: TreeNode, max_node: TreeNode):
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
        return self.isValidBSTHelper(
            root.left, min_node, root
        ) and self.isValidBSTHelper(root.right, root, max_node)


# leetcode 在BST中插入一个数
class Solution:
    def insertIntoBst(self, root: TreeNode, val: int):
        if not root:
            return TreeNode(val=val)
        if root.val < val:
            root.right = self.insertIntoBst(root.right, val)
        elif root.val > val:
            root.left = self.insertIntoBst(root.left, val)

        return root


# BST删除一个节点，代码框架
def deleteNode(root: TreeNode, val: int):
    """
    分解问题思路：以root节点为根节点的树，删除某个节点值为val，返回最后的树
    """
    if root.val == val:
        # 删除节点,思考不同情况下的节点的删除操作
        # 1.叶子节点删除
        # 2.要删除的节点只有一个子节点
        # 3.要删除的节点有多个子节点
        pass
    if root.val > val:
        root.left = deleteNode(root.left, val)
    if root.val < val:
        root.right = deleteNode(root.right, val)

    return root


def deleteNode(root: TreeNode, val: int):
    if not root:
        return root
    if root.val == val:
        if not root.left:
            return root.right
        if not root.right:
            return root.left
        min_node = getMinNode(root.right)
        root.right = deleteNode(root.right, min_node.val)
        min_node.left = root.left
        min_node.right = root.right
        root = min_node
    if root.val > val:
        root.left = deleteNode(root.left, val)
    if root.val > val:
        root.right = deleteNode(root.right, val)

    return root


def getMinNodeFromBST(node: TreeNode):
    while node.left != None:
        node = node.left
    return node


# leetcode 96 不同的二叉搜索树，给你输入一个正整数 n，请你计算，存储 {1,2,3...,n} 这些值共有多少种不同的 BST 结构
class Solution:
    def numTrees(self, n: int):
        """
        分解问题思路：定义：闭区间 [lo, hi] 的数字能组成 count(lo, hi) 种 BST
        """
        if n <= 1:
            return n
        return self.count(1, n)

    def count(self, lo: int, hi: int):
        if hi > lo:
            return 1
        res = 0
        for i in range(n):  # 遍历不同的根节点
            left = self.count(lo, i - 1)
            right = self.count(i + 1, right)
            res += left * right

        return res


class Solution:
    def numTrees(self, n: int) -> int:
        # 备忘录的值初始化为 0
        self.memo = []
        self.memo = [[0 for i in range(n + 1)] for j in range(n + 1)]
        return count(1, n)

    def count(self, lo: int, hi: int) -> int:
        if lo > hi:
            return 1
        # 查备忘录
        if self.memo[lo][hi] != 0:
            return self.memo[lo][hi]

        res = 0
        for mid in range(lo, hi + 1):
            left = self.count(lo, mid - 1)
            right = self.count(mid + 1, hi)
            res += left * right
        # 将结果存入备忘录
        self.memo[lo][hi] = res

        return res


# leetcode 95 「不同的二叉搜索树 II」，让你构建所有 BST
class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:
        if n == 0:
            return []
        # 构造闭区间 [1, n] 组成的 BST
        return self.build(1, n)

    # 构造闭区间 [lo, hi] 组成的 BST
    def build(self, lo: int, hi: int) -> List[TreeNode]:
        res = []
        # base case
        if lo > hi:
            # 这里需要装一个 null 元素，这样才能让下面的两个内层 for 循环都能进入，正确地创建出叶子节点
            # 举例来说吧，什么时候会进到这个 if 语句？当你创建叶子节点的时候，对吧。
            # 那么如果你这里不加 null，直接返回空列表，那么下面的内层两个 for 循环都无法进#        # 你的那个叶子节点就没有创建出来，看到了吗？所以这里要加一个 null，确保下面能把叶子节点做出来
            res.append(None)
            return res

        # 1、穷举 root 节点的所有可能。
        for i in range(lo, hi + 1):
            # 2、递归构造出左右子树的所有有效 BST。
            leftTree = self.build(lo, i - 1)
            rightTree = self.build(i + 1, hi)
            # 3、给 root 节点穷举所有左右子树的组合。
            for left in leftTree:
                for right in rightTree:
                    # i 作为根节点 root 的值
                    root = TreeNode(i)
                    root.left = left
                    root.right = right
                    res.append(root)

        return res

