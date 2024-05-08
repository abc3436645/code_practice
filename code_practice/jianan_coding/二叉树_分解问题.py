from typing import List


class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val  
        self.left = left
        self.right = right


class NTreeNode:
    def __init__(self, val: int, children: List[TreeNode]):
        self.val = val
        self.children = children


"""
二叉树解题的思维模式分两类：

1、是否可以通过遍历一遍二叉树得到答案？如果可以，用一个 traverse 函数配合外部变量来实现，这叫「遍历」的思维模式。

2、是否可以定义一个递归函数，通过子问题（子树）的答案推导出原问题的答案？如果可以，写出这个递归函数的定义，并充分利用这个函数的返回值，这叫「分解问题」的思维模式。

无论使用哪种思维模式，你都需要思考：

如果单独抽出一个二叉树节点，它需要做什么事情？需要在什么时候（前/中/后序位置）做？其他的节点不用你操心，递归函数会帮你在所有节点上执行相同的操作。
"""

# leetcode 二叉树的最大深度
class Solution:
    def MaxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        left = self.MaxDepth(root.left)
        right = self.MaxDepth(root.right)
        return max(left, right) + 1


# leetcode 二叉树最小深度
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        if not root.left and not root.right:
            return 1
        min_depth = float("inf")
        if root.left:
            min_depth = min(self.minDepth(root.left), min_depth)
        if root.right:
            min_depth = min(self.minDepth(root.right), min_depth)
        return min_depth + 1


class Solution:
    def minDepth(self, root: TreeNode):
        if not root:
            return 0
        if root.left is None and root.right is not None:
            return self.minDepth(root.right) + 1
        if root.right is None and root.left is not None:
            return self.minDepth(root.left) + 1
        left = self.minDepth(root.left)
        right = self.minDepth(root.right)
        res = min(left, right) + 1
        return res


# leetcode 二叉树最大路径和
class Solution:
    def __init__(self) -> None:
        self.max_sum = float("-inf")

    def maxPathSum(self, root: TreeNode) -> int:
        def dfs(root):
            if not root:
                return 0
            left = max(0, dfs(root.left))
            right = max(0, dfs(root.right))
            self.max_sum = max(self.max_sum, left + right + root.val)
            return max(left, right) + root.val

        dfs(root)
        return self.max_sum


# leetcode 998 最大二叉树 II
# 新增的 val 是添加在原始数组的最后的，根据构建最大二叉树的逻辑，正常来说最后的这个值一定是在右子树的，可以对右子树递归调用 insertIntoMaxTree 插入进去。
# 但是一种特殊情况是 val 比原始数组中的所有元素都大，那么根据构建最大二叉树的逻辑，原来的这棵树应该成为 val 节点的左子树。
class Solution:
    def insertIntoMaxTree(self, root: TreeNode, val: int):
        if not root:
            return TreeNode(val=val)
        if root.val > val:
            # 如果 val 不是最大的，那么就应该在右子树上，
            # 因为 val 节点是接在原始数组 a 的最后一个元素
            root.right = self.insertIntoMaxTree(root.right, val)
        elif root.val <= val:
            tmp = root
            root = TreeNode(val=val)
            root.left = tmp

        return root


# leetcode 1110 删点成林
# 给出二叉树的根节点 root，树上每个节点都有一个不同的值。如果节点值在 to_delete 中出现，我们就把该节点从树上删去，
# 最后得到一个森林（一些不相交的树构成的集合）。返回森林中的每棵树，你可以按任意顺序组织答案
class Solution:
    def __init__(self):
        self.res = []
        self.deleteSet = set()

    def delNodes(self, root: TreeNode, to_delete: List[int]):
        if not root:
            return []
        for det in to_delete:
            self.deleteSet.add(det)

        self.deleteNode(root, False)

        return self.res

    def deleteNode(self, root: TreeNode, hasParent: bool):
        """
        定义：输入一棵二叉树，删除 delSet 中的节点，返回删除完成后的根节点
        """
        if not root:
            return

        deleted = root.val in self.deleteSet
        if not deleted and not hasParent:
            self.res.append(root)
        root.left = self.deleteNode(root.left, not deleted)
        root.right = self.deleteNode(root.right, not deleted)

        return root if not deleted else None


# leetcode 1660 纠正二叉树
# 你有一棵二叉树，这棵二叉树有个小问题，其中有且只有一个无效节点，它的右子节点错误地指向了与其在同一层且在其右侧的一个其他节点。
# 给定一棵这样的问题二叉树的根节点 root，将该无效节点及其所有子节点移除（除被错误指向的节点外），然后返回新二叉树的根结点。
#
# 解题思路：
# 如何知道一个节点 x 是「错误」节点？需要要知道它的右子节点错误地指向同层右侧的一另一个节点。那么如何让节点 x 知道自己错指了呢？
# 如果用层序遍历的方式应该比较容易想到解法，我们可以从右向左遍历每一行的节点并记录下来访问过的节点，如果你发现某个节点 x 的右子节点已经被访问过，那么这个节点 x 就是错误节点。
# 如果用递归遍历的方式，如何做到「从右向左」遍历每一行的节点呢？只要稍微改一下二叉树的遍历框架，先遍历右子树后遍历左子树，这样就是先遍历右侧节点后遍历左侧节点，也就相当于从右向左遍历了。
# 同理，遍历的时候我们同时记录已遍历的节点。这样 x 如果发现自己的右子节点已经被访问了，就说明 x 节点是「错误」的
class Solution:
    def __init__(self):
        self.visited = set()

    def correctBinaryTree(self, root: TreeNode):
        if not root:
            return None
        if root.right and root.right in self.visited:
            # 找到「无效节点」，删除整棵子树
            return None
        self.visited.add(root)
        root.right = self.correctBinaryTree(root.right)
        root.left = self.correctBinaryTree(root.left)

        return root


# leetcode 100 相同的树
# 输入：p = [1,2,3], q = [1,2,3]
# 输出：true
class Solution:
    def isSameTree(self, root1: TreeNode, root2: TreeNode):
        if root1 and not root2:
            return False
        if not root1 and root2:
            return False
        if not root1 and not root2:
            return True
        if root1.val != root2.val:
            return False
        return self.isSameTree(root1.left, root2.left) and self.isSameTree(
            root1.right, root2.right
        )


# leetcode 101 对称二叉树
class Solution:
    def isSymmetric(self, root: TreeNode):
        return self.check(root.left, root.right)

    def check(self, root1: TreeNode, root2: TreeNode):
        if not root1 and root2:
            return False
        if root1 and not root2:
            return False
        if not root1 and not root2:
            return True
        if root1.val != root2.val:
            return False
        return self.check(root1.left, root2.right) and self.check(
            root1.right, root2.left
        )


# leetcode 951 翻转等价二叉树
class Solution:
    def flipEquiv(self, root1: TreeNode, root2: TreeNode):
        if not root1 and root2:
            return False
        if root1 and not root2:
            return False
        if not root1 and not root2:
            return True
        if root1.val != root2.val:
            return False
        return (
            self.flipEquiv(root1.left, root2.left)
            and self.flipEquiv(root1.right, root2.right)
        ) or (
            self.flipEquiv(root1.left, root2.right)
            and self.flipEquiv(root1.right, root2.left)
        )


# leetcode 124 二叉树中的最大路径和
class Solution:
    def __init__(self):
        self.res = float("-inf")

    def maxPathSum(self, root: TreeNode):
        if not root:
            return 0
        self.maxPathSum(root)
        return self.res

    def oneSideSum(self, root: TreeNode):
        if not root:
            return 0
        left_side_sum = max(0, self.oneSideSum(root.left))
        right_side_sum = max(0, self.oneSideSum(root.right))
        my_side_sum = left_side_sum + right_side_sum + root.val
        self.res = max(self.res, my_side_sum)

        return max(left_side_sum, right_side_sum) + root.val
    
# leetcode 814 二叉树剪枝
# 给你二叉树的根结点 root ，此外树的每个结点的值要么是 0 ，要么是 1 。
# 返回移除了所有不包含 1 的子树的原二叉树。
# 节点 node 的子树为 node 本身加上所有 node 的后代。
# 解题思路：这道题的难点在于要一直剪枝，直到没有值为 0 的叶子节点为止，只有从后序遍历位置自底向上处理才能获得最高的效率。
class Solution:
    def pruneTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None

        root.left = self.pruneTree(root.left) 
        root.right = self.pruneTree(root.right)
        if root.val == 0:
            if not root.left and not root.right:
                return None
        return root 