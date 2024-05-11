from typing import Counter, List

"""有些题目，你按照拍脑袋的方式去做，可能发现需要在递归代码中调用其他递归函数计算字数的信息。
一般来说，出现这种情况时你可以考虑用后序遍历的思维方式来优化算法，
利用后序遍历传递子树的信息，避免过高的时间复杂度。
"""


class TreeNode:
    def __init__(self, val, left=None, right=None) -> None:
        self.val = val
        self.left = left
        self.right = right


# leetcode 110 平衡二叉树
class Solution:
    def isBalanced(self, root: TreeNode):
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

    def maxDepth(self, root: TreeNode):
        if not root:
            return 0
        left_max_depth = self.maxDepth(root.left)
        right_max_depth = self.maxDepth(root.right)
        if abs(left_max_depth - right_max_depth) > 1:
            self.balance = False

        return max(left_max_depth, right_max_depth) + 1


# leetcode 250 统计同值子树
class Solution:
    def countUnivalSubtrees(self, root: TreeNode) -> int:
        if not root:
            # 先保证 root 不为空
            return 0
        self.res = 0
        self.getUnivalue(root)
        return self.res

    # 定义：输入一棵二叉树，如果这棵二叉树的所有节点值都相同，则返回它们的值，
    # 如果这棵二叉树的所有节点的值不是相同的，则返回 -1001。
    # （因为题目说节点的正常取值为 [-1000, 1000]，所以 -1001 是个特殊值）
    def getUnivalue(self, root: TreeNode) -> int:
        # 先算出左右子树的值是否全部相同
        left = root.left.val if root.left else root.val
        right = root.right.val if root.right else root.val
        if root.left:
            left = self.getUnivalue(root.left)
        if root.right:
            right = self.getUnivalue(root.right)
        # 如果有任何一棵子树的值不相同，那么以 root 为根的这棵树的值肯定不可能全部相同
        if left == -1001 or right == -1001:
            return -1001
        # 如果左右子树的值都相同，且等于 root.val，
        # 则说明以 root 为根的二叉树是一棵所有节点都相同的二叉树
        if left == right and root.val == left:
            # 给全局变量 res 加一
            self.res += 1
            return root.val
        # 否则，以 root 为根的二叉树不是一棵所有节点都相同的二叉树
        return -1001


# leetcode 333. 最大 BST 子树
class Solution:
    def largestBSTSubtree(self, root: TreeNode):
        """
        根据 BST 的特性：左子树的所有节点都要比根节点小，右子树的所有节点都要比根节点大，所以可以根据左子树的最大值和右子树的最小值来判断 BST 的合法性。
        同时，题目让计算最大的 BST 的节点个数，所以子树还应该回传节点总数。

        # 定义：输入一棵二叉树，如果这棵二叉树不是 BST，则返回 None，
        # 如果这棵树是 BST，则返回三个数：
        # 第一个数是这棵 BST 中的最小值，
        # 第二个数是这棵 BST 中的最大值，
        # 第三个数是这棵 BST 的节点总数
        """
        self.res = 0
        self.findBST(root)

        return self.res

    def findBST(self, root: TreeNode):
        if not root:
            return [float("-inf"), float("inf"), 0]
        left = self.findBST(root.left)
        right = self.findBST(root.right)
        # 后序位置，根据左右子树的情况推算以 root 为根的二叉树的情况
        if not left or not right:
            # 如果左右子树如果有一个不是 BST，
            # 则以 root 为根的二叉树也必然不是 BST
            return None
        leftMax, leftMin, leftCount = left
        rightMax, rightMin, rightCount = right
        if root.val > leftMax and root.val < rightMin:
            # 以 root 为根的二叉树是 BST
            rootMin = min(leftMin, root.val)
            rootMax = max(rightMax, root.val)
            rootCount = leftCount + rightCount + 1
            self.res = max(self.res, rootCount)

            return [rootMin, rootMax, rootCount]

        return None


# leecode 366 寻找二叉树的叶子节点
# 正常说的二叉树高度都是从上到下，从根节点到叶子结点递增的，而这题可以需要思考，把二叉树的高度理解成从叶子节点到根节点从下到上递增的，那么把相同高度的节点分到一起就是题目想要的答案。
class Solution:
    def findLeves(self, root: TreeNode):
        self.res = []
        self.maxDepth(root)

        return self.res

    def maxDepth(self, root: TreeNode):
        """
        定义：输入节点 root，返回以 root 为根的树的最大深度
        """
        if not root:
            return 0
        h = max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
        if len(self.res) < h:
            self.res.append([])
        # 把所有相同高度的叶子节点放在一起
        self.res[h - 1].append(root.val)


from collections import defaultdict


# leetcode 508. 出现次数最多的子树元素和
class Solution:
    def __init__(self):
        self.sumtoCount = defaultdict()

    def findFrequentTreeSum(self, root: TreeNode):
        self.sumtoCount(root)
        maxCount = 0
        for count in self.sumToCount.values():
            maxCount = max(maxCount, count)
        # 找到最大出现频率对应的的子树和
        res = []
        for key in self.sumToCount.keys():
            if self.sumToCount[key] == maxCount:
                res.append(key)
        return res

    def sumOfTree(self, root: TreeNode):
        if not root:
            return 0
        res = self.sumOfTree(root.left) + self.sumOfTree(root.right) + root.val
        self.sumtoCount[res] = self.sumtoCount.get(res, 0) + 1

        return res
