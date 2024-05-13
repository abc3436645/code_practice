from multiprocessing import parent_process
from types import prepare_class
from typing import Counter, List

from numpy import left_shift, right_shift

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


# leetcode 508. 出现次数最多的子树元素和
class Solution:
    def __init__(self):
        self.sumtoCount = {}

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


# leetcode 563 二叉树的坡度
"""
一个树的节点的坡度定义即为，该节点左子树的节点之和和右子树节点之和的差的绝对值。如果没有左子树的话，左子树的节点之和为 0，没有右子树的话也是一样，空结点的坡度是 0。
"""


class Solution:
    def findTilt(self, root: TreeNode):
        self.res = 0
        self.sumOfTree(root)
        return self.res

    def sumOfTree(self, root: TreeNode):
        """
        sum 函数记录二叉树的节点之和，在后序位置顺便计算二叉树的「坡度」即可
        """
        if not root:
            return 0
        left = self.sumOfTree(root.left)
        right = self.sumOfTree(root.right)
        self.res += abs(left - right)  # 顺便求出坡度

        return left + right + root.val


# leetcode 549 二叉树中最长连续序列
class Solution:
    def longestConsecutive(self, root: TreeNode):
        self.res = 0
        self.findSequence(root)
        return self.res

    def findSequence(self, root: TreeNode):
        if not root:
            return [0, 0]
        left = self.findSequence(root.left)
        right = self.findSequence(root.right)

        leftIncrLen, leftDrecLen = left[0], left[0]
        rightIncrLen, rightDecrLen = right[0], right[1]

        rootIncrLen, rootDecrLen = 1, 1
        if root.left:
            if root.left.val - 1 == root.val:
                rootIncrLen += leftIncrLen
            if root.left.val + 1 == root.val:
                rootDecrLen += leftDrecLen

        if root.right:
            if root.right.val - 1 == root.val:
                rootIncrLen = max(rootIncrLen, rightIncrLen + 1)
            if root.right.val + 1 == root.val:
                rootDecrLen = max(rootDecrLen, rightDecrLen + 1)

        # 子树递增序列 + 根节点 + 子树递减序列 = 题目要求的序列
        # 因为 rootIncrLen 和 rootDecrLen 中都算上了 root 节点，所以减一
        self.res = max(self.res, rootIncrLen + rootDecrLen - 1)

        return [rootIncrLen, rootDecrLen]


# leetcode 1325 删除给定值的叶子结点
class Solution:
    def removeLeafNodes(self, root: TreeNode, target: int):
        if not root:
            return None

        root.left = self.removeLeafNodes(root.left)
        root.right = self.removeLeafNodes(root.right)
        if root.val == target and not root.left and not root.right:
            return None

        return root


# leetcode 663 均匀树划分
"""
给定一棵有 n 个结点的二叉树，你的任务是检查是否可以通过去掉树上的一条边将树分成两棵，且这两棵树结点之和相等

解题思路：如果我知道了整棵树所有节点之和 treeSum，那么如果存在一棵和为 treeSum / 2 的子树，就说明这棵二叉树是可以分解成两部分的
"""


class Solution:
    def __init__(self) -> None:
        self.subTreeSum = set()

    def checkEqualTree(self, root: TreeNode):
        tree_sum = root.val + self.sumOfTree(root.left) + self.sumOfTree(root.right)
        if tree_sum % 2 != 0:
            return False
        return tree_sum // 2 in self.subTreeSum

    def sumOfTree(self, root: TreeNode):
        if not root:
            return 0
        left = self.sumOfTree(root.left)
        right = self.sumOfTree(root.right)
        rootSum = left + right + root.val

        self.subTreeSum.add(rootSum)

        return rootSum


# leetcode 687 最长同值路径
class Solution:
    def __init__(self) -> None:
        self.res = 0

    def longestUnivaluePath(self, root: TreeNode):
        if not root:
            return 0
        self.maxLen(root, root.val)
        return self.res

    def maxLen(self, root: TreeNode, parentVal: int):
        """
        定义：计算以 root 为根的这棵二叉树中，从 root 开始值为 parentVal 的最长树枝长度
        """
        if not root:
            return 0
        leftLen = self.maxLen(root.left, root.val)
        rightLen = self.maxLen(root.right, root.val)
        self.res = max(self.res, leftLen + rightLen)
        if root.val != parentVal:
            return 0

        return max(leftLen, rightLen) + 1


# leetcode 1026 节点与其祖先之间的最大差值
class Solution:
    def maxAncestorDiff(self, root: TreeNode):
        self.res = 0
        self.getMaxMin(root)
        return self.res

    def getMaxMin(self, root: TreeNode):
        if not root:
            return [float("inf"), float("-inf")]

        left = self.getMaxMin(root.left)
        right = self.getMaxMin(root.right)

        rootMin = min(root.val, left[0], right[0])
        rootMax = max(root.val, left[1], right[1])

        self.res = max(self.res, abs(root.val - rootMin), abs(root.val - rootMax))

        return [rootMin, rootMax]


# leetcode 1120 子树的最大平均值
"""
那么我们站在一个节点上，需要知道子树的什么信息，才能计算出以这个节点为根的这棵树的平均值呢？

显然，我们需要知道子树的节点个数和所有节点之和，就能算出平均值了
"""


class Solution:
    def maximumAverageSubtree(self, root: TreeNode):
        self.res = 0
        self.getCountAndSum(root)

        return self.res

    def getCountAndSum(self, root: TreeNode):
        if not root:
            return [0, 0]

        left = self.getCountAndSum(root.left)
        right = self.getCountAndSum(root.right)

        leftCount, leftSum = left[0], left[1]
        rightCount, rightSum = right[0], right[1]

        rootCount = leftCount + rightCount + 1
        rootSum = leftSum + rightSum + root.val

        self.res = max(self.res, 1.0 * (rootSum / rootCount))

        return [rootCount, rootSum]
