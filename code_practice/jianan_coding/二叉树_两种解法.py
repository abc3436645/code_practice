from typing import List


class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# leetcdode 112 路径总和
class Solution:
    def hasPathSum(self, root: TreeNode, targetSum: int):
        """
        遍历思路解法：使用外部变量和traverse()
        """
        self.found = False
        self.pathSum = 0
        self.targetSum = targetSum
        self.traverse(root)
        return self.found

    def traverse(self, root: TreeNode):
        if not root:
            return

        self.pathSum += root.val
        if not root.left and not root.right:
            if self.pathSum == self.targetSum:
                self.found = True
                return
        self.traverse(root.left)
        self.traverse(root.right)
        self.pathSum -= root.val


class Solution:
    def hasPathSum(self, root: TreeNode, targetSum: int):
        """
        分解问题思路：使用递归函数的定义，把用子问题的结果推导出大问题的结果
        函数定义：输入一个根节点，返回该根节点到叶子节点是否存在一条和为 targetSum 的路径
        """
        if not root:
            return False
        if not root.left and not root.right and root.val == targetSum:
            return True
        left_path = self.hasPathSum(root.left, targetSum - root.val)
        right_path = self.hasPathSum(root.right, targetSum - root.val)

        return left_path or right_path


# 113. 路径总和 II
# 给你二叉树的根节点 root 和一个整数目标和 targetSum，找出所有从根节点到叶子节点路径总和等于给定目标和的路径。
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int):
        """
        遍历解法:使用外部变量和traverse()函数,函数无返回值
        """
        self.res = []
        self.path = []
        self.traverse(root, targetSum)
        return self.res

    def traverse(self, root: TreeNode, targetSum: int):
        if not root:
            return
        self.path.append(root.val)
        if not root.left and not root.right and sum(self.path) == targetSum:
            self.res.append(self.path[:])
        self.traverse(root.left, targetSum)
        self.traverse(root.right, targetSum)
        self.path.pop()


class Solution:
    def pathSum(self, root: TreeNode, targetSum: int):
        """
        分解问题思路：输入一个根节点，返回以该根节点到叶子节点的路径和为targetSum的所有路径
        """
        root_answers = []
        if root is None:
            return root_answers

        # 如果是叶子节点并且值等于 targetSum，则找到一条路径
        if root.left is None and root.right is None and root.val == targetSum:
            path = [root.val]
            root_answers.append(path)
            return root_answers

        # 分别递归左右子树，找到子树中和为 targetSum - root.val 的路径
        left_answers = self.pathSum(root.left, targetSum - root.val)
        right_answers = self.pathSum(root.right, targetSum - root.val)

        # 左右子树的路径加上根节点，就是和为 targetSum 的路径
        for answer in left_answers:
            # 因为底层使用的是 list，所以这个操作的复杂度是 O(1)
            answer.insert(0, root.val)
            root_answers.append(answer)
        for answer in right_answers:
            # 因为底层使用的是 list，所以这个操作的复杂度是 O(1)
            answer.insert(0, root.val)
            root_answers.append(answer)

        return root_answers


# leetcode 897. 递增顺序搜索树
# 给你一棵二叉搜索树，请你按中序遍历将其重新排列为一棵递增顺序搜索树，使树中最左边的节点成为树的根节点，并且每个节点没有左子节点，只有一个右子节点。
class Solution:
    def increasingBST(self, root: TreeNode):
        self.res = []
        self.traverse(root)
        return self.build_tree(self.res)

    def traverse(self, root: TreeNode):
        if not root:
            return
        self.traverse(root.left)
        self.res.append(root.val)
        self.traverse(root.right)

    def build_tree(self, tree_list: List[TreeNode]):
        if not tree_list:
            return None
        root = TreeNode(val=tree_list[0])
        p = root
        n = 1
        while n < len(tree_list):
            tmp = TreeNode(val=tree_list[n])
            p.left = None
            p.right = tmp
            p = tmp
            n += 1
        return root


class Solution:
    def increasingBST(self, root: TreeNode):
        """
        分解问题思路：// 输入一棵 BST，返回一个有序「链表」
        """
        if not root:
            return None

        left = self.increasingBST(root.left)
        right = self.increasingBST(root.right)
        root.left = None
        root.right = right
        if not left:
            return root
        p = left
        while p and p.right:
            p = p.right
        p.right = root

        return left


# leetcode 938. 二叉搜索树的范围和
# 给定二叉搜索树的根结点 root，返回值位于范围 [low, high] 之间的所有结点的值的和
class Solution:
    def rangeSumBST(self, root: TreeNode, low: int, high: int):
        self.res = 0
        self.traverse(root, low, high)

        return self.res

    def traverse(self, root: TreeNode, low: int, high: int):
        if not root:
            return
        if low <= root.val and root.val <= high:
            self.res += root.val

        self.traverse(root.left, low, high)
        self.traverse(root.right, low, high)


class Solution:
    def rangeSumBST(self, root: TreeNode, low: int, high: int):
        """
        分解问题：给定root返回范围[low,high]之前所有节点的和
        """
        if not root:
            return 0
        if root.val < low:
            return self.rangeSumBST(root.right, low, high)
        if root.val > high:
            return self.rangeSumBST(root.left, low, high)
        else:
            return (
                root.val
                + self.rangeSumBST(root.left, low, high)
                + self.rangeSumBST(root.right, low, high)
            )


# leetcode 1379. 找出克隆二叉树中的相同节点
# 给你两棵二叉树，原始树 original 和克隆树 cloned，以及一个位于原始树 original 中的目标节点 target。
# 其中，克隆树 cloned 是原始树 original 的一个副本。请找出在树 cloned 中，与 target 对应的节点，并返回对该节点的引用。
class Solution:
    def getTargetCopy(self, original: TreeNode, cloned: TreeNode, target: TreeNode):
        self.res = None
        self.target = target
        self.traverse(original, cloned)

        return self.res

    def traverse(self, original: TreeNode, cloned: TreeNode):
        """
        遍历思路解法：使用外部变量和traverse(),traverse()函数无返回值
        """
        if not original or self.res:
            return
        if original == self.target:
            self.res = cloned
            return

        self.traverse(original.left, cloned.left)
        self.traverse(original.right, cloned.right)


class Solution:
    def getTargetCopy(self, original: TreeNode, cloned: TreeNode, target: TreeNode):
        """
        分解问题思路： 定义：找到 original 中 target 节点在 cloned 树中对应的节点
        """
        if not original:
            return None
        if original == target:
            return cloned
        left = self.getTargetCopy(original.left, cloned.left, target)

        return (
            left if left else self.getTargetCopy(original.right, cloned.right, target)
        )


# leetcode 1430. 判断给定的序列是否是二叉树从根到叶的路径
class Solution:
    def isValidSequence(self, root: TreeNode, arr: List[int]) -> bool:
        return self.check(root, arr, 0)

    # 定义：输入一棵根为 root 的二叉树，
    # 判断是否存在一条从根到叶子节点的路径的值为 arr[i..]
    def check(self, root: TreeNode, arr: List[int], i: int) -> bool:
        if not root or i == len(arr):
            return False
        if not root.left and not root.right:
            # 到达叶子结点，判断是否同时到达数组末端
            return arr[i] == root.val and i == len(arr) - 1
        if root.val != arr[i]:
            return False
        # 如果 root.val == arr[i]，则判断子树是否存在一条路径值为 arr[i+1..]
        return self.check(root.left, arr, i + 1) or self.check(root.right, arr, i + 1)


class Solution:
    def isValidSequence(self, root: TreeNode, arr: List[int]) -> bool:
        self.arr = arr
        self.traverse(root)
        return self.isValid

    # 记录遍历深度（函数的索引）
    d = 0
    arr = []
    isValid = False

    # 二叉树遍历函数
    def traverse(self, root: TreeNode) -> None:
        if not root or self.isValid:
            return

        if not root.left and not root.right:
            # 到达叶子结点，判断是否同时到达数组末端
            if self.d == len(self.arr) - 1 and self.arr[self.d] == root.val:
                self.isValid = True
            return

        if self.d >= len(self.arr) or self.arr[self.d] != root.val:
            return

        self.d += 1
        # 二叉树遍历框架
        self.traverse(root.left)
        self.traverse(root.right)
        self.d -= 1
